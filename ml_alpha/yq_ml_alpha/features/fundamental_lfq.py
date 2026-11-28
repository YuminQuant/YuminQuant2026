"""Point-in-time fundamental inputs for the daily ASTGNN factor."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json

import numpy as np
import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_daily, read_keyed_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.features.transforms import zscore_observed
from yq_ml_alpha.output import artifact_io


class FundamentalLfqHistory(DailyTensorSequence):
    def __init__(self, config):
        params = config.features.params
        columns = config.features.columns
        if len(columns) != 56 or len(set(columns)) != 56:
            raise ValueError("fundamental LFQ requires 56 distinct metric columns")
        self.metrics = list(columns)
        self.history_start = int(params["history_start_date"])
        self.normalization = params.get("normalization", "raw")
        if self.normalization not in {"raw", "zscore", "mad5_zscore"}:
            raise ValueError("invalid fundamental LFQ input normalization")
        # Numeric zero and missing source values remain distinguishable.
        inputs = self.fundamental_columns = self.metrics + [f"{column}__missing" for column in self.metrics]
        super().__init__(config.features.root, inputs, (len(inputs),),
                         length=int(config.model.params["sequence_length"]))
        self.filters = TradeFilters(config.filters, config.data_root)
        self.early_filters = TradeFilters(replace(config.filters, exclude_st=False), config.data_root)
        self.universe = Universe(config.universe.id, config.data_root)

    def window_dates(self, date):
        window = super().window_dates(date)
        return window if window and window[0] >= self.history_start else []

    def prefetch_targets(self, dates):
        if artifact_io.backend(self.root) is None or list(dates) != sorted(dates):
            return nullcontext()
        return self.prefetch_sources(sorted({day for date in dates for day in self.window_dates(date)}))

    def _load_day(self, date):
        frame = read_daily(self.root, date, self.metrics, required=self.metrics)
        if (frame.empty or frame.ts_code.isna().any() or frame.ts_code.duplicated().any()
                or not frame.trade_date.eq(date).all()):
            raise ValueError(f"invalid, missing or duplicate fundamental LFQ inputs for {date}: {self.root}")
        filters = self.filters if date >= 20160101 else self.early_filters
        frame = filters.apply(self.universe.filter(frame, date), date)
        values = frame[self.metrics].to_numpy(dtype=np.float32, copy=False)
        valid = np.isfinite(values)
        output = np.empty((len(frame), len(self.fundamental_columns)), dtype=np.float32)
        if self.normalization == "raw":
            output[:, :len(self.metrics)] = np.where(valid, values, 0)
        else:
            output[:, :len(self.metrics)] = zscore_observed(values, mad5=self.normalization == "mad5_zscore")
        output[:, len(self.metrics):] = ~valid
        return pd.DataFrame(output, index=pd.Index(frame.ts_code, name="ts_code"),
                            columns=self.fundamental_columns, copy=False)


class FundamentalLfqProvider(DailySequenceProvider):
    def __init__(self, config, *, history=None):
        super().__init__(history if history is not None else FundamentalLfqHistory(config))
        self._manifest = None

    def set_calendar_dates(self, dates):
        self.dataset_identity()
        super().set_calendar_dates(dates)

    def dataset_identity(self):
        if self._manifest is None:
            path = self.sequence.root / "manifest.json"
            expected = {"version": "astgnn_lfq_daily_v1", "columns": self.sequence.metrics,
                        "profit_proxy": "n_income_attr_p", "frequency": "daily",
                        "preprocess": "raw_fundamental_metrics"}
            if not artifact_io.exists(path):
                raise FileNotFoundError(f"missing fundamental LFQ manifest: {path}; run derive-fundamental-lfq")
            if json.loads(artifact_io.read_text(path, encoding="utf-8")) != expected:
                raise ValueError(f"incompatible fundamental LFQ dataset manifest: {path}")
            self._manifest = {**expected, "missing_input": "zero_with_indicators"}
            if self.sequence.normalization != "raw":
                self._manifest["model_input_normalization"] = self.sequence.normalization
        return self._manifest


class FundamentalStyleHistory(FundamentalLfqHistory):
    """Point-in-time metrics and daily Barra exposures on the same stock keys."""

    time_major_batch = True

    def __init__(self, config):
        super().__init__(config)
        params = config.features.params
        self.barra_columns = list(params["barra_columns"])
        inputs = self.columns + self.barra_columns
        if "SIZE" not in self.barra_columns or len(set(inputs)) != len(inputs):
            raise ValueError("fundamental/style inputs require distinct Barra columns")
        self.barra_root = config.data_root / "barra" / "stock" / "daily" / params["barra_model"]
        self.industry_path = config.data_root / "index_data" / "member_ci" / "ci_members.parquet"
        self.size_column = self.barra_columns.index("SIZE")
        self.size_exposures = {}
        self.columns = inputs
        self.shape = (len(self.columns),)

    def _load_day(self, date):
        fundamental = super()._load_day(date)
        barra = read_keyed_daily(self.barra_root, date, self.barra_columns).set_index("ts_code")
        values = barra.reindex(fundamental.index)[self.barra_columns].to_numpy(dtype=np.float32, na_value=np.nan)
        self.size_exposures[date] = pd.Series(values[:, self.size_column].copy(), index=fundamental.index)
        fundamental[self.barra_columns] = zscore_observed(values, mad5=False)
        return fundamental

    def close(self):
        super().close()
        self.size_exposures.clear()


class FundamentalStyleProvider(FundamentalLfqProvider):
    def __init__(self, config):
        super().__init__(config, history=FundamentalStyleHistory(config))

    def dataset_identity(self):
        return {**super().dataset_identity(), "barra_columns": self.sequence.barra_columns,
                "barra_root": str(self.sequence.barra_root), "barra_preprocess": "zscore_observed_then_zero"}
