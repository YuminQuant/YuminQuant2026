"""Daily normalized factor snapshots shared by graph and temporal encoders."""
from yq_ml_alpha.output import artifact_io as data_io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.data.factor_files import factor_dates
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.stock_relations import StockRelations
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.transforms import apply_cross_section_transform


class FactorHistory(DailyTensorSequence):
    def __init__(self, config, provider, sequence_length=None, memory_budget_mb=None):
        params = config.features.params
        super().__init__(config.features.root, provider.feature_columns, (len(provider.feature_columns),),
                         int(params.get("sequence_length", 40) if sequence_length is None else sequence_length),
                         int(params.get("memory_budget_mb", 512) if memory_budget_mb is None else memory_budget_mb))
        read_workers = int(params.get("history_read_workers", 1))
        if read_workers < 1:
            raise ValueError("history_read_workers must be positive")
        # Parallel history IO is for remote storage; local reads stay serial.
        self.read_workers = read_workers if data_io.remote_enabled() else 1
        self._read_pool = None
        self.config = config
        self.provider = provider
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self.fill_value = config.preprocess.feature_fill_value
        self.history_transform = str(params.get("history_transform", "mad5_zscore_rank_label"))
        if self.history_transform not in {"mad5_zscore_rank_label", "zscore_clip3"}:
            raise ValueError("unsupported factor history_transform")
        self.available_dates = factor_dates(self.root)
        if not self.available_dates:
            raise ValueError(f"no factor history under {self.root}")
        self.first_date = self.available_dates[0]

    def day(self, date):
        cached = self._cached(date)
        if cached is not None:
            return cached
        return self._remember(date, self._load_day(date))

    def _load_day(self, date):
        files = getattr(self.provider, "files", None)
        if not (files.exists(date) if files is not None else data_io.exists(daily_path(self.root, date))):
            raise FileNotFoundError(f"missing factor history for {date}")
        frame = self.provider.load(date)
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid factor snapshot keys on {date}")
        frame = self.filters.apply(self.universe.filter(frame, date), date)
        frame = self._filter_frame(frame, date)
        # Normalize before label joining. Missing future labels never change graph inputs.
        frame[self.columns] = frame[self.columns].apply(pd.to_numeric, errors="coerce").astype("float64")
        frame = apply_cross_section_transform(
            frame, self.history_transform, self.columns, [], self.fill_value,
        ).set_index("ts_code")[self.columns].astype("float32")
        return frame

    def _days(self, dates):
        if self.read_workers == 1:
            yield from super()._days(dates)
            return
        if self._read_pool is None:
            self._read_pool = ThreadPoolExecutor(max_workers=self.read_workers,
                                                 thread_name_prefix="factor-history")
        # Bound in-flight frames as well as active reads. Cache mutation and
        # sequence assembly stay on the caller thread, in calendar order.
        for start in range(0, len(dates), self.read_workers):
            batch = dates[start:start + self.read_workers]
            resident = {day: self._cached(day) for day in batch}
            pending = {day: self._read_pool.submit(self._load_day, day)
                       for day in batch if resident[day] is None}
            for day in batch:
                frame = resident[day]
                if frame is None:
                    frame = self._remember(day, pending.pop(day).result())
                yield day, frame

    def close(self):
        if self._read_pool is not None:
            self._read_pool.shutdown(wait=True, cancel_futures=True)
            self._read_pool = None
        super().close()

    def _filter_frame(self, frame, date):
        """Optional universe refinement before any cross-sectional normalization."""
        return frame

    def symbols(self, date):
        dates = self.window_dates(date)
        if not dates or dates[0] < self.first_date:
            return []
        return super().symbols(date)

    def forward_view(self, length):
        """Independent bounded reader for supervised future slices; caller must close it."""
        view = FactorHistory(self.config, self.provider, sequence_length=length,
                             memory_budget_mb=self.config.features.params.get("future_memory_budget_mb", 128))
        view.set_calendar_dates(self.dates)
        return view


class FactorSequenceProvider(FeatureProvider):
    """Complete eligible cross-sections with lazy, RAM-bounded factor histories."""

    def __init__(self, config, provider):
        self.feature_columns = list(provider.feature_columns)
        self.sequence = FactorHistory(config, provider)

    def set_calendar_dates(self, dates):
        self.sequence.set_calendar_dates(dates)

    def dataset_identity(self):
        transform = "mad5_zscore_v1" if self.sequence.history_transform == "mad5_zscore_rank_label" else "zscore_clip3_v1"
        return {"factor_history": transform, "membership": "complete_history_before_label_join"}

    def load(self, trade_date):
        symbols = self.sequence.symbols(trade_date)
        return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "ts_code": symbols})


class GraphFactorFrameProvider(FactorSequenceProvider):
    def __init__(self, config, provider):
        super().__init__(config, provider)
        self.relations = StockRelations(_project_path(config.model.params.get("relation_root", "data/derived/stock/relations")))

    def dataset_identity(self):
        return {"factor_history": super().dataset_identity()["factor_history"], "relations": self.relations.identity()}

    def load(self, trade_date):
        self.relations.identity()
        frame = super().load(trade_date)
        if not frame.empty and not data_io.exists(daily_path(self.relations.root, trade_date)):
            raise FileNotFoundError(f"missing stock relations for {trade_date}; run derive-stock-relations")
        return frame
