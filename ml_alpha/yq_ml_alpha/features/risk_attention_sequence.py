"""Thirty daily bars and point-in-time target-date risk exposures."""
from __future__ import annotations

from bisect import bisect_right
from contextlib import closing, contextmanager
from dataclasses import replace

import numpy as np
import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_keyed_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.features.transforms import scale_market_window, zscore_observed
from yq_ml_alpha.output import artifact_io


MARKET_COLUMNS = ["open", "high", "low", "close", "vol"]


class IndustryMembership:
    """Same inclusive interval and latest-effective-row convention as the Rust reader."""

    def __init__(self, frame, codes):
        self.groups = {}
        positions = {code: i for i, code in enumerate(codes)}
        columns = ["ts_code", "l1_code", "in_date", "out_date"]
        for symbol, code, start, stop in frame[columns].itertuples(index=False, name=None):
            if pd.isna(symbol) or pd.isna(code) or pd.isna(start) or not str(code).strip():
                continue
            if code not in positions:
                raise ValueError(f"unknown CITIC level-one code {code}; check the configured industry schema")
            start = int(start)
            stop = 99991231 if pd.isna(stop) or not str(stop).strip() else int(stop)
            if stop < start:
                raise ValueError(f"invalid CITIC industry interval: {symbol}/{start}/{stop}")
            self.groups.setdefault(symbol, []).append((start, stop, positions[code]))
        for stock, rows in self.groups.items():
            rows.sort(key=lambda row: row[0])
            self.groups[stock] = tuple(zip(*rows))

    def indices(self, symbols, date):
        result = np.full(len(symbols), -1, dtype=np.intp)
        for i, stock in enumerate(symbols):
            history = self.groups.get(stock)
            if history is None:
                continue
            starts, ends, codes = history
            for j in range(bisect_right(starts, date) - 1, -1, -1):
                if date <= ends[j]:
                    result[i] = codes[j]
                    break
        return result


class RiskAttentionHistory(DailyTensorSequence):
    def __init__(self, config):
        params = config.features.params
        super().__init__(config.features.root, MARKET_COLUMNS, (len(MARKET_COLUMNS),),
                         length=int(config.model.params["sequence_length"]))
        self.barra_columns = list(params["barra_columns"])
        self.industry_codes = list(params["industry_codes"])
        if (config.features.columns != MARKET_COLUMNS or self.length != 30
                or len(self.barra_columns) != 9 or len(set(self.barra_columns)) != 9
                or not self.industry_codes or len(set(self.industry_codes)) != len(self.industry_codes)):
            raise ValueError("risk attention requires 30-day OHLCV, nine Barra styles and distinct CITIC industries")
        self.barra_root = config.data_root / "barra" / "stock" / "daily" / "CNE6"
        self.industry_path = config.data_root / "index_data" / "member_ci" / "ci_members.parquet"
        self.adjustment_root = self.root.parent / "adj_factor"
        self.adjust_prices = params["adjust_prices"]
        if type(self.adjust_prices) is not bool:
            raise ValueError("risk attention adjust_prices must be boolean")
        self.history_start = int(params["history_start_date"])
        self.filters = TradeFilters(config.filters, config.data_root)
        self.early_filters = TradeFilters(replace(config.filters, exclude_st=False), config.data_root)
        self.universe = Universe(config.universe.id, config.data_root)
        self.industry = None
        self.risk_days = {}
        self.risk_reads = None

    def window_dates(self, date):
        window = super().window_dates(date)
        return window if window and window[0] >= self.history_start else []

    @contextmanager
    def prefetch_targets(self, dates):
        if artifact_io.backend(self.root) is None or dates != sorted(dates):
            yield
            return
        windows = {date: self.window_dates(date) for date in dates}
        history = sorted({day for window in windows.values() for day in window})
        missing_risk = (date for date, window in windows.items() if window and date not in self.risk_days)
        with self.prefetch_sources(history), closing(artifact_io.ordered_io_map(
            lambda date: (date, self._load_risk_day(date)), missing_risk, root=self.barra_root,
        )) as reads:
            self.risk_reads = reads
            try:
                yield
            finally:
                self.risk_reads = None

    def _load_day(self, date):
        frame = read_keyed_daily(self.root, date, MARKET_COLUMNS)
        filters = self.filters if date >= 20160101 else self.early_filters
        frame = filters.apply(self.universe.filter(frame, date), date)
        values = frame[MARKET_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        valid = np.isfinite(values).all(axis=1) & (values[:, :4] > 0).all(axis=1) & (values[:, 4] >= 0)
        if self.adjust_prices:
            adjustment = read_keyed_daily(self.adjustment_root, date, ["adj_factor"]).set_index("ts_code")
            factor = adjustment.adj_factor.reindex(frame.ts_code).to_numpy(dtype=np.float64)
            valid &= np.isfinite(factor) & (factor > 0)
            values[:, :4] *= factor[:, None]
        values = values[valid].astype(np.float32)
        return pd.DataFrame(values, index=pd.Index(frame.ts_code[valid], name="ts_code"),
                            columns=self.columns, copy=False)

    def _load_risk_day(self, date):
        frame = read_keyed_daily(self.barra_root, date, self.barra_columns).set_index("ts_code")
        return frame[self.barra_columns].astype(np.float32)

    def risk_day(self, date):
        if date not in self.risk_days:
            if self.risk_reads is not None:
                loaded, frame = next(self.risk_reads)
                if loaded != date:
                    raise RuntimeError(f"out-of-order risk prefetch: expected {date}, got {loaded}")
            else:
                frame = self._load_risk_day(date)
            self.risk_days[date] = frame
        return self.risk_days[date]

    def _industry_indices(self, symbols, date):
        if self.industry is None:
            columns = ["ts_code", "l1_code", "in_date", "out_date"]
            memberships = artifact_io.read_projected_frame(self.industry_path, columns, required=columns)
            self.industry = IndustryMembership(memberships, self.industry_codes)
        return self.industry.indices(symbols, date)

    def eligible(self, date):
        if not self.window_dates(date):
            return []
        self.risk_day(date)
        symbols = self.symbols(date)
        if not symbols:
            return symbols
        keep = self._industry_indices(symbols, date) >= 0
        return np.asarray(symbols)[keep].tolist()

    def inputs(self, date, symbols):
        market = self.get(date, symbols)
        scale_market_window(market, price_columns=4, eps=1e-8)
        flat = market.reshape(-1, len(self.columns))
        flat -= flat.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = flat.std(axis=0, dtype=np.float64)
        flat /= np.where(scale > 1e-8, scale, 1).astype(np.float32)
        styles = self.risk_day(date).reindex(symbols).to_numpy(dtype=np.float32, copy=False)
        industry_indices = self._industry_indices(symbols, date)
        if (industry_indices < 0).any():
            raise ValueError(f"unclassified stocks reached the risk attention cross-section: {date}")
        industry = np.zeros((len(symbols), len(self.industry_codes)), dtype=np.float32)
        industry[np.arange(len(symbols)), industry_indices] = 1
        return market, zscore_observed(np.concatenate((styles, industry), axis=1))

    def close(self):
        super().close()
        self.risk_days.clear()
        self.industry = None


class RiskAttentionProvider(DailySequenceProvider):
    def __init__(self, config):
        super().__init__(RiskAttentionHistory(config))

    def load(self, trade_date):
        symbols = self.sequence.eligible(trade_date)
        return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "ts_code": symbols})

    def dataset_identity(self):
        return {"kind": "risk_attention_daily_v1", "market_normalization": "stock_window_scale_then_joint_zscore",
                "risk_normalization": "date_cross_section_per_column",
                "industry_membership": "citic_level_one_latest_effective_inclusive",
                "barra_missing": "zero_after_normalization", "unclassified_industry": "exclude"}
