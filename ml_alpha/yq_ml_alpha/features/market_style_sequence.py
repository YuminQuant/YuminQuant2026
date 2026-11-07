"""Observed daily market data and Barra exposures, joined without disk staging."""
from __future__ import annotations

import numpy as np

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider


PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]


def _read(root, date, columns):
    frame = read_daily(root, date, columns, required=columns)
    if (frame[["trade_date", "ts_code"]].isna().any().any()
            or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all()):
        raise ValueError(f"invalid daily market/style keys: {root}, {date}")
    return frame


class MarketStyleHistory(DailyTensorSequence):
    def __init__(self, config):
        params = config.features.params
        columns = config.features.columns
        self.barra_columns = list(params["barra_columns"])
        self.turnover_column = str(params["turnover_column"])
        self.market_columns = [*PRICE_COLUMNS, "volume", self.turnover_column]
        if self.turnover_column not in {"turnover_rate", "turnover_rate_f"}:
            raise ValueError("unsupported daily turnover field")
        if (not self.barra_columns or len(set(self.barra_columns)) != len(self.barra_columns)
                or not isinstance(columns, list) or columns != self.market_columns + self.barra_columns):
            raise ValueError("market_style_sequence columns must be market inputs followed by explicit Barra columns")
        super().__init__(config.features.root, columns, (len(columns),),
                         length=int(params["sequence_length"]), memory_budget_mb=params.get("memory_budget_mb"))
        self.basic_root = _project_path(params.get("basic_root", self.root.parent / "basic"))
        self.adj_root = _project_path(params.get("adj_root", self.root.parent / "adj_factor"))
        self.barra_root = _project_path(params.get("barra_root", config.data_root / "barra" / "stock" / "daily" / params["barra_model"]))
        self.history_start_date = int(params.get("history_start_date", 0))
        self.minimum_history_sessions = int(params.get("minimum_history_sessions", self.length))
        if self.minimum_history_sessions < self.length:
            raise ValueError("minimum_history_sessions must cover the input sequence")
        self.filters = TradeFilters(config.filters, config.data_root)
        self.universe = Universe(config.universe.id, config.data_root)
        from yq_ml_alpha.output import artifact_io
        workers = int(params["history_read_workers"])
        prefetch_mb = int(params["history_prefetch_mb"])
        if workers < 1 or prefetch_mb < 1:
            raise ValueError("history read workers and prefetch budget must be positive")
        self.read_workers = workers if artifact_io.remote_enabled() else 1
        self.prefetch_bytes = prefetch_mb * 1024**2
        self.normalization = str(params["normalization"])
        if self.normalization not in {"window_mean", "last_value"}:
            raise ValueError("unsupported market/style normalization")

    def set_calendar_dates(self, dates):
        super().set_calendar_dates([date for date in dates if date >= self.history_start_date])

    def _load_day(self, date):
        market = _read(self.root, date, ["open", "high", "low", "close", "vol", "amount"])
        basic = _read(self.basic_root, date, [self.turnover_column])
        adjustment = _read(self.adj_root, date, ["adj_factor"])
        barra = _read(self.barra_root, date, self.barra_columns)
        frame = market.merge(basic, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        frame = frame.merge(adjustment, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        frame = frame.merge(barra, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        frame = self.filters.apply(self.universe.filter(frame, date), date)
        numeric = ["open", "high", "low", "close", "vol", "amount", "adj_factor", self.turnover_column, *self.barra_columns]
        frame[numeric] = frame[numeric].astype("float64")
        valid = np.isfinite(frame[["open", "high", "low", "close", "vol", "amount", "adj_factor", self.turnover_column]]).all(axis=1)
        valid &= (frame[["open", "high", "low", "close", "adj_factor"]] > 0).all(axis=1)
        valid &= (frame[["vol", "amount", self.turnover_column]] >= 0).all(axis=1)
        frame = frame.loc[valid].copy()
        # Tushare amount is thousand CNY; vol is hundred shares. Zero-volume
        # observed sessions retain VWAP=0, as in the other bar providers.
        frame["vwap"] = np.divide(frame.amount.to_numpy() * 10, frame.vol.to_numpy(),
                                  out=np.zeros(len(frame)), where=frame.vol.to_numpy() > 0)
        frame["volume"] = frame.vol * 100
        frame[PRICE_COLUMNS] = frame[PRICE_COLUMNS].mul(frame.adj_factor, axis=0)
        return frame.set_index("ts_code")[self.columns].replace([np.inf, -np.inf], np.nan).astype("float32")

    def symbols(self, date):
        if self.positions.get(date, -1) + 1 < self.minimum_history_sessions:
            return []
        symbols = super().symbols(date)
        if not symbols or self.normalization != "last_value":
            return symbols
        last = self.day(date).loc[self._symbols, self.market_columns]
        keep = np.isfinite(last).all(axis=1).to_numpy() & last.ne(0).all(axis=1).to_numpy()
        if not keep.all():
            self._symbols = self._symbols[keep]
            self._positions_by_day = {day: positions[keep] for day, positions in self._positions_by_day.items()}
        return self._symbols.tolist()


class MarketStyleSequenceProvider(DailySequenceProvider):
    def __init__(self, config):
        super().__init__(MarketStyleHistory(config))

    def dataset_identity(self):
        return {"market_style_sequence": "adjusted_daily_market_barra_v1",
                "membership": "complete_market_history_before_label_join",
                "normalization": self.sequence.normalization,
                "barra_missing": "date_cross_section_zscore_then_zero"}


def normalize_market_style(values, *, mode, eps):
    """Normalize one full stock cross-section, independently of return labels."""
    values = np.array(values, dtype=np.float64, copy=True)
    if values.ndim != 3 or values.shape[2] < 8:
        raise ValueError("market/style normalization expects market inputs plus Barra")
    if not np.isfinite(values[:, :, :7]).all():
        raise ValueError("market inputs must contain complete finite history")
    if mode == "window_mean":
        values[:, :, :5] /= values[:, -1:, 3:4]
        denominator = values[:, :, 5:7].mean(axis=1, keepdims=True)
        values[:, :, 5:7] /= np.where(denominator > eps, denominator, 1.0)
    elif mode == "last_value":
        denominator = values[:, -1:, :7].copy()
        if (np.abs(denominator) <= eps).any():
            raise ValueError("zero final market input must be excluded before normalization")
        values[:, :, :7] /= denominator
    else:
        raise ValueError("unsupported market/style normalization")
    # No label or training/validation membership enters these statistics.
    finite = np.isfinite(values)
    count = finite.sum(axis=0, keepdims=True)
    mean = np.where(finite, values, 0.0).sum(axis=0, keepdims=True) / np.maximum(count, 1)
    centered = np.where(finite, values - mean, 0.0)
    std = np.sqrt(np.square(centered).sum(axis=0, keepdims=True) / np.maximum(count, 1))
    return (centered / np.where(std > eps, std, 1.0)).astype("float32")
