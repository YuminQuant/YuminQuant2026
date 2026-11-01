"""Adjusted daily OHLCV histories with per-stock, per-window standardization."""
from __future__ import annotations

import numpy as np

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.output import artifact_io


RAWBAR_COLUMNS = ["open", "high", "low", "close", "vwap", "volume"]


class RawBarHistory(DailyTensorSequence):
    def __init__(self, config):
        params = config.features.params
        if config.features.columns != RAWBAR_COLUMNS:
            raise ValueError("rawbar_sequence requires open, high, low, close, vwap, volume in order")
        super().__init__(config.features.root, RAWBAR_COLUMNS, (6,),
                         length=int(params["sequence_length"]), memory_budget_mb=int(params["memory_budget_mb"]))
        self.adj_root = _project_path(params.get("adj_root", self.root.parent / "adj_factor"))
        self.filters = TradeFilters(config.filters, config.data_root)
        self.universe = Universe(config.universe.id, config.data_root)
        workers, prefetch = int(params["history_read_workers"]), int(params["history_prefetch_mb"])
        if workers < 1 or prefetch < 1:
            raise ValueError("history workers and prefetch budget must be positive")
        self.read_workers = workers if artifact_io.remote_enabled() else 1
        self.prefetch_bytes = prefetch * 1024**2
        self._mean = self._std = None

    def _read(self, root, date, columns):
        frame = read_daily(root, date, columns, required=columns)
        if (frame[["trade_date", "ts_code"]].isna().any().any()
                or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all()):
            raise ValueError(f"invalid daily rawbar keys: {root}, {date}")
        return frame

    def _load_day(self, date):
        market = self._read(self.root, date, ["open", "high", "low", "close", "vol", "amount"])
        adjustment = self._read(self.adj_root, date, ["adj_factor"])
        frame = market.merge(adjustment, on=["trade_date", "ts_code"], how="left", validate="one_to_one")
        frame = self.filters.apply(self.universe.filter(frame, date), date)
        numeric = ["open", "high", "low", "close", "vol", "amount", "adj_factor"]
        frame[numeric] = frame[numeric].astype("float64")
        valid = np.isfinite(frame[numeric]).all(axis=1)
        valid &= (frame[["open", "high", "low", "close", "adj_factor"]] > 0).all(axis=1)
        valid &= (frame[["vol", "amount"]] >= 0).all(axis=1)
        frame = frame.loc[valid].copy()
        # Tushare amount: thousand CNY; vol: hundred shares. Follow the shared
        # project convention VWAP=0 on observed zero-volume sessions.
        frame["vwap"] = np.divide(frame.amount.to_numpy() * 10, frame.vol.to_numpy(),
                                  out=np.zeros(len(frame)), where=frame.vol.to_numpy() > 0)
        frame["volume"] = frame.vol * 100
        prices = RAWBAR_COLUMNS[:-1]
        frame[prices] = frame[prices].mul(frame.adj_factor, axis=0)
        values = frame.set_index("ts_code")[self.columns].astype("float32")
        return values.loc[np.isfinite(values.to_numpy()).all(axis=1)]

    def _activate(self, date):
        if self._active_date != date:
            self._mean = self._std = None
        super()._activate(date)

    def symbols(self, date):
        symbols = super().symbols(date)
        if not symbols or self._std is not None:
            return symbols
        # Welford statistics avoid a second full N x T x F working array and
        # cancellation for almost constant prices. Only observed history enters.
        mean = np.zeros((len(symbols), len(self.columns)), dtype="float64")
        m2 = np.zeros_like(mean)
        for count, (day, frame) in enumerate(self._days(self.window_dates(date)), start=1):
            values = frame.to_numpy(copy=False)[self._positions_by_day[day]].astype("float64")
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)
        std = np.sqrt(np.maximum(m2 / self.length, 0.0))
        keep = np.isfinite(mean).all(axis=1) & np.isfinite(std).all(axis=1) & (std > 0).all(axis=1)
        self._symbols = self._symbols[keep]
        self._positions_by_day = {day: positions[keep] for day, positions in self._positions_by_day.items()}
        self._mean, self._std = mean[keep], std[keep]
        return self._symbols.tolist()

    def get(self, date, symbols):
        values = super().get(date, symbols).astype("float64")
        selected = self._symbols.get_indexer(symbols)
        return ((values - self._mean[selected, None, :]) / self._std[selected, None, :]).astype("float32")

    def close(self):
        super().close()
        self._mean = self._std = None


class RawBarSequenceProvider(DailySequenceProvider):
    def __init__(self, config):
        super().__init__(RawBarHistory(config))

    def dataset_identity(self):
        return {"rawbar_sequence": "adjusted_ohlcv_v1", "normalization": "stock_window_zscore_ddof0",
                "membership": "complete_finite_history_and_positive_std_all_six_features"}
