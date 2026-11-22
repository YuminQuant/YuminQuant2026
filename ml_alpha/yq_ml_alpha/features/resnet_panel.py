from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path, read_daily
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.bar_panel import BarTensorWindow, _feature_columns
from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.output.artifact_io import ordered_io_map


class _Session(NamedTuple):
    symbols: pd.Index
    values: np.ndarray
    rows: np.ndarray


class ResNetPanelProvider(FeatureProvider):
    """Report-96 week/msnew images, packed as chronological bar tensors."""

    def __init__(self, root, columns, params=None):
        params = dict(params or {})
        self.root = Path(root)
        self.requires_full_cross_section = params.get("full_cross_section", False)
        if not isinstance(self.requires_full_cross_section, bool):
            raise ValueError("full_cross_section must be a boolean")
        self.dataset = str(params.get("dataset", "week"))
        if self.dataset not in {"week", "msnew"}:
            raise ValueError("resnet_panel dataset must be week or msnew")
        self.lookback_sessions = 150 if self.dataset == "week" else 30
        self.steps_per_session = 1 if self.dataset == "week" else 8
        self.total_steps = self.lookback_sessions * self.steps_per_session
        self.output_features = (["open", "high", "low", "close", "vwap", "turnover_rate_f"]
                                if self.dataset == "week" else ["open", "high", "low", "close", "amount"])
        self.include_daily_return = params.get("include_daily_return", False)
        if not isinstance(self.include_daily_return, bool) or (self.include_daily_return and self.dataset != "week"):
            raise ValueError("include_daily_return is a boolean option for the week dataset")
        if self.include_daily_return:
            self.output_features.append("daily_return_zscore")
        if list(columns) != self.output_features:
            raise ValueError(f"resnet_panel {self.dataset} requires columns {self.output_features}")
        daily_root = params.get("daily_root")
        if daily_root is None:
            daily_root = (self.root.parent if self.dataset == "week"
                          else self.root.parents[3] / "chn_stock_data" / "daily")
        self.daily_root = Path(daily_root)
        self.feature_columns = _feature_columns(self.output_features, self.total_steps)
        self._sessions = OrderedDict()
        self._symbol_ids = {}
        self._prefetched = None

    def load(self, trade_date):
        raise NotImplementedError("resnet_panel requires calendar-aware load_window_tensor")

    def required_history_dates(self, history_dates):
        return list(history_dates)[-self.lookback_sessions:]

    def cache_policy_summary(self):
        return f"mode=memory_only limit=unlimited dataset={self.dataset}"

    def empty_tensor_window(self, trade_date=None, *, tensor_key="bar"):
        return BarTensorWindow(pd.DataFrame(columns=["trade_date", "ts_code"]),
                               {tensor_key: np.empty((0, self.total_steps, len(self.output_features)), dtype="float32")},
                               {tensor_key: self.feature_columns})

    def load_window(self, trade_date, history_dates, **kwargs):
        window = self.load_window_tensor(trade_date, history_dates, **kwargs)
        values = window.tensors["bar"].reshape(len(window.frame), len(self.feature_columns))
        return pd.concat([window.frame, pd.DataFrame(values, columns=self.feature_columns)], axis=1)

    def load_window_tensor(self, trade_date, history_dates, *, tensor_key="bar", exclude_bj=False,
                           st_symbols_by_date=None, progress=None):
        history = self.required_history_dates([d for d in history_dates if d <= trade_date])
        if len(history) < self.lookback_sessions or history[-1] != trade_date:
            return self.empty_tensor_window(trade_date, tensor_key=tensor_key)
        started = perf_counter()
        cached = sum(date in self._sessions for date in history)
        sessions = list(self._history_sessions(history))
        loaded = perf_counter()
        symbols, ids = self._align_symbols(sessions, history, exclude_bj, st_symbols_by_date)
        aligned = perf_counter()
        if progress is not None:
            progress(f"source sessions={len(history)} cache_hits={cached} "
                     f"session_load_s={loaded - started:.3f} align_s={aligned - loaded:.3f} "
                     f"aligned_rows={len(symbols)} dataset={self.dataset}")
        if symbols.empty:
            return self.empty_tensor_window(trade_date, tensor_key=tensor_key)
        values = np.empty((len(symbols), self.total_steps, len(self.output_features)), dtype="float64")
        width = self.steps_per_session
        for step, session in enumerate(sessions):
            values[:, step * width:(step + 1) * width] = session.values[session.rows[ids]]
        assembled = perf_counter()
        if self.dataset == "week":
            values, symbols = self._prepare_week_window(values, symbols)
        finite = np.isfinite(values).all(axis=(1, 2))
        if not finite.all():
            values = values[finite]
        frame = pd.DataFrame({"trade_date": trade_date, "ts_code": symbols[finite]})
        values = values.astype("float32")
        if progress is not None:
            progress(f"panel assemble_s={assembled - aligned:.3f} "
                     f"normalize_s={perf_counter() - assembled:.3f} rows={len(frame)}")
        return BarTensorWindow(frame, {tensor_key: values},
                               {tensor_key: self.feature_columns}, skipped_nonfinite=int((~finite).sum()))

    def _align_symbols(self, sessions, history, exclude_bj, st_symbols_by_date):
        # IDs created after an older session cannot be present in that session.
        limit = min(len(session.rows) for session in sessions)
        common = np.ones(limit, dtype=bool)
        for session in sessions:
            common &= session.rows[:limit] >= 0
        if st_symbols_by_date:
            # Any historical ST flag removes the stock from the whole window.
            for date in history:
                for symbol in st_symbols_by_date.get(date, ()):
                    symbol_id = self._symbol_ids.get(symbol)
                    if symbol_id is not None and symbol_id < limit:
                        common[symbol_id] = False
        ids = np.flatnonzero(common)
        first = sessions[0]
        symbols = first.symbols.take(first.rows[ids])
        if exclude_bj and len(symbols):
            eligible = ~symbols.str.endswith(".BJ")
            symbols, ids = symbols[eligible], ids[eligible]
        order = symbols.argsort()
        return symbols.take(order), ids[order]

    def _prepare_week_window(self, values, symbols):
        # All 150 prices use the sample's last adjusted close, not each patch's close.
        values[:, :, :5] /= values[:, -1:, 3:4].copy()
        return values, symbols

    def _read_session(self, date):
        return self._week_session(date) if self.dataset == "week" else self._minute_session(date)

    def _remember_session(self, date, session):
        self._sessions[date] = session
        self._sessions.move_to_end(date)
        return session

    def _index_session(self, raw):
        symbols, values = raw
        ids = np.fromiter((self._symbol_ids.setdefault(symbol, len(self._symbol_ids)) for symbol in symbols),
                          dtype=np.intp, count=len(symbols))
        rows = np.full(len(self._symbol_ids), -1, dtype=np.int32)
        rows[ids] = np.arange(len(symbols), dtype=np.int32)
        return _Session(symbols, values, rows)

    @contextmanager
    def prefetch_sessions(self, dates):
        """Keep the remote read pool fed across the windows of one prediction batch."""
        cached = {date: self._sessions[date] for date in dates if date in self._sessions}
        missing = [date for date in dates if date not in cached]
        reads = ordered_io_map(lambda date: (date, self._read_session(date)), missing, root=self.root)
        self._prefetched = cached, reads
        try:
            yield
        finally:
            self._prefetched = None
            reads.close()

    def _history_sessions(self, history):
        cached = {date: self._sessions[date] for date in history if date in self._sessions}
        if self._prefetched is not None:
            batch_cache, reads = self._prefetched
            for date in history:
                session = batch_cache.pop(date, None)
                if session is None:
                    session = cached.get(date)
                if session is None:
                    loaded_date, raw = next(reads)
                    if loaded_date != date:
                        raise RuntimeError(f"out-of-order bar-panel prefetch: expected {date}, got {loaded_date}")
                    session = self._index_session(raw)
                yield self._remember_session(date, session)
            return
        missing = (date for date in history if date not in cached)
        reads = ordered_io_map(self._read_session, missing, root=self.root)
        try:
            for date in history:
                session = cached[date] if date in cached else self._index_session(next(reads))
                yield self._remember_session(date, session)
        finally:
            reads.close()

    def _week_session(self, date):
        columns = ["open", "high", "low", "close", "vol", "amount"]
        if self.include_daily_return:
            columns.append("pre_close")
        frame = _read_required(self.root, date, columns)
        if self.include_daily_return:
            close = frame.close.astype("float64")
            previous = frame.pre_close.astype("float64")
            returns = close.where(close > 0) / previous.where(previous > 0) - 1.0
            frame = frame.assign(daily_return_zscore=_zscore_series(returns))
        adjustment = _read_required(self.daily_root / "adj_factor", date, ["adj_factor"])
        basic = _read_required(self.daily_root / "basic", date, ["turnover_rate_f"])
        frame = frame.merge(adjustment, on=["trade_date", "ts_code"], validate="one_to_one")
        frame = frame.merge(basic, on=["trade_date", "ts_code"], validate="one_to_one").set_index("ts_code")
        numeric = frame[["open", "high", "low", "close", "vol", "amount", "adj_factor", "turnover_rate_f"]].astype("float64")
        valid = np.isfinite(numeric).all(axis=1) & (numeric.iloc[:, :7] > 0).all(axis=1) & (numeric.turnover_rate_f >= 0)
        numeric = numeric.loc[valid].copy()
        numeric["vwap"] = numeric["amount"] * 10.0 / numeric["vol"]  # Daily source: thousand yuan / lots.
        numeric.loc[:, self.output_features[:5]] = numeric[self.output_features[:5]].mul(numeric.adj_factor, axis=0)
        if self.include_daily_return:
            numeric["daily_return_zscore"] = frame["daily_return_zscore"]
        return numeric.index, numeric[self.output_features].to_numpy(dtype="float64")[:, None, :]

    def _minute_session(self, date):
        bars = _read_required(self.root, date, ["bar_index", "minute_count", *self.output_features])
        daily = _read_required(self.daily_root / "pv", date, ["pre_close"])
        basic = _read_required(self.daily_root / "basic", date, ["circ_mv"])
        if bars.duplicated(["ts_code", "bar_index"]).any():
            raise ValueError(f"duplicate half-hour bars on {date} under {self.root}")
        frame = bars.merge(daily, on=["trade_date", "ts_code"], validate="many_to_one")
        frame = frame.merge(basic, on=["trade_date", "ts_code"], validate="many_to_one")
        numeric_cols = ["bar_index", "minute_count", *self.output_features, "pre_close", "circ_mv"]
        frame[numeric_cols] = frame[numeric_cols].astype("float64")
        valid = np.isfinite(frame[numeric_cols]).all(axis=1)
        valid &= (frame[["open", "high", "low", "close", "pre_close", "circ_mv"]] > 0).all(axis=1)
        valid &= (frame.amount >= 0) & (frame.minute_count == 30) & frame.bar_index.isin(range(8))
        frame = frame.loc[valid].copy()
        counts = frame.groupby("ts_code").size()
        frame = frame[frame.ts_code.isin(counts.index[counts == 8])].sort_values(["ts_code", "bar_index"])
        frame.loc[:, self.output_features[:4]] = frame[self.output_features[:4]].div(frame.pre_close, axis=0)
        frame["amount"] = frame.amount / (frame.circ_mv * 10000.0)  # Minute amount yuan; circ_mv ten-thousand yuan.
        symbols = pd.Index(frame.ts_code.drop_duplicates(), name="ts_code")
        return symbols, frame[self.output_features].to_numpy(dtype="float64").reshape(len(symbols), 8, 5)


def _read_required(root, date, columns):
    path = daily_path(root, date)
    frame = read_daily(root, date, columns, required=["ts_code", *columns])
    if not frame.empty and not frame.trade_date.eq(date).all():
        raise ValueError(f"resnet_panel mismatched trade_date in {path}")
    return frame
