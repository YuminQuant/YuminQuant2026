from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path, parquet_columns, read_daily
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.bar_panel import BarTensorWindow, _feature_columns


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
        if list(columns) != self.output_features:
            raise ValueError(f"resnet_panel {self.dataset} requires columns {self.output_features}")
        daily_root = params.get("daily_root")
        if daily_root is None:
            daily_root = (self.root.parent if self.dataset == "week"
                          else self.root.parents[3] / "chn_stock_data" / "daily")
        self.daily_root = Path(daily_root)
        self.feature_columns = _feature_columns(self.output_features, self.total_steps)
        self.max_cache_sessions = int(params.get("max_cache_sessions", self.lookback_sessions))
        if self.max_cache_sessions < 0:
            raise ValueError("max_cache_sessions must be nonnegative")
        self._sessions = OrderedDict()

    def load(self, trade_date):
        raise NotImplementedError("resnet_panel requires calendar-aware load_window_tensor")

    def required_history_dates(self, history_dates):
        return list(history_dates)[-self.lookback_sessions:]

    def cache_policy_summary(self):
        return f"mode=memory_only limit={self.max_cache_sessions} dataset={self.dataset}"

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
        sessions = []
        symbols = None
        for index, date in enumerate(history):
            session_symbols, values = self._session(date)
            eligible = session_symbols
            if exclude_bj:
                eligible = eligible[~eligible.str.endswith(".BJ")]
            excluded = (st_symbols_by_date or {}).get(date, set())
            if excluded:
                eligible = eligible[~eligible.isin(excluded)]
            symbols = eligible if symbols is None else symbols.intersection(eligible, sort=False)
            sessions.append((session_symbols, values))
            if progress is not None:
                progress(f"source {index + 1}/{len(history)} date={date} rows={len(session_symbols)} dataset={self.dataset}")
        if symbols.empty:
            return self.empty_tensor_window(trade_date, tensor_key=tensor_key)
        symbols = symbols.sort_values()
        values = np.concatenate([data[idx.get_indexer(symbols)] for idx, data in sessions], axis=1)
        if self.dataset == "week":
            # All 150 prices use the sample's last adjusted close, not each patch's close.
            values[:, :, :5] /= values[:, -1:, 3:4].copy()
        finite = np.isfinite(values).all(axis=(1, 2))
        frame = pd.DataFrame({"trade_date": trade_date, "ts_code": symbols[finite]})
        return BarTensorWindow(frame, {tensor_key: values[finite].astype("float32")},
                               {tensor_key: self.feature_columns}, skipped_nonfinite=int((~finite).sum()))

    def _session(self, date):
        if date in self._sessions:
            self._sessions.move_to_end(date)
            return self._sessions[date]
        session = self._week_session(date) if self.dataset == "week" else self._minute_session(date)
        self._sessions[date] = session
        while len(self._sessions) > self.max_cache_sessions:
            self._sessions.popitem(last=False)
        return session

    def _week_session(self, date):
        frame = _read_required(self.root, date, ["open", "high", "low", "close", "vol", "amount"])
        adjustment = _read_required(self.daily_root / "adj_factor", date, ["adj_factor"])
        basic = _read_required(self.daily_root / "basic", date, ["turnover_rate_f"])
        frame = frame.merge(adjustment, on=["trade_date", "ts_code"], validate="one_to_one")
        frame = frame.merge(basic, on=["trade_date", "ts_code"], validate="one_to_one").set_index("ts_code")
        numeric = frame[["open", "high", "low", "close", "vol", "amount", "adj_factor", "turnover_rate_f"]].astype("float64")
        valid = np.isfinite(numeric).all(axis=1) & (numeric.iloc[:, :7] > 0).all(axis=1) & (numeric.turnover_rate_f >= 0)
        numeric = numeric.loc[valid].copy()
        numeric["vwap"] = numeric["amount"] * 10.0 / numeric["vol"]  # Daily source: thousand yuan / lots.
        numeric.loc[:, self.output_features[:5]] = numeric[self.output_features[:5]].mul(numeric.adj_factor, axis=0)
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
    if not path.exists():
        raise FileNotFoundError(f"resnet_panel missing source {path}; msnew requires Rust derive-bar --bar-size 30")
    missing = set(["ts_code", *columns]) - set(parquet_columns(path))
    if missing:
        raise ValueError(f"resnet_panel missing columns {sorted(missing)} in {path}")
    frame = read_daily(root, date, columns)
    if not frame.empty and not frame.trade_date.eq(date).all():
        raise ValueError(f"resnet_panel mismatched trade_date in {path}")
    return frame
