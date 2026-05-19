from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import is_all_column_request, read_daily
from yq_ml_alpha.features.base import FeatureProvider


BAR_FEATURES = ["open", "high", "low", "close", "vwap", "volume"]
MINUTE_REQUIRED_COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
DAILY_REQUIRED_COLUMNS = ["open", "high", "low", "close", "vol", "amount"]


class BarPanelProvider(FeatureProvider):
    """Build fixed-width end-to-end bar tensors from minute or daily OHLCV data."""

    def __init__(self, root: str | Path, columns: list[str] | str, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        self.root = Path(root)
        self.source_frequency = str(params.get("source_frequency", "minute")).strip().lower()
        self.bar_size = int(params.get("bar_size", 1))
        self.lookback_sessions = int(params.get("lookback_sessions", 1))
        self.time_series_scale = str(params.get("time_series_scale", "none")).strip().lower()
        self.strict = bool(params.get("strict", True))
        self.max_cache_sessions = int(params.get("max_cache_sessions", params.get("max_cache_days", 80)))
        self.output_features = BAR_FEATURES if is_all_column_request(columns) else list(columns)
        _validate_features(self.output_features)
        self.steps_per_session = _steps_per_session(
            self.source_frequency,
            self.bar_size,
            self.lookback_sessions,
        )
        self.total_steps = self.lookback_sessions * self.steps_per_session
        if self.source_frequency in {"daily", "day", "1d"}:
            self.total_steps = self.steps_per_session
        self.feature_columns = _feature_columns(self.output_features, self.total_steps)
        self._session_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

    def load(self, trade_date: int) -> pd.DataFrame:
        raise NotImplementedError("BarPanelProvider requires load_window(..., history_dates=...)")

    def load_window(self, trade_date: int, history_dates: list[int]) -> pd.DataFrame:
        history = list(history_dates)[-self.lookback_sessions :]
        if len(history) < self.lookback_sessions:
            return self.empty_frame(trade_date)
        if self.source_frequency in {"minute", "1m"}:
            return self._load_minute_window(trade_date, history)
        if self.source_frequency in {"daily", "day", "1d"}:
            return self._load_daily_window(trade_date, history)
        raise ValueError(f"unsupported bar_panel source_frequency: {self.source_frequency}")

    def empty_frame(self, trade_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "ts_code", *self.feature_columns])

    def _load_minute_window(self, trade_date: int, history: list[int]) -> pd.DataFrame:
        frames = []
        for day_idx, date in enumerate(history):
            daily = self._load_minute_session(date)
            if daily.empty and self.strict:
                return self.empty_frame(trade_date)
            renamed = daily.rename(
                columns={
                    _session_column(feature, bar_idx): _feature_column(
                        feature, day_idx * self.steps_per_session + bar_idx
                    )
                    for bar_idx in range(self.steps_per_session)
                    for feature in self.output_features
                }
            )
            frames.append(renamed[["ts_code", *[_feature_column(feature, day_idx * self.steps_per_session + bar_idx) for bar_idx in range(self.steps_per_session) for feature in self.output_features]]])
        return self._merge_window_frames(trade_date, frames)

    def _load_daily_window(self, trade_date: int, history: list[int]) -> pd.DataFrame:
        frames = []
        for day_idx, date in enumerate(history):
            daily = self._read_daily_session(date)
            if daily.empty and self.strict:
                return self.empty_frame(trade_date)
            daily = daily.copy()
            daily["__day_idx"] = day_idx
            daily["__bar_idx"] = day_idx // self.bar_size
            frames.append(daily)
        if not frames:
            return self.empty_frame(trade_date)
        bars = _aggregate_daily_bars(pd.concat(frames, ignore_index=True), self.bar_size, self.steps_per_session)
        if bars.empty:
            return self.empty_frame(trade_date)
        pieces = []
        for feature in self.output_features:
            pivot = bars.pivot(index="ts_code", columns="__bar_idx", values=feature)
            pivot = pivot.reindex(columns=list(range(self.steps_per_session)))
            pivot.columns = [_feature_column(feature, int(idx)) for idx in pivot.columns]
            pieces.append(pivot)
        wide = pd.concat(pieces, axis=1).reset_index()
        return self._finalize_window(trade_date, wide)

    def _merge_window_frames(self, trade_date: int, frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return self.empty_frame(trade_date)
        merged = frames[0]
        for frame in frames[1:]:
            how = "inner" if self.strict else "outer"
            merged = merged.merge(frame, on="ts_code", how=how)
            if merged.empty and self.strict:
                return self.empty_frame(trade_date)
        return self._finalize_window(trade_date, merged)

    def _finalize_window(self, trade_date: int, wide: pd.DataFrame) -> pd.DataFrame:
        if wide.empty:
            return self.empty_frame(trade_date)
        wide = wide.copy()
        if self.strict:
            wide = wide.dropna(subset=self.feature_columns)
            if wide.empty:
                return self.empty_frame(trade_date)
        wide.insert(0, "trade_date", int(trade_date))
        wide = wide[["trade_date", "ts_code", *self.feature_columns]]
        if self.time_series_scale == "mean":
            wide = _time_series_mean_scale(wide, self.output_features, self.total_steps)
        elif self.time_series_scale == "last":
            wide = _time_series_last_scale(wide, self.output_features, self.total_steps)
        elif self.time_series_scale in {"", "none"}:
            pass
        else:
            raise ValueError(f"unsupported bar_panel time_series_scale: {self.time_series_scale}")
        return wide

    def _load_minute_session(self, trade_date: int) -> pd.DataFrame:
        cached = self._cache_get(trade_date)
        if cached is not None:
            return cached
        frame = _read_minute_session(self.root, trade_date)
        session = _aggregate_minute_session(frame, self.bar_size, self.steps_per_session, self.strict)
        self._cache_put(trade_date, session)
        return session

    def _read_daily_session(self, trade_date: int) -> pd.DataFrame:
        cached = self._cache_get(trade_date)
        if cached is not None:
            return cached
        frame = read_daily(self.root, trade_date, DAILY_REQUIRED_COLUMNS)
        if frame.empty:
            session = pd.DataFrame(columns=["trade_date", "ts_code", *DAILY_REQUIRED_COLUMNS])
        else:
            session = frame.rename(columns={"vol": "volume"}).copy()
        self._cache_put(trade_date, session)
        return session

    def _cache_get(self, trade_date: int) -> pd.DataFrame | None:
        if trade_date not in self._session_cache:
            return None
        frame = self._session_cache.pop(trade_date)
        self._session_cache[trade_date] = frame
        return frame

    def _cache_put(self, trade_date: int, frame: pd.DataFrame) -> None:
        self._session_cache[trade_date] = frame
        while len(self._session_cache) > self.max_cache_sessions:
            self._session_cache.popitem(last=False)


class MultiBarPanelProvider(FeatureProvider):
    """Compose multiple bar panels into one feature frame with stable prefixes."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        panels = params.get("panels")
        if not isinstance(panels, dict) or not panels:
            raise ValueError("multi_bar_panel requires [features.panels.<name>] sections")
        self.strict = bool(params.get("strict", True))
        self.panels: list[tuple[str, BarPanelProvider]] = []
        self.feature_columns: list[str] = []
        for name, raw_panel in panels.items():
            if not isinstance(raw_panel, dict):
                raise ValueError(f"multi_bar_panel panel {name!r} must be a table")
            prefix = str(raw_panel.get("prefix", name)).strip()
            if not prefix:
                raise ValueError(f"multi_bar_panel panel {name!r} has empty prefix")
            if "__" in prefix:
                raise ValueError(f"multi_bar_panel prefix {prefix!r} cannot contain '__'")
            if "root" not in raw_panel:
                raise ValueError(f"multi_bar_panel panel {name!r} requires root")
            columns = raw_panel.get("columns", BAR_FEATURES)
            provider_params = {key: value for key, value in raw_panel.items() if key not in {"root", "columns", "prefix"}}
            provider = BarPanelProvider(raw_panel["root"], columns, provider_params)
            self.panels.append((prefix, provider))
            self.feature_columns.extend(_prefix_columns(prefix, provider.feature_columns))

    def load(self, trade_date: int) -> pd.DataFrame:
        raise NotImplementedError("MultiBarPanelProvider requires load_window(..., history_dates=...)")

    def load_window(self, trade_date: int, history_dates: list[int]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for prefix, provider in self.panels:
            frame = provider.load_window(trade_date, history_dates)
            if frame.empty and self.strict:
                return self.empty_frame(trade_date)
            renamed = frame.rename(columns={column: _prefix_column(prefix, column) for column in provider.feature_columns})
            frames.append(renamed[["trade_date", "ts_code", *_prefix_columns(prefix, provider.feature_columns)]])
        if not frames:
            return self.empty_frame(trade_date)
        merged = frames[0]
        for frame in frames[1:]:
            how = "inner" if self.strict else "outer"
            merged = merged.merge(frame, on=["trade_date", "ts_code"], how=how)
            if merged.empty and self.strict:
                return self.empty_frame(trade_date)
        if self.strict:
            merged = merged.dropna(subset=self.feature_columns)
        if merged.empty:
            return self.empty_frame(trade_date)
        return merged[["trade_date", "ts_code", *self.feature_columns]].reset_index(drop=True)

    def empty_frame(self, trade_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "ts_code", *self.feature_columns])


def _validate_features(features: list[str]) -> None:
    unsupported = [feature for feature in features if feature not in BAR_FEATURES]
    if unsupported:
        raise ValueError(f"unsupported bar_panel columns: {unsupported}; supported: {BAR_FEATURES}")


def _steps_per_session(source_frequency: str, bar_size: int, lookback_sessions: int) -> int:
    if source_frequency in {"minute", "1m"}:
        if bar_size < 1 or bar_size > 120:
            raise ValueError("minute bar_panel requires 1 <= bar_size <= 120")
        return 2 * (120 // bar_size)
    if source_frequency in {"daily", "day", "1d"}:
        if bar_size <= 0:
            raise ValueError("daily bar_panel requires bar_size > 0")
        if lookback_sessions <= 0:
            raise ValueError("daily bar_panel requires lookback_sessions > 0")
        if lookback_sessions % bar_size != 0:
            raise ValueError("daily bar_panel requires lookback_sessions to be divisible by bar_size")
        return lookback_sessions // bar_size
    raise ValueError(f"unsupported bar_panel source_frequency: {source_frequency}")


def _read_minute_session(root: Path, trade_date: int) -> pd.DataFrame:
    path = root / str(trade_date // 10000) / f"{trade_date}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=MINUTE_REQUIRED_COLUMNS)
    return pd.read_parquet(path, columns=MINUTE_REQUIRED_COLUMNS)


def _aggregate_minute_session(
    frame: pd.DataFrame,
    bar_size: int,
    steps_per_session: int,
    strict: bool,
) -> pd.DataFrame:
    columns = ["ts_code", *[_session_column(feature, idx) for idx in range(steps_per_session) for feature in BAR_FEATURES]]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    minute = _minute_of_day(data["trade_time"])
    morning_start = 9 * 60 + 31
    afternoon_start = 13 * 60 + 1
    bars_per_half = 120 // bar_size
    morning_end = morning_start + bars_per_half * bar_size - 1
    afternoon_end = afternoon_start + bars_per_half * bar_size - 1
    morning = (minute >= morning_start) & (minute <= morning_end)
    afternoon = (minute >= afternoon_start) & (minute <= afternoon_end)
    data = data.loc[morning | afternoon].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    minute = minute.loc[data.index]
    data["__bar_idx"] = np.where(
        minute <= morning_end,
        ((minute - morning_start) // bar_size).astype("int16"),
        (bars_per_half + (minute - afternoon_start) // bar_size).astype("int16"),
    )
    data = data.sort_values(["ts_code", "__bar_idx", "trade_time"])
    agg = (
        data.groupby(["ts_code", "__bar_idx"], sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("vol", "sum"),
            amount=("amount", "sum"),
            minute_count=("close", "count"),
        )
        .reset_index()
    )
    agg = agg.loc[(agg["__bar_idx"] >= 0) & (agg["__bar_idx"] < steps_per_session)].copy()
    if strict:
        agg = agg.loc[agg["minute_count"] == bar_size].copy()
    agg["vwap"] = np.where(agg["volume"].astype("float64").abs() > 1e-12, agg["amount"] / agg["volume"], np.nan)
    return _bars_to_wide(agg, steps_per_session)


def _aggregate_daily_bars(frame: pd.DataFrame, bar_size: int, steps_per_session: int) -> pd.DataFrame:
    data = frame.copy()
    data = data.loc[(data["__bar_idx"] >= 0) & (data["__bar_idx"] < steps_per_session)].copy()
    if data.empty:
        return pd.DataFrame()
    data = data.sort_values(["ts_code", "__day_idx"])
    grouped = data.groupby(["ts_code", "__bar_idx"], sort=True)
    agg = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        day_count=("close", "count"),
    ).reset_index()
    agg = agg.loc[agg["day_count"] == bar_size].copy()
    agg["vwap"] = np.where(agg["volume"].astype("float64").abs() > 1e-12, agg["amount"] * 10.0 / agg["volume"], np.nan)
    return agg


def _bars_to_wide(agg: pd.DataFrame, steps_per_session: int) -> pd.DataFrame:
    pieces = []
    for feature in BAR_FEATURES:
        pivot = agg.pivot(index="ts_code", columns="__bar_idx", values=feature)
        pivot = pivot.reindex(columns=list(range(steps_per_session)))
        pivot.columns = [_session_column(feature, int(idx)) for idx in pivot.columns]
        pieces.append(pivot)
    wide = pd.concat(pieces, axis=1).reset_index()
    needed = [_session_column(feature, idx) for idx in range(steps_per_session) for feature in BAR_FEATURES]
    return wide.dropna(subset=needed).reset_index(drop=True)


def _minute_of_day(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.extract(r"(?P<hour>\d{2}):(?P<minute>\d{2})(?::\d{2})?$")
    hour = pd.to_numeric(text["hour"], errors="coerce")
    minute = pd.to_numeric(text["minute"], errors="coerce")
    return hour * 60 + minute


def _time_series_mean_scale(frame: pd.DataFrame, features: list[str], total_steps: int) -> pd.DataFrame:
    output = frame.copy()
    for feature in features:
        columns = [_feature_column(feature, step) for step in range(total_steps)]
        values = output[columns].replace([np.inf, -np.inf], np.nan).astype("float32")
        means = values.mean(axis=1, skipna=True)
        valid = np.isfinite(means.to_numpy(dtype="float64")) & (np.abs(means.to_numpy(dtype="float64")) > 1e-12)
        scaled = values.div(means, axis=0)
        scaled.loc[~valid, :] = np.nan
        output[columns] = scaled.astype("float32")
    return output


def _time_series_last_scale(frame: pd.DataFrame, features: list[str], total_steps: int) -> pd.DataFrame:
    output = frame.copy()
    for feature in features:
        columns = [_feature_column(feature, step) for step in range(total_steps)]
        values = output[columns].replace([np.inf, -np.inf], np.nan).astype("float32")
        last = values.iloc[:, -1]
        valid = np.isfinite(last.to_numpy(dtype="float64")) & (np.abs(last.to_numpy(dtype="float64")) > 1e-12)
        scaled = values.div(last, axis=0)
        scaled.loc[~valid, :] = np.nan
        output[columns] = scaled.astype("float32")
    return output


def _feature_columns(features: list[str], total_steps: int) -> list[str]:
    return [_feature_column(feature, step) for step in range(total_steps) for feature in features]


def _feature_column(feature: str, step: int) -> str:
    return f"{feature}__t{step:03d}"


def _session_column(feature: str, bar_idx: int) -> str:
    return f"{feature}__b{bar_idx:03d}"


def _prefix_columns(prefix: str, columns: list[str]) -> list[str]:
    return [_prefix_column(prefix, column) for column in columns]


def _prefix_column(prefix: str, column: str) -> str:
    return f"{prefix}__{column}"
