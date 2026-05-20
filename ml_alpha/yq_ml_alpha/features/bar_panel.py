from __future__ import annotations

import inspect
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import is_all_column_request, read_daily
from yq_ml_alpha.features.base import FeatureProvider


BAR_FEATURES = ["open", "high", "low", "close", "vwap", "volume"]
MINUTE_REQUIRED_COLUMNS = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
DERIVED_MINUTE_REQUIRED_COLUMNS = [
    "trade_date",
    "trade_time",
    "bar_index",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "minute_count",
]
DAILY_REQUIRED_COLUMNS = ["open", "high", "low", "close", "vol", "amount"]
ProgressCallback = Callable[[str], None]


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
        self._session_cache: OrderedDict[Any, pd.DataFrame] = OrderedDict()

    def load(self, trade_date: int) -> pd.DataFrame:
        raise NotImplementedError("BarPanelProvider requires load_window(..., history_dates=...)")

    def required_history_dates(self, history_dates: list[int]) -> list[int]:
        return list(history_dates)[-self.lookback_sessions :]

    def load_window(
        self,
        trade_date: int,
        history_dates: list[int],
        *,
        exclude_bj: bool = False,
        st_symbols_by_date: dict[int, set[str]] | None = None,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        history = list(history_dates)[-self.lookback_sessions :]
        if len(history) < self.lookback_sessions:
            return self.empty_frame(trade_date)
        if self.source_frequency in {"minute", "1m", "minute_bar", "derived_minute"}:
            return self._load_minute_window(trade_date, history, exclude_bj, st_symbols_by_date or {}, progress)
        if self.source_frequency in {"daily", "day", "1d"}:
            return self._load_daily_window(trade_date, history)
        raise ValueError(f"unsupported bar_panel source_frequency: {self.source_frequency}")

    def empty_frame(self, trade_date: int | None = None) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "ts_code", *self.feature_columns])

    def _load_minute_window(
        self,
        trade_date: int,
        history: list[int],
        exclude_bj: bool,
        st_symbols_by_date: dict[int, set[str]],
        progress: ProgressCallback | None,
    ) -> pd.DataFrame:
        frames = []
        for day_idx, date in enumerate(history):
            daily = self._load_minute_session(
                date,
                exclude_bj,
                st_symbols_by_date.get(date, set()),
                progress=(
                    (lambda message, idx=day_idx: progress(f"source {idx + 1}/{len(history)} {message}"))
                    if progress is not None
                    else None
                ),
            )
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

    def _load_minute_session(
        self,
        trade_date: int,
        exclude_bj: bool = False,
        st_symbols: set[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        cache_key = (trade_date, bool(exclude_bj), frozenset(st_symbols or set()))
        cached = self._cache_get(cache_key)
        if cached is not None:
            if progress is not None:
                progress(f"date={trade_date} cache=hit rows={len(cached)}")
            return cached
        if progress is not None:
            progress(f"date={trade_date} cache=miss step=read")
        if self.source_frequency in {"minute_bar", "derived_minute"}:
            frame = _read_derived_minute_session(self.root, trade_date)
            if progress is not None:
                progress(f"date={trade_date} step=pivot derived_rows={len(frame)}")
            session = _derived_minute_session_to_wide(
                frame,
                self.steps_per_session,
                self.strict,
                exclude_bj=exclude_bj,
                st_symbols=st_symbols or set(),
            )
        else:
            frame = _read_minute_session(self.root, trade_date)
            if progress is not None:
                progress(f"date={trade_date} step=resample raw_rows={len(frame)}")
            session = _aggregate_minute_session(
                frame,
                trade_date,
                self.bar_size,
                self.steps_per_session,
                self.strict,
                exclude_bj=exclude_bj,
                st_symbols=st_symbols or set(),
            )
        self._cache_put(cache_key, session)
        if progress is not None:
            progress(f"date={trade_date} step=done stocks={len(session)}")
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

    def _cache_get(self, key: Any) -> pd.DataFrame | None:
        if key not in self._session_cache:
            return None
        frame = self._session_cache.pop(key)
        self._session_cache[key] = frame
        return frame

    def _cache_put(self, key: Any, frame: pd.DataFrame) -> None:
        self._session_cache[key] = frame
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

    def required_history_dates(self, history_dates: list[int]) -> list[int]:
        required: list[int] = []
        seen: set[int] = set()
        for _, provider in self.panels:
            for date in provider.required_history_dates(history_dates):
                if date not in seen:
                    required.append(date)
                    seen.add(date)
        return required

    def load_window(
        self,
        trade_date: int,
        history_dates: list[int],
        *,
        exclude_bj: bool = False,
        st_symbols_by_date: dict[int, set[str]] | None = None,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for prefix, provider in self.panels:
            frame = provider.load_window(
                trade_date,
                history_dates,
                exclude_bj=exclude_bj,
                st_symbols_by_date=st_symbols_by_date,
                progress=(
                    (lambda message, panel=prefix: progress(f"panel={panel} {message}"))
                    if progress is not None
                    else None
                ),
            )
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
    if source_frequency in {"minute", "1m", "minute_bar", "derived_minute"}:
        if bar_size < 1 or bar_size > 120:
            raise ValueError("minute bar_panel requires 1 <= bar_size <= 120")
        if source_frequency in {"minute_bar", "derived_minute"}:
            if 240 % bar_size != 0 or bar_size <= 1 or bar_size > 120:
                raise ValueError("derived minute bar_panel requires bar_size to divide 240 and satisfy 1 < bar_size <= 120")
            return 240 // bar_size
        return len(_canonical_minute_bar_labels(bar_size))
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


def _read_derived_minute_session(root: Path, trade_date: int) -> pd.DataFrame:
    path = root / str(trade_date // 10000) / f"{trade_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing derived minute bar file for {trade_date}: {path}. "
            "Run: cargo run --release --manifest-path factor_engine\\Cargo.toml -- "
            f"derive-bar --asset stock --source minute --bar-size <N> --start-date {trade_date} --end-date {trade_date}"
        )
    return pd.read_parquet(path, columns=DERIVED_MINUTE_REQUIRED_COLUMNS)


def _derived_minute_session_to_wide(
    frame: pd.DataFrame,
    steps_per_session: int,
    strict: bool,
    *,
    exclude_bj: bool = False,
    st_symbols: set[str] | None = None,
) -> pd.DataFrame:
    columns = ["ts_code", *[_session_column(feature, idx) for idx in range(steps_per_session) for feature in BAR_FEATURES]]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    if exclude_bj:
        data = data.loc[~data["ts_code"].astype("string").str.upper().str.endswith(".BJ", na=False)].copy()
    if st_symbols:
        data = data.loc[~data["ts_code"].isin(st_symbols)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["__bar_idx"] = pd.to_numeric(data["bar_index"], errors="coerce")
    data = data.loc[data["__bar_idx"].notna()].copy()
    data["__bar_idx"] = data["__bar_idx"].astype("int16")
    data = data.loc[(data["__bar_idx"] >= 0) & (data["__bar_idx"] < steps_per_session)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    return _bars_to_wide(data, steps_per_session, strict)


def _aggregate_minute_session(
    frame: pd.DataFrame,
    trade_date: int,
    bar_size: int,
    steps_per_session: int,
    strict: bool,
    *,
    exclude_bj: bool = False,
    st_symbols: set[str] | None = None,
) -> pd.DataFrame:
    columns = ["ts_code", *[_session_column(feature, idx) for idx in range(steps_per_session) for feature in BAR_FEATURES]]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.copy()
    if exclude_bj:
        data = data.loc[~data["ts_code"].astype("string").str.upper().str.endswith(".BJ", na=False)].copy()
    if st_symbols:
        data = data.loc[~data["ts_code"].isin(st_symbols)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data["trade_time"] = _normalize_trade_time(data["trade_time"], trade_date)
    data = data.loc[data["trade_time"].notna()].copy()
    minute = _minute_of_day(data["trade_time"])
    data = data.loc[minute != 9 * 60 + 30].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    data = data.sort_values(["ts_code", "trade_time"])
    indexed = data.set_index("trade_time")
    agg = (
        indexed.drop(columns=["ts_code"])
        .groupby(indexed["ts_code"])
        .resample(
            f"{bar_size}min",
            **_minute_resample_kwargs(),
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "vol": "sum",
                "amount": "sum",
            }
        )
        .dropna(subset=["open"])
        .reset_index()
    )
    if agg.empty:
        return pd.DataFrame(columns=columns)
    label_to_idx = {label: idx for idx, label in enumerate(_canonical_minute_bar_labels(bar_size))}
    labels = agg["trade_time"].dt.strftime("%H:%M:%S")
    agg["__bar_idx"] = labels.map(label_to_idx)
    agg = agg.loc[agg["__bar_idx"].notna()].copy()
    if agg.empty:
        return pd.DataFrame(columns=columns)
    agg["__bar_idx"] = agg["__bar_idx"].astype("int16")
    agg = agg.loc[(agg["__bar_idx"] >= 0) & (agg["__bar_idx"] < steps_per_session)].copy()
    agg = agg.rename(columns={"vol": "volume"})
    agg["vwap"] = np.where(agg["volume"].astype("float64").abs() > 1e-12, agg["amount"] / agg["volume"], np.nan)
    return _bars_to_wide(agg, steps_per_session, strict)


@lru_cache(maxsize=1)
def _resample_supports_origin() -> bool:
    return "origin" in inspect.signature(pd.DataFrame.resample).parameters


def _minute_resample_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "label": "right",
        "closed": "right",
    }
    if _resample_supports_origin():
        kwargs.update(
            {
                "origin": "start_day",
                "offset": "9h30min",
            }
        )
    else:
        kwargs["base"] = 30
    return kwargs


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


def _bars_to_wide(agg: pd.DataFrame, steps_per_session: int, strict: bool) -> pd.DataFrame:
    pieces = []
    for feature in BAR_FEATURES:
        pivot = agg.pivot(index="ts_code", columns="__bar_idx", values=feature)
        pivot = pivot.reindex(columns=list(range(steps_per_session)))
        pivot.columns = [_session_column(feature, int(idx)) for idx in pivot.columns]
        pieces.append(pivot)
    wide = pd.concat(pieces, axis=1).reset_index()
    if strict:
        needed = [_session_column(feature, idx) for idx in range(steps_per_session) for feature in BAR_FEATURES]
        wide = wide.dropna(subset=needed)
    return wide.reset_index(drop=True)


def _minute_of_day(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.hour * 60 + parsed.dt.minute


def _normalize_trade_time(values: pd.Series, trade_date: int) -> pd.Series:
    text = values.astype("string")
    date_text = f"{trade_date // 10000:04d}-{(trade_date // 100) % 100:02d}-{trade_date % 100:02d}"
    time_only = text.str.match(r"^\d{2}:\d{2}(?::\d{2})?$", na=False)
    normalized = text.where(~time_only, date_text + " " + text)
    return pd.to_datetime(normalized, errors="coerce")


@lru_cache(maxsize=None)
def _canonical_minute_bar_labels(bar_size: int) -> list[str]:
    date = pd.Timestamp("2000-01-03")
    morning = pd.date_range(date + pd.Timedelta(hours=9, minutes=31), periods=120, freq="min")
    afternoon = pd.date_range(date + pd.Timedelta(hours=13, minutes=1), periods=120, freq="min")
    frame = pd.DataFrame({"value": 1.0}, index=morning.append(afternoon))
    labels = (
        frame.resample(
            f"{bar_size}min",
            **_minute_resample_kwargs(),
        )
        .agg({"value": "first"})
        .dropna()
        .index
    )
    return [label.strftime("%H:%M:%S") for label in labels]


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
