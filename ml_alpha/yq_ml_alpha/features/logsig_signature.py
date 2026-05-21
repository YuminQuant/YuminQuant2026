from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.features.base import FeatureProvider

ProgressCallback = Callable[[str], None]


def signature_width(order: int) -> int:
    return sum(2**level for level in range(1, order + 1))


class LogsigSignatureProvider(FeatureProvider):
    """Compute Logsig-Alpha-v signature features from cached 5m bar files."""

    def __init__(
        self,
        root: str | Path,
        columns: list[str] | str,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = dict(params or {})
        self.root = Path(root)
        self.lookback_days = int(params.get("lookback_days", 20))
        self.bar_size = int(params.get("bar_size", 5))
        self.order = int(params.get("order", 10))
        self.volume_column = str(params.get("volume_column", "volume"))
        self.cache_days = max(1, int(params.get("cache_days", 128)))
        self.progress_interval = max(1, int(params.get("progress_interval", 500)))
        if self.lookback_days <= 0:
            raise ValueError("logsig_signature lookback_days must be positive")
        if self.bar_size <= 0 or 240 % self.bar_size != 0:
            raise ValueError("logsig_signature bar_size must divide 240")
        if self.order <= 0:
            raise ValueError("logsig_signature order must be positive")
        self.steps_per_day = 240 // self.bar_size
        self.window_steps = self.lookback_days * self.steps_per_day
        width = signature_width(self.order)
        self.feature_columns = [f"sig_{idx:04}" for idx in range(1, width + 1)]
        self._bar_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._calendar_dates: list[int] = []

    def set_calendar_dates(self, dates: list[int]) -> None:
        self._calendar_dates = list(dates)

    def load(self, trade_date: int, progress: ProgressCallback | None = None) -> pd.DataFrame:
        source_dates = self._source_dates(trade_date)
        if len(source_dates) < self.lookback_days:
            if progress is not None:
                progress(f"source_days={len(source_dates)}/{self.lookback_days} step=insufficient_history")
            return self._empty_frame()
        if progress is not None:
            progress(
                f"source_days={len(source_dates)} window_steps={self.window_steps} "
                f"order={self.order} width={len(self.feature_columns)} step=read"
            )
        daily_frames = []
        for day_idx, source_date in enumerate(source_dates, start=1):
            frame = self._load_bar_day(
                source_date,
                progress=(
                    (lambda message, idx=day_idx: progress(f"source {idx}/{len(source_dates)} {message}"))
                    if progress is not None
                    else None
                ),
            )
            if frame.empty:
                if progress is not None:
                    progress(f"source {day_idx}/{len(source_dates)} date={source_date} step=empty")
                return self._empty_frame()
            daily_frames.append(frame)
        stacked = pd.concat(daily_frames, ignore_index=True)
        if progress is not None:
            progress(
                f"step=stack rows={len(stacked)} stocks={stacked['ts_code'].nunique()} "
                f"columns=trade_date,ts_code,bar_index,{self.volume_column}"
            )
        rows = []
        groups = stacked.groupby("ts_code", sort=True)
        total_groups = groups.ngroups
        skipped_incomplete = 0
        skipped_nonfinite = 0
        if progress is not None:
            progress(f"step=signature_start stocks={total_groups}")
        for group_idx, (ts_code, group) in enumerate(groups, start=1):
            group = (
                group.sort_values(["trade_date", "bar_index"], kind="mergesort")
                .drop_duplicates(["trade_date", "bar_index"], keep="last")
            )
            if len(group) != self.window_steps:
                skipped_incomplete += 1
                continue
            volumes = group[self.volume_column].astype("float64").to_numpy()
            if not np.isfinite(volumes).all():
                skipped_nonfinite += 1
                continue
            values = _signature_from_volume(volumes, self.order).astype("float32", copy=False)
            rows.append((str(ts_code), values))
            if progress is not None and (
                group_idx == total_groups or group_idx % self.progress_interval == 0
            ):
                progress(
                    f"step=signature progress={group_idx}/{total_groups} stocks={len(rows)} "
                    f"skipped_incomplete={skipped_incomplete} skipped_nonfinite={skipped_nonfinite}"
                )
        if not rows:
            if progress is not None:
                progress(
                    f"step=signature_done stocks=0 skipped_incomplete={skipped_incomplete} "
                    f"skipped_nonfinite={skipped_nonfinite}"
                )
            return self._empty_frame()
        data = np.vstack([values for _, values in rows]).astype("float32", copy=False)
        output = pd.DataFrame(data, columns=self.feature_columns)
        output.insert(0, "ts_code", [ts_code for ts_code, _ in rows])
        output.insert(0, "trade_date", np.int32(trade_date))
        if progress is not None:
            progress(
                f"step=signature_done stocks={len(output)} skipped_incomplete={skipped_incomplete} "
                f"skipped_nonfinite={skipped_nonfinite}"
            )
        return output

    def _source_dates(self, trade_date: int) -> list[int]:
        if self._calendar_dates:
            history = [date for date in self._calendar_dates if date <= trade_date]
            return history[-self.lookback_days :]
        candidates = sorted(int(path.stem) for path in self.root.glob("*/*.parquet"))
        history = [date for date in candidates if date <= trade_date]
        return history[-self.lookback_days :]

    def _load_bar_day(self, trade_date: int, progress: ProgressCallback | None = None) -> pd.DataFrame:
        if trade_date in self._bar_cache:
            frame = self._bar_cache.pop(trade_date)
            self._bar_cache[trade_date] = frame
            if progress is not None:
                progress(f"date={trade_date} cache=hit rows={len(frame)} stocks={frame['ts_code'].nunique()}")
            return frame
        path = daily_path(self.root, trade_date)
        columns = ["trade_date", "ts_code", "bar_index", self.volume_column]
        if not path.exists():
            if progress is not None:
                progress(f"date={trade_date} cache=miss step=missing path={path}")
            return pd.DataFrame(columns=columns)
        if progress is not None:
            progress(f"date={trade_date} cache=miss step=read columns={','.join(columns)}")
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.dropna(subset=["trade_date", "ts_code", "bar_index", self.volume_column])
        frame["trade_date"] = frame["trade_date"].astype("int32")
        frame["bar_index"] = frame["bar_index"].astype("int32")
        frame["ts_code"] = frame["ts_code"].astype(str)
        self._bar_cache[trade_date] = frame[columns]
        while len(self._bar_cache) > self.cache_days:
            self._bar_cache.popitem(last=False)
        if progress is not None:
            progress(
                f"date={trade_date} step=done rows={len(self._bar_cache[trade_date])} "
                f"stocks={self._bar_cache[trade_date]['ts_code'].nunique()}"
            )
        return self._bar_cache[trade_date]

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "ts_code", *self.feature_columns])


def _signature_from_volume(volume: np.ndarray, order: int) -> np.ndarray:
    log_values = np.log(np.maximum(volume, 1.0)).astype(np.float64)
    width = signature_width(order)
    level_offsets = np.empty(order + 1, dtype=np.int64)
    running = 0
    level_offsets[0] = 0
    for level in range(1, order + 1):
        level_offsets[level] = running
        running += 2**level
    levels = np.zeros(width, dtype=np.float64)
    previous = np.empty(width, dtype=np.float64)
    scaled = np.empty(order + 1, dtype=np.float64)
    try:
        _signature_from_log_volume_numba(log_values, int(order), levels, previous, level_offsets, scaled)
        return levels
    except NameError as exc:  # pragma: no cover
        raise ImportError("LogsigSignatureProvider requires installing numba") from exc


try:
    from numba import njit
except ImportError:  # pragma: no cover
    pass
else:

    @njit(cache=True)
    def _signature_from_log_volume_numba(values, order, levels, previous, level_offsets, scaled):
        for idx in range(1, len(values)):
            delta = values[idx] - values[idx - 1]
            if abs(delta) <= 1e-15:
                continue
            _append_axis_segment(levels, previous, level_offsets, scaled, 0, delta, order)
            _append_axis_segment(levels, previous, level_offsets, scaled, 1, delta, order)

    @njit(cache=True)
    def _append_axis_segment(levels, previous, level_offsets, scaled, axis, delta, order):
        for idx in range(levels.shape[0]):
            previous[idx] = levels[idx]
        scaled[0] = 1.0
        for level in range(1, order + 1):
            scaled[level] = scaled[level - 1] * delta / level
        for level in range(1, order + 1):
            width = 2**level
            offset = level_offsets[level]
            for word in range(width):
                suffix = _repeated_axis_suffix_len(word, level, axis)
                value = previous[offset + word]
                for repeat in range(1, suffix + 1):
                    if level - repeat == 0:
                        prefix_value = 1.0
                    else:
                        prefix_value = previous[level_offsets[level - repeat] + (word >> repeat)]
                    value += prefix_value * scaled[repeat]
                levels[offset + word] = value

    @njit(cache=True)
    def _repeated_axis_suffix_len(word, level, axis):
        count = 0
        for bit in range(level):
            if ((word >> bit) & 1) == axis:
                count += 1
            else:
                break
        return count
