from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from yq_ml_alpha.data.stores import daily_path


def _minute_index(value: str | None) -> int:
    if value is None:
        return -1
    try:
        parts = value.strip().rsplit(" ", 1)[-1].split(":")
        hour, minute = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return -1
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return -1
    clock = hour * 60 + minute
    if 571 <= clock <= 690:
        return clock - 571
    if 781 <= clock <= 900:
        return 120 + clock - 781
    return -1


def read_minute_volume_day(path: Path, bar_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Match derive-bar --columns volume, without materializing repeated strings."""
    slots = 240 // bar_size
    if not path.exists():
        return np.array([], dtype=object), np.empty((0, slots), dtype=np.float64)
    table = pq.read_table(path, columns=["ts_code", "trade_time", "vol"], use_threads=False)
    codes = table.column("ts_code").combine_chunks().dictionary_encode()
    times = table.column("trade_time").combine_chunks().dictionary_encode()
    symbols = np.asarray(codes.dictionary.to_pylist(), dtype=object)
    code_ids = codes.indices.to_numpy(zero_copy_only=False)
    time_ids = times.indices.to_numpy(zero_copy_only=False)
    volume = table.column("vol").combine_chunks().to_numpy(zero_copy_only=False)
    time_index = np.asarray([_minute_index(t) for t in times.dictionary.to_pylist()], dtype=np.int32)
    valid = np.isfinite(code_ids) & np.isfinite(time_ids) & np.isfinite(volume)
    code_ids = code_ids[valid].astype(np.int64)
    minutes = time_index[time_ids[valid].astype(np.int64)]
    volume = volume[valid].astype(np.float64)
    in_session = minutes >= 0
    code_ids, minutes, volume = code_ids[in_session], minutes[in_session], volume[in_session]
    # Stable chronological order retains duplicate minute rows, as the Rust aggregator does.
    order = np.argsort(code_ids * 240 + minutes, kind="stable")
    keys = code_ids[order] * slots + minutes[order] // bar_size
    values = np.full((len(symbols), slots), np.nan, dtype=np.float64)
    if keys.size:
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        values.reshape(-1)[keys[starts]] = np.add.reduceat(volume[order], starts)
    # A stock missing any bar cannot be used in any complete window containing this date.
    complete = np.isfinite(values).all(axis=1)
    symbols, values = symbols[complete], values[complete]
    order = np.argsort(symbols)
    return symbols[order], np.ascontiguousarray(values[order])


class MinuteVolumeWindow:
    """Bounded date-keyed state; retains compact daily arrays, not minute tables."""

    def __init__(self, root: Path, bar_size: int):
        self.root = root
        self.bar_size = bar_size
        self.days = OrderedDict()

    def matrix(self, dates: list[int], fingerprints: list) -> tuple[list[str], np.ndarray]:
        required = set(dates)
        for old in list(self.days):
            if old not in required:
                del self.days[old]
        for date, fingerprint in zip(dates, fingerprints):
            cached = self.days.get(date)
            if cached is None or cached[0] != fingerprint:
                symbols, values = read_minute_volume_day(daily_path(self.root, date), self.bar_size)
                self.days[date] = (fingerprint, symbols, values)
        symbols = self.days[dates[0]][1]
        for date in dates[1:]:
            symbols = np.intersect1d(symbols, self.days[date][1], assume_unique=True)
        slots = 240 // self.bar_size
        matrix = np.empty((len(symbols), len(dates) * slots), dtype=np.float64)
        for i, date in enumerate(dates):
            _, day_symbols, values = self.days[date]
            matrix[:, i * slots:(i + 1) * slots] = values[np.searchsorted(day_symbols, symbols)]
        return symbols.tolist(), matrix
