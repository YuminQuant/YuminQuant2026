"""Bounded, read-only daily tensor sequences; no temporary files or disk cache."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path


class DailyTensorSequence:
    def __init__(self, root, columns, shape, length=30, memory_budget_mb=128):
        self.root = Path(root)
        self.columns = list(columns)
        self.shape = tuple(shape)
        self.length = int(length)
        self.budget = int(memory_budget_mb) * 1024 * 1024
        if self.length < 1 or self.budget < 1 or np.prod(self.shape) != len(self.columns):
            raise ValueError("invalid daily tensor sequence dimensions or memory budget")
        self.dates = []
        self.positions = {}
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self._active_date = None
        self._protected = set()
        self._symbols = None
        self._positions_by_day = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.uncached_reads = 0
        self._day_sizes = {}

    def _activate(self, date):
        if self._active_date == date:
            return
        self._active_date = date
        self._protected = set(self.window_dates(date))
        self._symbols = None
        self._positions_by_day = {}
        # Keep overlapping days. A scan larger than the budget must not evict
        # its own resident days and turn every subsequent stock chunk cold.
        for day in list(self.cache):
            if day not in self._protected:
                _, size = self.cache.pop(day)
                self.cache_bytes -= size

    def _cached(self, date):
        if date in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(date)
            return self.cache[date][0]
        self.cache_misses += 1
        return None

    def _remember(self, date, frame):
        positions = self._positions_by_day.get(date)
        if positions is not None and len(positions):
            if positions.max() >= len(frame) or not frame.index.take(positions).equals(self._symbols):
                raise ValueError(f"source stock order changed while reading sequence date {date}")
        size = int(frame.memory_usage(index=True, deep=True).sum())
        self._day_sizes[date] = size
        for old in list(self.cache):
            if self.cache_bytes + size <= self.budget:
                break
            if old not in self._protected:
                _, removed = self.cache.pop(old)
                self.cache_bytes -= removed
        if self.cache_bytes + size <= self.budget:
            self.cache[date] = (frame, size)
            self.cache_bytes += size
        else:
            # Oversized working sets stream their nonresident days. The cache
            # stays bounded and resident overlap is never displaced by a scan.
            self.uncached_reads += 1
        return frame

    def set_calendar_dates(self, dates):
        self.dates = list(dates)
        self.positions = {date: i for i, date in enumerate(dates)}

    def window_dates(self, date):
        position = self.positions[date]
        if position + 1 < self.length:
            return []
        return self.dates[position + 1 - self.length:position + 1]

    def day(self, date):
        cached = self._cached(date)
        if cached is not None:
            return cached
        path = daily_path(self.root, date)
        if not path.exists():
            raise FileNotFoundError(f"Missing derived daily tensor: {path}; materialize derived data first")
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", *self.columns])
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid/duplicate daily tensor keys: {path}")
        frame = frame.set_index("ts_code")[self.columns].astype("float32")
        frame = frame.loc[np.isfinite(frame.to_numpy()).all(axis=1)]
        return self._remember(date, frame)

    def symbols(self, date):
        self._activate(date)
        if self._symbols is not None:
            return self._symbols.tolist()
        dates = self.window_dates(date)
        if not dates:
            return []
        symbols = None
        indexes = {}
        working_bytes = 0
        for day in dates:
            frame = self.day(day)
            index = frame.index
            indexes[day] = index
            working_bytes += self._day_sizes[day]
            symbols = index if symbols is None else symbols.intersection(index, sort=False)
        self._symbols = symbols.sort_values()
        self._positions_by_day = {day: index.get_indexer(self._symbols) for day, index in indexes.items()}
        if working_bytes > self.budget and not getattr(self, "_budget_reported", False):
            print(f"sequence working_set_mib={working_bytes / 2**20:.1f} "
                  f"budget_mib={self.budget / 2**20:.1f} mode=bounded_scan_resistant "
                  "nonresident_days=streamed", flush=True)
            self._budget_reported = True
        return self._symbols.tolist()

    def get(self, date, symbols):
        self.symbols(date)
        dates = self.window_dates(date)
        if len(dates) != self.length:
            raise ValueError(f"insufficient daily tensor history: {date}")
        result = np.empty((len(symbols), self.length, *self.shape), dtype="float32")
        selected = self._symbols.get_indexer(symbols)
        if (selected < 0).any():
            raise ValueError(f"requested stocks lack complete sequence history: {date}")
        for i, day in enumerate(dates):
            positions = self._positions_by_day[day][selected]
            result[:, i] = self.day(day).to_numpy(copy=False)[positions].reshape(len(symbols), *self.shape)
        return result

    def close(self):
        self.cache.clear()
        self.cache_bytes = 0
        self._active_date = None
        self._protected.clear()
        self._symbols = None
        self._positions_by_day.clear()
        self._day_sizes.clear()
