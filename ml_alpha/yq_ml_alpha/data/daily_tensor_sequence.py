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

    def set_calendar_dates(self, dates):
        self.dates = list(dates)
        self.positions = {date: i for i, date in enumerate(dates)}

    def window_dates(self, date):
        position = self.positions[date]
        if position + 1 < self.length:
            return []
        return self.dates[position + 1 - self.length:position + 1]

    def day(self, date):
        if date in self.cache:
            self.cache.move_to_end(date)
            return self.cache[date][0]
        path = daily_path(self.root, date)
        if not path.exists():
            raise FileNotFoundError(f"Missing derived daily tensor: {path}; materialize derived data first")
        frame = pd.read_parquet(path, columns=["trade_date", "ts_code", *self.columns])
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid/duplicate daily tensor keys: {path}")
        frame = frame.set_index("ts_code")[self.columns].astype("float32")
        frame = frame.loc[np.isfinite(frame.to_numpy()).all(axis=1)]
        size = int(frame.memory_usage(index=True, deep=True).sum())
        while self.cache and self.cache_bytes + size > self.budget:
            _, (_, removed) = self.cache.popitem(last=False)
            self.cache_bytes -= removed
        if size <= self.budget:
            self.cache[date] = (frame, size)
            self.cache_bytes += size
        return frame

    def symbols(self, date):
        dates = self.window_dates(date)
        if not dates:
            return []
        symbols = None
        for day in dates:
            index = self.day(day).index
            symbols = index if symbols is None else symbols.intersection(index, sort=False)
        return sorted(symbols.tolist())

    def get(self, date, symbols):
        dates = self.window_dates(date)
        if len(dates) != self.length:
            raise ValueError(f"insufficient daily tensor history: {date}")
        result = np.empty((len(symbols), self.length, *self.shape), dtype="float32")
        for i, day in enumerate(dates):
            result[:, i] = self.day(day).loc[list(symbols)].to_numpy().reshape(len(symbols), *self.shape)
        return result

    def close(self):
        self.cache.clear()
        self.cache_bytes = 0
