"""Read-only daily tensor sequences; no temporary files or disk cache."""
from yq_ml_alpha.output import artifact_io as data_io
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path


class DailyTensorSequence:
    def __init__(self, root, columns, shape, length=30):
        self.root = Path(root)
        self.columns = list(columns)
        self.shape = tuple(shape)
        self.length = int(length)
        if self.length < 1 or np.prod(self.shape) != len(self.columns):
            raise ValueError("invalid daily tensor sequence dimensions")
        self.dates = []
        self.positions = {}
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self._active_date = None
        self._symbols = None
        self._positions_by_day = {}
        self._symbol_ids = {}
        self._daily_rows = {}
        self._arrays = {}
        self._window_symbols = {}
        self._prefetched = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.uncached_reads = 0
        self.loaded_days = 0
        self.cache_evictions = 0

    def _activate(self, date):
        if self._active_date == date:
            return
        self._active_date = date
        self._symbols = None
        self._positions_by_day = {}

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
        if list(frame.columns) == self.columns and frame.dtypes.eq(np.dtype("float32")).all():
            values = np.ascontiguousarray(frame.to_numpy(copy=False))
            frame = pd.DataFrame(values, index=frame.index, columns=frame.columns, copy=False)
            self._arrays[date] = values
        else:
            self._arrays.pop(date, None)
        ids = np.fromiter((self._symbol_ids.setdefault(code, len(self._symbol_ids)) for code in frame.index),
                          dtype=np.intp, count=len(frame))
        rows = np.full(len(self._symbol_ids), -1, dtype=np.int32)
        rows[ids] = np.arange(len(frame), dtype=np.int32)
        self._daily_rows[date] = rows
        size = int(frame.memory_usage(index=True, deep=True).sum()) + rows.nbytes
        if date in self.cache:
            self._window_symbols.clear()
        self.loaded_days += 1
        previous = self.cache.pop(date, None)
        if previous is not None:
            self.cache_bytes -= previous[1]
        self.cache[date] = (frame, size)
        self.cache_bytes += size
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
        return self._remember(date, self._load_day(date))

    @contextmanager
    def prefetch_sources(self, dates):
        """Read ahead across target windows on the shared HDFS read pool."""
        missing = (date for date in dates if date not in self.cache)
        reads = data_io.ordered_io_map(lambda date: (date, self._load_day(date)), missing, root=self.root)
        self._prefetched = reads
        try:
            yield
        finally:
            self._prefetched = None
            reads.close()

    def _load_day(self, date):
        path = daily_path(self.root, date)
        if not data_io.exists(path):
            raise FileNotFoundError(f"Missing derived daily tensor: {path}; materialize derived data first")
        frame = data_io.read_parquet(path, columns=["trade_date", "ts_code", *self.columns])
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid/duplicate daily tensor keys: {path}")
        frame = frame.set_index("ts_code")[self.columns].astype("float32")
        frame = frame.loc[np.isfinite(frame.to_numpy()).all(axis=1)]
        return frame

    def _days(self, dates):
        resident = {date: self._cached(date) for date in dates}
        missing = [date for date in dates if resident[date] is None]
        if not missing:
            for date in dates:
                yield date, resident[date]
            return
        if self._prefetched is not None:
            for date in dates:
                frame = resident[date]
                if frame is None:
                    loaded_date, frame = next(self._prefetched)
                    if loaded_date != date:
                        raise RuntimeError(f"out-of-order sequence prefetch: expected {date}, got {loaded_date}")
                    frame = self._remember(date, frame)
                yield date, frame
            return
        # Resident days never enter the IO queue or trigger remote stat calls.
        frames = data_io.ordered_io_map(
            self._load_day, missing, root=self.root,
            workers=getattr(self, "read_workers", None),
        )
        try:
            for date in dates:
                frame = resident[date]
                yield date, frame if frame is not None else self._remember(date, next(frames))
        finally:
            frames.close()

    def cache_info(self):
        return dict(hits=self.cache_hits, misses=self.cache_misses,
                    loaded_days=self.loaded_days, evictions=self.cache_evictions,
                    uncached_reads=self.uncached_reads, resident_days=len(self.cache),
                    resident_bytes=self.cache_bytes, budget_bytes=None,
                    parquet_opens=data_io.parquet_cache_info()["opens"])

    def log_cache(self, stage, *, before=None):
        info = self.cache_info()
        delta = "" if before is None else " ".join(
            f"{key}_delta={info[key] - before[key]}"
            for key in ("hits", "misses", "loaded_days", "evictions", "parquet_opens")
        )
        print(f"sequence-cache stage={stage} resident_days={info['resident_days']} "
              f"resident_mib={self.cache_bytes / 2**20:.1f} "
              f"cache=unlimited "
              f"loaded_days={info['loaded_days']} evictions={info['evictions']} "
              f"parquet_opens={info['parquet_opens']} {delta}", flush=True)
        return info

    def symbols(self, date):
        self._activate(date)
        if self._symbols is not None:
            return self._symbols.tolist()
        dates = self.window_dates(date)
        if not dates:
            return []
        if date not in self._window_symbols:
            for _ in self._days(dates):
                pass
            rows = [self._daily_rows[day] for day in dates]
            limit = min(map(len, rows))
            common = np.ones(limit, dtype=bool)
            for positions in rows:
                common &= positions[:limit] >= 0
            ids = np.flatnonzero(common)
            first = self.cache[dates[0]][0].index
            symbols = first.take(rows[0][ids])
            order = symbols.argsort()
            self._window_symbols[date] = symbols.take(order), ids[order]
        self._symbols, ids = self._window_symbols[date]
        self._positions_by_day = {day: self._daily_rows[day][ids] for day in dates}
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
        for i, (day, frame) in enumerate(self._days(dates)):
            positions = self._positions_by_day[day][selected]
            values = self._arrays.get(day)
            if values is None:
                values = frame.to_numpy(copy=False)
            result[:, i] = values[positions].reshape(len(symbols), *self.shape)
        return result

    def close(self):
        self.cache.clear()
        self.cache_bytes = 0
        self._active_date = None
        self._symbols = None
        self._positions_by_day.clear()
        self._symbol_ids.clear()
        self._daily_rows.clear()
        self._arrays.clear()
        self._window_symbols.clear()
