"""Task-local daily snapshots plus window indices, without expanded-window files.

The existing providers still define aggregation, adjustment, normalization and
eligibility. Only their session reads are adapted on isolated provider copies.
"""
from collections import OrderedDict
import copy
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


class IndexedPanelStore:
    def __init__(self, root, provider, preprocess, memory_budget_mb=256):
        root = Path(root).resolve()
        if root.drive.lower() == "c:":
            raise ValueError("indexed tensor scratch must not be placed on C:")
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="indexed-", dir=root)
        self.entries = {}
        self.maps = OrderedDict()
        self.frames = {}
        self.map_sizes = {}
        self.map_bytes = 0
        self.budget = int(memory_budget_mb) * 2**20
        self.windows = {}
        self.preprocess = preprocess
        self._prepared_date = None
        self._prepared = None
        self.closed = False
        self.disk_bytes = 0
        self.provider = None
        self.provider = self._adapt(provider, "root")

    def _adapt(self, provider, branch):
        source = copy.copy(provider)
        if hasattr(source, "_session_cache"):
            source._session_cache = OrderedDict()
        if hasattr(source, "_sessions"):
            source._sessions = OrderedDict()
        if hasattr(source, "_tensor_blocks"):
            source._tensor_blocks = OrderedDict()
        adapted = copy.copy(source)
        if hasattr(source, "panels"):
            adapted.panels = [(name, self._adapt(child, branch + "_" + name)) for name, child in source.panels]
        elif hasattr(source, "_session"):
            adapted._session = lambda date: self._snapshot((branch, date), lambda: source._session(date))
        else:
            adapted._read_daily_session = lambda date: self._snapshot(
                (branch, "daily", date), lambda: source._read_daily_session(date))

            def minute(date, exclude_bj=False, st_symbols=None, progress=None):
                key = (branch, "minute", date, bool(exclude_bj), frozenset(st_symbols or ()))
                return self._snapshot(key, lambda: source._load_minute_session(
                    date, exclude_bj, st_symbols, progress))
            adapted._load_minute_session = minute
        return adapted

    def _snapshot(self, key, read):
        if self.closed:
            raise RuntimeError("indexed panel store is closed")
        if key not in self.entries:
            item = read()
            if isinstance(item, tuple):
                symbols, values = item
                columns, dtypes, kind = None, None, "array"
            else:
                symbols = item.ts_code.astype(str)
                columns = [column for column in item.columns if column != "ts_code"]
                dtypes = {column: item[column].dtype for column in columns}
                values = item[columns].to_numpy(dtype="float64", na_value=np.nan)
                kind = "frame"
            symbols = pd.Index([sys.intern(str(code)) for code in symbols], name="ts_code")
            path = Path(self.directory.name) / (str(len(self.entries)) + ".npy")
            np.save(path, np.asarray(values), allow_pickle=False)
            self.disk_bytes += path.stat().st_size
            self.entries[key] = (path, symbols, columns, dtypes, kind, values.size)
        path, symbols, columns, dtypes, kind, count = self.entries[key]
        if key in self.frames:
            self.maps.move_to_end(key)
            return self.frames[key]
        if key in self.maps:
            self.maps.move_to_end(key)
            values = self.maps[key]
        else:
            # np.memmap cannot map an empty array.
            values = np.load(path, mmap_mode="r" if count else None, allow_pickle=False)
            cost = values.nbytes * (3 if kind == "frame" else 1) + int(symbols.memory_usage(deep=True))
            while self.maps and self.map_bytes + cost > self.budget:
                old_key, old = self.maps.popitem(last=False)
                self.map_bytes -= self.map_sizes.pop(old_key)
                self.frames.pop(old_key, None)
                # Existing window views may still refer to this map. NumPy
                # releases it when those views expire; never close it early.
            if cost <= self.budget:
                self.maps[key] = values
                self.map_sizes[key] = cost
                self.map_bytes += cost
        if kind == "array":
            return symbols, values
        frame = pd.DataFrame(values, columns=columns).astype(dtypes)
        frame.insert(0, "ts_code", symbols.to_numpy())
        if key in self.maps:
            self.frames[key] = frame
        return frame

    def add(self, date, history, kwargs, eligible_symbols, final_symbols):
        history = sorted(set(self.provider.required_history_dates(history)))
        self.windows[int(date)] = (
            list(history), kwargs, pd.Index(eligible_symbols), pd.Index(final_symbols),
        )

    def finish(self, frame, shapes):
        self.row_dates = frame.trade_date.to_numpy(dtype="int32")
        self.row_positions = np.empty(len(frame), dtype="int64")
        for date, rows in frame.groupby("trade_date", sort=False).indices.items():
            self.row_positions[rows] = np.arange(len(rows))
        print(f"indexed-panel source_snapshots={len(self.entries)} "
              f"scratch_mib={self.disk_bytes / 2**20:.1f} samples={len(frame)} "
              "expanded_window_files=0", flush=True)
        return {key: IndexedPanelTensor(self, key, (len(frame), *shape)) for key, shape in shapes.items()}

    def _materialize(self, date):
        if date != self._prepared_date:
            self._prepared = None
            self._prepared_date = None
            history, kwargs, eligible, final = self.windows[date]
            window = self.provider.load_window_tensor(date, history, **kwargs)
            positions = pd.Index(window.frame.ts_code).get_indexer(eligible)
            if (positions < 0).any():
                raise ValueError(f"indexed panel snapshot changed eligible stocks on {date}")
            tensors = {key: value[positions] for key, value in window.tensors.items()}
            # Transform across the original eligible cross-section, before
            # dropping missing labels, exactly as the eager builder does.
            frame = pd.DataFrame({"trade_date": date, "ts_code": eligible})
            tensors = self.preprocess(frame, tensors)
            selected = eligible.get_indexer(final)
            if (selected < 0).any():
                raise ValueError(f"indexed panel final rows misaligned on {date}")
            self._prepared = {key: value[selected] for key, value in tensors.items()}
            self._prepared_date = date
        return self._prepared

    def take(self, key, rows, shape):
        rows = np.asarray(rows, dtype="int64")
        output = np.empty((len(rows), *shape), dtype="float32")
        dates = self.row_dates[rows]
        for date in np.unique(dates):
            at = np.flatnonzero(dates == date)
            output[at] = self._materialize(int(date))[key][self.row_positions[rows[at]]]
        return output

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._prepared = None
        self.provider = None
        self.preprocess = None
        self.frames.clear()
        for values in self.maps.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.maps.clear()
        self.map_sizes.clear()
        self.entries.clear()
        self.windows.clear()
        self.directory.cleanup()

    def __del__(self):
        if hasattr(self, "closed") and not self.closed:
            self.close()


class IndexedPanelTensor:
    """Array-like row access used by the existing model batch helpers."""
    ndim = 3
    dtype = np.dtype("float32")

    def __init__(self, owner, key, shape):
        self.owner, self.key, self.shape = owner, key, shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, rows):
        if isinstance(rows, slice):
            rows = np.arange(self.shape[0])[rows]
        if np.isscalar(rows):
            return self.owner.take(self.key, [rows], self.shape[1:])[0]
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        return self.owner.take(self.key, rows, self.shape[1:])
