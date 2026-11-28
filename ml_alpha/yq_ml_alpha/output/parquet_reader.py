"""Process-local Parquet metadata and projected Arrow tables.

Only immutable source data is shared. Each pandas consumer receives independent
arrays; model transforms, split membership and tensors are never cached here.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import json
from threading import RLock

_projection_hints = ContextVar('ml_projection_hints', default={})
_retain_tables = ContextVar("ml_retain_parquet_tables", default=True)
_input_versions = ContextVar("ml_parquet_input_versions", default=None)


@contextmanager
def parquet_input_snapshot():
    """Reuse successful HDFS stats for immutable inputs within one load."""
    if _input_versions.get() is not None:
        yield
        return
    token = _input_versions.set({})
    try:
        yield
    finally:
        _input_versions.reset(token)


@contextmanager
def transient_parquet_tables():
    """Keep metadata but avoid a second resident copy of normalized snapshots."""
    token = _retain_tables.set(False)
    try:
        yield
    finally:
        _retain_tables.reset(token)


@contextmanager
def projected_inputs(requests):
    """Union columns only; each caller still receives its own projected table."""
    token = _projection_hints.set({str(path): tuple(dict.fromkeys(columns))
                                   for path, columns in requests.items()})
    try:
        yield
    finally:
        _projection_hints.reset(token)


class ParquetReadCache:
    def __init__(self):
        self.tables = OrderedDict()
        self.metadata = OrderedDict()
        self.table_keys = {}
        self.versions_by_path = {}
        self.table_bytes = self.metadata_bytes = 0
        self.hits = self.misses = self.evictions = self.opens = self.footer_reads = 0
        self.lock = RLock()
        self.read_locks = [RLock() for _ in range(256)]

    def identity(self, path):
        from . import artifact_io as io
        path = io.read_path(path).absolute()
        backend = io.backend(path)
        versions = _input_versions.get() if backend is not None else None
        key = str(path), id(backend)
        if versions is not None and key in versions:
            return versions[key]
        try:
            info = io.stat(path)
        except FileNotFoundError:
            self.invalidate(path)
            raise
        version = (*key, info.st_size, info.st_mtime_ns)
        if versions is not None:
            versions[key] = version
        return version

    def invalidate(self, path=None):
        with self.lock:
            versions = _input_versions.get()
            if versions is not None:
                if path is None:
                    versions.clear()
                else:
                    from . import artifact_io as io
                    versions.pop((str(path), id(io.backend(path))), None)
            if path is None:
                self.tables.clear()
                self.metadata.clear()
                self.table_keys.clear()
                self.versions_by_path.clear()
                self.table_bytes = self.metadata_bytes = 0
                return
            for version in self.versions_by_path.pop(str(path), ()):
                metadata = self.metadata.pop(version, None)
                if metadata is not None:
                    self.metadata_bytes -= metadata[1]
                for key in self.table_keys.pop(version, ()):
                    _, size = self.tables.pop(key)
                    self.table_bytes -= size

    def get(self, entries, key):
        with self.lock:
            value = entries.get(key)
            if value is not None:
                entries.move_to_end(key)
                return value[0]
        return None

    def put(self, entries, key, value, size, kind):
        with self.lock:
            attr = kind + "_bytes"
            if key in entries:
                _, old_size = entries.pop(key)
                setattr(self, attr, getattr(self, attr) - old_size)
            entries[key] = (value, size)
            version = key[0] if kind == "table" else key
            if kind == "table":
                self.table_keys.setdefault(version, set()).add(key)
            self.versions_by_path.setdefault(version[0], set()).add(version)
            setattr(self, attr, getattr(self, attr) + size)

    @contextmanager
    def open(self, path, version=None, *, pre_buffer=False):
        import pyarrow.parquet as pq
        from . import artifact_io as io
        version = version or self.identity(path)
        metadata = self.get(self.metadata, version)
        # Keep the stream alive for the parser; the adapter limits native calls.
        with io.open_file(path, "rb") as stream:
            self.opens += 1
            with pq.ParquetFile(stream, metadata=metadata,
                                pre_buffer=pre_buffer and io.backend(path) is not None) as reader:
                if metadata is None:
                    self.footer_reads += 1
                    self.put(self.metadata, version, reader.metadata,
                             max(reader.metadata.serialized_size, 1024), "metadata")
                yield reader

    def schema(self, path):
        with self.read_locks[hash(str(path)) % len(self.read_locks)]:
            version = self.identity(path)
            metadata = self.get(self.metadata, version)
            if metadata is not None:
                return metadata.schema.to_arrow_schema()
            with self.open(path, version) as reader:
                return reader.schema_arrow

    def read(self, path, columns=None, *, allow_missing=False, required=(),
             use_threads=True, pandas_metadata=False, pre_buffer=False):
        # Shared-input workers must not all miss and reopen the same file.
        with self.read_locks[hash(str(path)) % len(self.read_locks)]:
            return self._read(path, columns, allow_missing=allow_missing, required=required,
                              use_threads=use_threads, pandas_metadata=pandas_metadata, pre_buffer=pre_buffer)

    def _read(self, path, columns=None, *, allow_missing=False, required=(),
              use_threads=True, pandas_metadata=False, pre_buffer=False):
        from . import artifact_io as io
        hint = _projection_hints.get().get(str(io.read_path(path).absolute()))
        if hint and columns is not None and allow_missing and not set(hint) <= set(columns):
            union = tuple(dict.fromkeys([*columns, *hint]))
            table = self.read(path, union, allow_missing=True, required=required,
                              use_threads=use_threads, pandas_metadata=pandas_metadata, pre_buffer=pre_buffer)
            return table.select([name for name in columns if name in table.column_names])
        version = self.identity(path)
        requested = None if columns is None else tuple(dict.fromkeys(columns))
        key = (version, requested, allow_missing, tuple(required), pandas_metadata)
        table = self.get(self.tables, key)
        if table is not None:
            self.hits += 1
            return table
        metadata = self.get(self.metadata, version)
        if requested is not None and metadata is not None:
            available = set(metadata.schema.to_arrow_schema().names)
            missing = set(required) - available
            if missing:
                raise ValueError(f"missing columns {sorted(missing)} in {path}")
            selected = [column for column in requested if column in available] if allow_missing else list(requested)
            if pandas_metadata and metadata.metadata and b"pandas" in metadata.metadata:
                for index in json.loads(metadata.metadata[b"pandas"]).get("index_columns", []):
                    if isinstance(index, str) and index not in selected:
                        selected.append(index)
            with self.lock:
                for candidate in self.table_keys.get(version, ()):
                    value, _ = self.tables[candidate]
                    if candidate[-1] == pandas_metadata and set(selected) <= set(value.column_names):
                        table = value.select(selected)
                        self.tables.move_to_end(candidate)
                        self.hits += 1
                        return table
        self.misses += 1
        with self.open(path, version, pre_buffer=pre_buffer) as reader:
            available = set(reader.schema_arrow.names)
            missing_required = set(required) - available
            if missing_required:
                raise ValueError(f"missing columns {sorted(missing_required)} in {path}")
            selected = requested
            if requested is not None and allow_missing:
                selected = [column for column in requested if column in available]
            table = reader.read(columns=selected, use_threads=use_threads,
                                use_pandas_metadata=pandas_metadata)
        # Account for referenced buffers, including dictionary/zero-copy parents.
        if _retain_tables.get():
            self.put(self.tables, key, table, max(table.get_total_buffer_size(), 1), "table")
        return table

    def info(self):
        with self.lock:
            return {name: getattr(self, name) for name in (
                "table_bytes", "metadata_bytes",
                "hits", "misses", "evictions", "opens", "footer_reads")}


_cache = None
_cache_lock = RLock()


def cache():
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = ParquetReadCache()
        return _cache


def to_frame(table):
    # Never expose writable views into shared Arrow buffers to a caller.
    frame = table.to_pandas()
    return frame.copy(deep=True)
