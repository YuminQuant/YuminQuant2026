"""Process-local, byte-bounded Parquet metadata and projected Arrow tables.

Only immutable source data is shared. Each pandas consumer receives independent
arrays; model transforms, split membership and tensors are never cached here.
"""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from threading import RLock

_projection_hints = ContextVar('ml_projection_hints', default={})


@contextmanager
def projected_inputs(requests):
    """Union columns only; each caller still receives its own projected table."""
    token = _projection_hints.set({str(path): tuple(dict.fromkeys(columns))
                                   for path, columns in requests.items()})
    try:
        yield
    finally:
        _projection_hints.reset(token)


def _budget(name, default):
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


class ParquetReadCache:
    def __init__(self):
        self.table_limit = _budget("YQ_PARQUET_CACHE_BYTES", 64 * 1024**2)
        self.metadata_limit = _budget("YQ_PARQUET_METADATA_BYTES", 8 * 1024**2)
        self.tables = OrderedDict()
        self.metadata = OrderedDict()
        self.table_bytes = self.metadata_bytes = 0
        self.hits = self.misses = self.evictions = self.opens = self.footer_reads = 0
        self.lock = RLock()

    def identity(self, path):
        from . import artifact_io as io
        path = io.read_path(path).absolute()
        try:
            info = io.stat(path)
        except FileNotFoundError:
            self.invalidate(path)
            raise
        # A stat on each access also detects rewrites made outside this process.
        return (str(path), id(io.backend(path)), info.st_size, info.st_mtime_ns)

    def invalidate(self, path=None):
        with self.lock:
            for entries, kind in ((self.tables, "table"), (self.metadata, "metadata")):
                for key in list(entries):
                    version = key[0] if kind == "table" else key
                    if path is None or version[0] == str(path):
                        _, size = entries.pop(key)
                        attr = kind + "_bytes"
                        setattr(self, attr, getattr(self, attr) - size)

    def get(self, entries, key):
        with self.lock:
            value = entries.get(key)
            if value is not None:
                entries.move_to_end(key)
                return value[0]
        return None

    def put(self, entries, key, value, size, kind):
        limit = getattr(self, kind + "_limit")
        if not limit or size > limit:
            return
        with self.lock:
            attr = kind + "_bytes"
            if key in entries:
                _, old_size = entries.pop(key)
                setattr(self, attr, getattr(self, attr) - old_size)
            while entries and (getattr(self, attr) + size > limit or len(entries) >= 4096):
                _, (_, old_size) = entries.popitem(last=False)
                setattr(self, attr, getattr(self, attr) - old_size)
                self.evictions += 1
            entries[key] = (value, size)
            setattr(self, attr, getattr(self, attr) + size)

    @contextmanager
    def open(self, path, version=None):
        import pyarrow.parquet as pq
        from . import artifact_io as io
        version = version or self.identity(path)
        metadata = self.get(self.metadata, version)
        # Keep HDFS IO semaphore and NativeFile alive for the whole operation.
        with io.open_file(path, "rb") as stream:
            self.opens += 1
            with pq.ParquetFile(stream, metadata=metadata, pre_buffer=False) as reader:
                if metadata is None:
                    self.footer_reads += 1
                    self.put(self.metadata, version, reader.metadata,
                             max(reader.metadata.serialized_size, 1024), "metadata")
                yield reader

    def schema(self, path):
        version = self.identity(path)
        metadata = self.get(self.metadata, version)
        if metadata is not None:
            return metadata.schema.to_arrow_schema()
        with self.open(path, version) as reader:
            return reader.schema_arrow

    def read(self, path, columns=None, *, allow_missing=False, required=(),
             use_threads=True, pandas_metadata=False):
        from . import artifact_io as io
        hint = _projection_hints.get().get(str(io.read_path(path).absolute()))
        if hint and columns is not None and allow_missing and not set(hint) <= set(columns):
            union = tuple(dict.fromkeys([*columns, *hint]))
            table = self.read(path, union, allow_missing=True, required=required,
                              use_threads=use_threads, pandas_metadata=pandas_metadata)
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
                for candidate, (value, _) in reversed(self.tables.items()):
                    if candidate[0] == version and candidate[-1] == pandas_metadata and set(selected) <= set(value.column_names):
                        table = value.select(selected)
                        self.tables.move_to_end(candidate)
                        self.hits += 1
                        return table
        self.misses += 1
        with self.open(path, version) as reader:
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
        self.put(self.tables, key, table, max(table.get_total_buffer_size(), 1), "table")
        return table

    def info(self):
        with self.lock:
            return {name: getattr(self, name) for name in (
                "table_limit", "metadata_limit", "table_bytes", "metadata_bytes",
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
    if table.get_total_buffer_size() <= cache().table_limit and cache().table_limit:
        return frame.copy(deep=True)
    # Oversized/disabled-cache reads have no shared resident table to protect.
    return frame
