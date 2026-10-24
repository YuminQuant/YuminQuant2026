"""Optional artifact backend registered by a deployment launcher; local by default.

Backends own atomic commit and distributed locks. Registration precedes worker
threads; model algorithms and input datasets do not depend on the backend.
"""
from contextlib import contextmanager
from pathlib import Path
import os
from types import SimpleNamespace

_backend = None

def read_path(path):
    for line in os.environ.get('YQ_METADATA_SNAPSHOTS','').splitlines():
        source, target = line.split('\t',1)
        if Path(path) == Path(source):
            return Path(target)
    return Path(path)


def register(backend):
    global _backend
    if _backend is not None:
        raise RuntimeError("artifact backend already registered")
    _backend = backend


def backend(path):
    return _backend if _backend is not None and _backend.handles(Path(path)) else None


def exists(path):
    path = read_path(path)
    store = backend(path)
    return store.exists(Path(path)) if store else Path(path).exists()


def identity_path(path):
    """Deployment-stable input identity; local model fingerprints stay unchanged."""
    identify = getattr(_backend, "identity_path", None)
    return identify(Path(path)) if identify else str(path)


def identity_values(value):
    if isinstance(value, Path):
        return identity_path(value)
    if isinstance(value, dict):
        return {key: identity_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [identity_values(item) for item in value]
    return value


def mkdir(path, **kwargs):
    store = backend(path)
    return store.mkdir(Path(path)) if store else Path(path).mkdir(**kwargs)


@contextmanager
def open_file(path, mode="r", **kwargs):
    if mode.startswith('r'): path = read_path(path)
    else: invalidate_parquet(path)
    store = backend(path)
    try:
        with (store.open(Path(path), mode, **kwargs) if store else Path(path).open(mode, **kwargs)) as stream:
            yield stream
    finally:
        if not mode.startswith('r'):
            invalidate_parquet(path)


def write_text(path, text, **kwargs):
    with open_file(path, "w", **kwargs) as stream:
        return stream.write(text)


def read_text(path, **kwargs):
    with open_file(path, "r", **kwargs) as stream:
        return stream.read()


def write_parquet(frame, path, **kwargs):
    invalidate_parquet(path)
    kwargs.setdefault("compression", "snappy")
    if backend(path):
        with open_file(path, "wb") as stream:
            return frame.to_parquet(stream, **kwargs)
    try:
        return frame.to_parquet(path, **kwargs)
    finally:
        invalidate_parquet(path)


def read_parquet(path, **kwargs):
    path = read_path(path)
    import pandas as pd
    if set(kwargs) <= {"columns", "use_threads"}:
        from .parquet_reader import cache, to_frame
        return to_frame(cache().read(path, pandas_metadata=True, **kwargs))
    if backend(path):
        with open_file(path, "rb") as stream:
            return pd.read_parquet(stream, **kwargs)
    return pd.read_parquet(path, **kwargs)


def read_arrow_table(path, **kwargs):
    import pyarrow.parquet as pq
    path=read_path(path)
    if set(kwargs) <= {"columns", "use_threads"}:
        from .parquet_reader import cache
        return cache().read(path, **kwargs)
    if backend(path):
        with open_file(path,'rb') as stream:
            return pq.read_table(stream,**kwargs)
    return pq.read_table(path,**kwargs)


def parquet_columns(path):
    return list(parquet_schema(path).names)


def glob(path, pattern):
    store = backend(path)
    return store.glob(Path(path), pattern) if store else Path(path).glob(pattern)


def torch_save(value, path, **kwargs):
    import torch
    if backend(path):
        with open_file(path, "wb") as stream:
            return torch.save(value, stream, **kwargs)
    return torch.save(value, path, **kwargs)


def torch_load(path, **kwargs):
    import torch
    if backend(path):
        with open_file(path, "rb") as stream:
            return torch.load(stream, **kwargs)
    return torch.load(path, **kwargs)


def remote_enabled():
    return os.environ.get("YQ_STORAGE_BACKEND", "local") == "hdfs"


def stat(path):
    path = read_path(path)
    store = backend(path)
    if not store:
        return Path(path).stat()
    info = store.stat(Path(path))
    return SimpleNamespace(st_size=info["size"], st_mtime_ns=info["modified_ms"] * 1_000_000,
                           st_mtime=info["modified_ms"] / 1000)


def is_file(path):
    store = backend(path)
    return store.is_file(Path(path)) if store else Path(path).is_file()


def is_dir(path):
    store = backend(path)
    return store.is_dir(Path(path)) if store else Path(path).is_dir()


def unlink(path, missing_ok=False):
    invalidate_parquet(path)
    store = backend(path)
    if store:
        if missing_ok and not store.exists(Path(path)):
            return
        return store.remove(Path(path))
    return Path(path).unlink(missing_ok=missing_ok)


def replace(path, destination):
    invalidate_parquet(path)
    invalidate_parquet(destination)
    store = backend(path)
    if store:
        store.replace(Path(path), Path(destination))
        return Path(destination)
    return Path(path).replace(destination)


def read_bytes(path):
    with open_file(path, "rb") as stream:
        return stream.read()


def write_bytes(path, payload):
    with open_file(path, "wb") as stream:
        return stream.write(payload)


def parquet_schema(path):
    from .parquet_reader import cache
    return cache().schema(read_path(path))


def open_parquet(path):
    """Managed reader: schema, row count and projected reads share one handle."""
    from .parquet_reader import cache
    return cache().open(read_path(path))


def read_projected_frame(path, columns, *, required=()):
    from .parquet_reader import cache, to_frame
    return to_frame(cache().read(read_path(path), columns, allow_missing=True,
                                 required=required, pandas_metadata=True))


def invalidate_parquet(path=None):
    from .parquet_reader import cache
    cache().invalidate(None if path is None else read_path(path).absolute())


def parquet_cache_info():
    from .parquet_reader import cache
    return cache().info()


def parquet_file(path):
    path = read_path(path)
    import pyarrow.parquet as pq
    store = backend(path)
    return pq.ParquetFile(store.input_file(Path(path)) if store else path)
