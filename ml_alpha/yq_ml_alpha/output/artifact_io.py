"""Optional artifact backend registered by a deployment launcher; local by default.

Backends own atomic commit and distributed locks. Registration precedes worker
threads; model algorithms and input datasets do not depend on the backend.
"""
from contextlib import contextmanager
from pathlib import Path

_backend = None


def register(backend):
    global _backend
    if _backend is not None:
        raise RuntimeError("artifact backend already registered")
    _backend = backend


def backend(path):
    return _backend if _backend is not None and _backend.handles(Path(path)) else None


def exists(path):
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
    store = backend(path)
    with (store.open(Path(path), mode, **kwargs) if store else Path(path).open(mode, **kwargs)) as stream:
        yield stream


def write_text(path, text, **kwargs):
    with open_file(path, "w", **kwargs) as stream:
        return stream.write(text)


def read_text(path, **kwargs):
    with open_file(path, "r", **kwargs) as stream:
        return stream.read()


def write_parquet(frame, path, **kwargs):
    if backend(path):
        with open_file(path, "wb") as stream:
            return frame.to_parquet(stream, **kwargs)
    return frame.to_parquet(path, **kwargs)


def read_parquet(path, **kwargs):
    import pandas as pd
    if backend(path):
        with open_file(path, "rb") as stream:
            return pd.read_parquet(stream, **kwargs)
    return pd.read_parquet(path, **kwargs)


def parquet_columns(path):
    import pyarrow.parquet as pq
    if backend(path):
        with open_file(path, "rb") as stream:
            return list(pq.read_schema(stream).names)
    return list(pq.read_schema(path).names)


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
