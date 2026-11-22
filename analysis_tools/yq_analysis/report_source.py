"""Report inputs: local files or the deployment's authenticated, shared HDFS client.

Only inputs use this adapter. HTML, figures and scratch remain local. Invoke HDFS
reports through the deployment's authenticated exec wrapper; no secrets are copied.
"""
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import os
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import pandas as pd

_hdfs = None
_config = None


def initialize_hdfs(config=None):
    global _hdfs, _config
    if config is None:
        return
    config = Path(config).resolve()
    sys.path.insert(0, str(config.parent))
    from yq_cluster.storage import Hdfs
    from yq_cluster.paths import FIXED_FACTOR_IO
    settings = tomllib.loads(config.read_text(encoding="utf-8"))["hdfs"]
    settings.update(FIXED_FACTOR_IO)
    _hdfs = Hdfs(settings, os.environ)
    _config = str(config)


def hdfs_config():
    return _config


class HdfsPath:
    def __init__(self, value):
        value = str(value)
        prefix = _hdfs.root + "/"
        if value.startswith(prefix):
            value = value[len(prefix):]
        self.relative = _hdfs.path(value)[len(prefix):]

    def __str__(self):
        return "hdfs:" + _hdfs.path(self.relative)

    def __truediv__(self, value):
        return HdfsPath(str(PurePosixPath(self.relative) / str(value)))

    @property
    def name(self):
        return PurePosixPath(self.relative).name

    @property
    def parent(self):
        return HdfsPath(str(PurePosixPath(self.relative).parent))

    def with_name(self, name):
        return self.parent / name

    def stat(self):
        info = _hdfs.stat(self.relative)
        return SimpleNamespace(st_size=info["size"], st_mtime_ns=info["modified_ms"] * 1_000_000)

    def exists(self):
        return _hdfs.exists(self.relative)

    def is_file(self):
        try:
            return _hdfs.stat(self.relative)["kind"] == "file"
        except FileNotFoundError:
            return False

    def is_dir(self):
        try:
            return _hdfs.stat(self.relative)["kind"] == "directory"
        except FileNotFoundError:
            return False

    def open(self, mode="rb"):
        if mode != "rb":
            raise ValueError("HDFS report inputs are read-only")
        return _hdfs.open_input(self.relative)

    def iterdir(self):
        import pyarrow.fs as afs
        return [HdfsPath(item.path) for item in _hdfs.fs.get_file_info(afs.FileSelector(_hdfs.path(self.relative)))]

    def glob(self, pattern):
        import pyarrow.fs as afs
        root = _hdfs.path(self.relative)
        return [HdfsPath(item.path) for item in _hdfs.fs.get_file_info(afs.FileSelector(root, recursive=True))
                if fnmatch.fnmatchcase(item.path[len(root)+1:], pattern)]


def source_path(value):
    if isinstance(value, HdfsPath):
        return value
    return HdfsPath(value) if _hdfs is not None else Path(value)


def read_parquet(value, **kwargs):
    path = source_path(value)
    with path.open("rb") as stream:
        return pd.read_parquet(stream, **kwargs)


def read_many(paths):
    """Concurrent reads share the deployment's 50-slot/1000-read-op budget."""
    with ThreadPoolExecutor(max_workers=50) as executor:
        return dict(zip(paths, executor.map(lambda p: read_parquet(p) if p.exists() else None, paths)))
