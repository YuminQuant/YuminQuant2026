"""Small, RAM-only column snapshots scoped to one prediction invocation."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

from yq_ml_alpha.output import artifact_io


_active = ContextVar("prediction_column_catalog", default=None)


class ColumnCatalog:
    def __init__(self, max_entries=32):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.lock = RLock()
        self.hits = self.scans = 0

    def columns(self, root, end_date, scan):
        key = str(artifact_io.read_path(root).absolute()), end_date
        with self.lock:
            if key in self.entries:
                self.hits += 1
                self.entries.move_to_end(key)
                return self.entries[key]
            columns = tuple(scan())
            self.scans += 1
            self.entries[key] = columns
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
            return columns


def discovered_columns(root, end_date, scan):
    catalog = _active.get()
    return tuple(scan()) if catalog is None else catalog.columns(root, end_date, scan)


@contextmanager
def column_discovery_scope():
    """Freeze only ordered column names, never tensors or file handles.

    A new invocation scans again, so changes between prediction runs remain
    visible. Exclusions are applied by each caller after this common snapshot.
    Training and other callers outside this scope retain fresh discovery.
    """
    existing = _active.get()
    if existing is not None:
        yield existing
        return
    catalog = ColumnCatalog()
    token = _active.set(catalog)
    try:
        yield catalog
    finally:
        _active.reset(token)
        catalog.entries.clear()
