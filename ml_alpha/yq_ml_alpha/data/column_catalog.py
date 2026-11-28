"""Ordered factor columns shared within a process or pinned by a submission."""
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
import json
import os
import sys
from threading import RLock

from yq_ml_alpha.output import artifact_io


_active = ContextVar("prediction_column_catalog", default=None)
SUBMISSION_CATALOG_ENV = "YQ_FACTOR_COLUMN_CATALOG"


def _catalog_key(root, end_date):
    return str(artifact_io.read_path(root).absolute()), end_date


class ColumnCatalog:
    def __init__(self, max_entries=32):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.lock = RLock()
        self.hits = self.scans = 0

    def columns(self, root, end_date, scan):
        key = _catalog_key(root, end_date)
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
    snapshot = os.environ.get(SUBMISSION_CATALOG_ENV)
    if snapshot:
        key = _catalog_key(root, end_date)
        for entry in json.loads(snapshot):
            if (entry["root"], entry["end_date"]) == key:
                return tuple(entry["columns"])
    catalog = _active.get()
    return tuple(scan()) if catalog is None else catalog.columns(root, end_date, scan)


def training_column_catalog(configs):
    """Resolve remote all-column schemas once, in the same order as training."""
    from yq_ml_alpha.data.stores import discover_value_columns, is_all_column_request

    types = {"factor_frame", "factor_sequence", "graph_factor_frame", "hist_factor_frame",
             "concept_factor_sequence", "analyst_factor_sequence"}
    entries = {}
    for config in configs:
        features = config.features
        if (features.type not in types or not is_all_column_request(features.columns)
                or features.params.get("include_tags_any") or artifact_io.backend(features.root) is None):
            continue
        key = _catalog_key(features.root, config.dates.train[1])
        if key in entries:
            continue
        with redirect_stdout(sys.stderr):
            columns = discover_value_columns(features.root, key[1])
        entries[key] = dict(root=key[0], end_date=key[1], columns=columns)
    return list(entries.values())


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
