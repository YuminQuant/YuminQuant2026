"""Ordered factor columns shared within a process or pinned by a submission."""
from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
from contextvars import ContextVar
import json
import os
import sys
from threading import RLock
from pathlib import Path

from yq_ml_alpha.output import artifact_io


_active = ContextVar("prediction_column_catalog", default=None)
SUBMISSION_CATALOG_ENV = "YQ_FACTOR_COLUMN_CATALOG"
FACTOR_FRAME_TYPES = {"factor_frame", "factor_sequence", "graph_factor_frame", "hist_factor_frame",
                      "concept_factor_sequence", "analyst_factor_sequence"}


def _catalog_key(root, end_date):
    return artifact_io.identity_path(artifact_io.read_path(root).absolute()), end_date


class ColumnCatalog:
    def __init__(self, max_entries=32):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.lock = RLock()
        self.hits = self.scans = 0
        self.resources = {}

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
            if "selector" not in entry and (entry["root"], entry["end_date"]) == key:
                return tuple(entry["columns"])
    catalog = _active.get()
    return tuple(scan()) if catalog is None else catalog.columns(root, end_date, scan)


def discovery_resource(key, load):
    catalog = _active.get()
    if catalog is None:
        return load()
    with catalog.lock:
        if key not in catalog.resources:
            catalog.resources[key] = load()
        return catalog.resources[key]


def _selector(config):
    params = config.features.params
    include_generated = params.get("include_model_generated", False)
    if not isinstance(include_generated, bool):
        raise ValueError("features.params.include_model_generated must be boolean")
    tags = []
    for name in ("include_tags_any", "exclude_tags_any"):
        values = params.get(name, [])
        if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"features.params.{name} must be a list of nonempty tags")
        tags.append(sorted(set(values)))
    selector = [config.features.columns, config.output.asset, config.output.frequency,
                include_generated, *tags]
    return json.dumps(selector, separators=(",", ":"))


def _resolved(config, selector):
    from yq_ml_alpha.data.stores import discover_value_columns, is_all_column_request

    requested, asset, frequency, include_generated, include_tags, exclude_tags = json.loads(selector)
    include_tags, exclude_tags = set(include_tags), set(exclude_tags)
    allowed = set() if include_tags else None
    excluded = set()
    metadata = Path(config.data_root) / "factors" / "factor_metadata.parquet"
    frame = discovery_resource(("factor_metadata", metadata),
                               lambda: artifact_io.read_parquet(metadata) if artifact_io.exists(metadata) else None)
    if frame is not None:
        for field, value in (("asset_class", asset), ("frequency", frequency)):
            if field in frame.columns:
                frame = frame.loc[frame[field].eq(value)]
        for row in frame[["output_column", "tags_json"]].itertuples(index=False):
            tags = set(json.loads(row.tags_json or "[]"))
            if ("deprecated" in tags or ("model_generated" in tags and not include_generated)
                    or tags & exclude_tags):
                excluded.add(row.output_column)
            elif allowed is not None and tags & include_tags:
                allowed.add(row.output_column)
    elif allowed is not None or exclude_tags:
        raise FileNotFoundError(f"factor tag selection requires metadata: {metadata}")
    all_columns = is_all_column_request(requested)
    if all_columns:
        requested = sorted(allowed) if allowed is not None else discover_value_columns(
            config.features.root, config.dates.train[1])
    return tuple(column for column in requested if column not in excluded
                 and (allowed is None or column in allowed))


def factor_columns(config):
    selector = _selector(config)
    root, end = _catalog_key(config.features.root, config.dates.train[1])
    columns = _selected_columns(config, root, end, selector)
    result = [column for column in columns if column != config.output.id]
    if not result:
        raise ValueError("no eligible feature columns after metadata exclusions")
    return result


def _selected_columns(config, root, end, selector):
    snapshot = json.loads(os.environ.get(SUBMISSION_CATALOG_ENV) or "[]")
    for entry in snapshot:
        if (entry["root"], entry["end_date"], entry.get("selector")) == (root, end, selector):
            return entry["columns"]
    return discovery_resource(("factor_columns", root, end, selector), lambda: _resolved(config, selector))


def training_column_catalog(configs):
    """Resolve schema and tag selectors using the same code as workers."""
    entries = {}
    with column_discovery_scope(), redirect_stdout(sys.stderr):
        for config in configs:
            if config.features.type not in FACTOR_FRAME_TYPES or artifact_io.backend(config.features.root) is None:
                continue
            selector = _selector(config)
            root, end = _catalog_key(config.features.root, config.dates.train[1])
            key = root, end, selector
            if key not in entries:
                entries[key] = dict(root=root, end_date=end, selector=selector,
                                    columns=_selected_columns(config, root, end, selector))
    return list(entries.values())


@contextmanager
def column_discovery_scope():
    """Freeze only ordered column names, never tensors or file handles.

    Each invocation sees a consistent read-only metadata snapshot.
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
        catalog.resources.clear()
