from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

import json
import os
from contextlib import contextmanager
import time
from pathlib import Path

import pandas as pd

from yq_ml_alpha.config import MlAlphaConfig
from yq_ml_alpha.output import artifact_io
from yq_ml_alpha.data.factor_files import normalize_metadata


METADATA_COLUMNS = [
    "factor_id",
    "aliases_json",
    "version",
    "output_column",
    "name",
    "asset_class",
    "frequency",
    "tags_json",
    "dependencies_json",
    "description",
    "updated_at",
    "storage_partition",
]


def write_factor_metadata(config: MlAlphaConfig) -> Path | None:
    if config.output.kind != "factor" or not config.output.write_metadata:
        return None
    if os.environ.get('YQ_METADATA_MANAGED') == '1':
        return Path(config.output.root) / 'factor_metadata.parquet'
    return write_factor_metadata_many([config])[0]


def write_factor_metadata_many(configs, *, replace_all=False):
    groups = {}
    for config in configs:
        if config.output.kind == 'factor' and config.output.write_metadata:
            groups.setdefault(Path(config.output.root), []).append(config)
    return [_write_metadata_group(root, values, replace_all) for root, values in groups.items()]


def _write_metadata_group(root, configs, replace_all):
    config = configs[0]
    factor_id = config.factor_id or config.output.id
    if not factor_id:
        raise ValueError("factor output requires factor_id or output.id")
    path = Path(config.output.root) / "factor_metadata.parquet"
    artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
    with metadata_lock(path.parent):
        source = path.with_name("factor_metadata.ml_alpha.parquet")
        native = path.with_name("factor_metadata.rust.parquet")
        legacy = _read_existing(path)
        external_mask = legacy.tags_json.map(lambda value: "model_generated" in json.loads(value or "[]"))
        rows = (_read_existing(source) if artifact_io.exists(source) else legacy.loc[external_mask])
        if replace_all:
            rows = rows.iloc[0:0]
        native_rows = _read_existing(native) if artifact_io.exists(native) else legacy.loc[~external_mask]
        additions = [_metadata_row(item, item.factor_id or item.output.id) for item in configs]
        ids = [item['factor_id'] for item in additions]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate ML factor metadata IDs')
        rows = rows.loc[~rows["factor_id"].isin(ids)].copy()
        rows = pd.concat([rows, pd.DataFrame(additions)], ignore_index=True)
        if set(rows.factor_id) & set(native_rows.factor_id):
            raise ValueError("Rust and ml_alpha factor metadata IDs collide")
        normalize_metadata(pd.concat([native_rows, rows], ignore_index=True))
        _write_atomic(source, rows)
        _write_atomic(native, native_rows)
        _write_atomic(path, pd.concat([native_rows, rows], ignore_index=True))
    return path


@contextmanager
def metadata_lock(root: Path):
    from yq_ml_alpha.output.file_lock import daily_file_lock
    with daily_file_lock(root / "factor_metadata"):
        yield


def _write_atomic(path: Path, rows: pd.DataFrame) -> None:
    output = rows.sort_values(["asset_class", "frequency", "factor_id"]).reset_index(drop=True)
    if artifact_io.backend(path):
        artifact_io.write_parquet(output[METADATA_COLUMNS], path, index=False)
        return
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        data_io.write_parquet(output[METADATA_COLUMNS], tmp, index=False)
        data_io.replace(tmp, path)
    finally:
        if data_io.exists(tmp):
            data_io.unlink(tmp)


def _read_existing(path: Path) -> pd.DataFrame:
    if not artifact_io.exists(path):
        return pd.DataFrame(columns=METADATA_COLUMNS)
    frame = artifact_io.read_parquet(path)
    frame = normalize_metadata(frame)
    for column in METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame[METADATA_COLUMNS]


def _metadata_row(config: MlAlphaConfig, factor_id: str) -> dict[str, str]:
    tags = ["e2e", "model_generated"]
    tags.extend(tag for tag in config.tags if tag not in tags)
    model_name = str(config.model.name)
    if model_name:
        tags.append(model_name)
    dependencies = {
        "label": config.label.id,
        "features_type": config.features.type,
        "model_class": config.model.class_path,
    }
    description = (
        f"End-to-end ML factor {factor_id}; model={config.model.name}; "
        f"features={config.features.type}; label={config.label.id}."
    )
    return {
        "factor_id": factor_id,
        "aliases_json": "[]",
        "version": "0.1.0",
        "output_column": factor_id,
        "name": factor_id,
        "asset_class": config.output.asset,
        "frequency": config.output.frequency,
        "tags_json": json.dumps(tags, separators=(",", ":")),
        "dependencies_json": json.dumps([dependencies], separators=(",", ":")),
        "description": description,
        "updated_at": str(int(time.time())),
        "storage_partition": "ml",
    }
