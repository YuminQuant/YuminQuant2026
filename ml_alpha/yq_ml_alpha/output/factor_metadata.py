from __future__ import annotations

import json
import os
from contextlib import contextmanager
import time
from pathlib import Path

import pandas as pd

from yq_ml_alpha.config import MlAlphaConfig


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
]


def write_factor_metadata(config: MlAlphaConfig) -> Path | None:
    if config.output.kind != "factor" or not config.output.write_metadata:
        return None
    factor_id = config.factor_id or config.output.id
    if not factor_id:
        raise ValueError("factor output requires factor_id or output.id")
    path = Path(config.output.root) / "factor_metadata.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_lock(path.parent):
        source = path.with_name("factor_metadata.ml_alpha.parquet")
        native = path.with_name("factor_metadata.rust.parquet")
        legacy = _read_existing(path)
        external_mask = legacy.tags_json.map(lambda value: "model_generated" in json.loads(value or "[]"))
        rows = _read_existing(source) if source.exists() else legacy.loc[external_mask]
        native_rows = _read_existing(native) if native.exists() else legacy.loc[~external_mask]
        rows = rows.loc[rows["factor_id"] != factor_id].copy()
        rows = pd.concat([rows, pd.DataFrame([_metadata_row(config, factor_id)])], ignore_index=True)
        if set(rows.factor_id) & set(native_rows.factor_id):
            raise ValueError("Rust and ml_alpha factor metadata IDs collide")
        _write_atomic(source, rows)
        _write_atomic(native, native_rows)
        _write_atomic(path, pd.concat([native_rows, rows], ignore_index=True))
    return path


@contextmanager
def metadata_lock(root: Path):
    path = root / "factor_metadata.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"metadata writer is active (or stale lock after crash): {path}") from exc
    try:
        os.close(fd)
        yield
    finally:
        path.unlink()


def _write_atomic(path: Path, rows: pd.DataFrame) -> None:
    output = rows.sort_values(["asset_class", "frequency", "factor_id"]).reset_index(drop=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        output[METADATA_COLUMNS].to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=METADATA_COLUMNS)
    frame = pd.read_parquet(path)
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
    }
