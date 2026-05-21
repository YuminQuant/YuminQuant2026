from __future__ import annotations

import json
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
    rows = _read_existing(path)
    rows = rows.loc[rows["factor_id"] != factor_id].copy()
    row = pd.DataFrame([_metadata_row(config, factor_id)])
    output = pd.concat([rows, row], ignore_index=True)
    output = output.sort_values(["asset_class", "frequency", "factor_id"]).reset_index(drop=True)
    tmp = path.with_name(f"{path.name}.tmp")
    output[METADATA_COLUMNS].to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


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
