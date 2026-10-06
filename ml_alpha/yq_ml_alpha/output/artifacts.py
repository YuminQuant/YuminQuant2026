from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os


def window_artifact_path(artifact_dir: str | Path, window_id: str) -> Path:
    path = Path(artifact_dir) / window_id / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def artifact_identity(config, window, dataset) -> dict:
    return {
        "pipeline_version": 2,
        "model": {"class": config.model.class_path, "params": config.model.params, "search": config.model.search},
        "features": asdict(config.features),
        "feature_columns": list(dataset.feature_provider.feature_columns),
        "preprocess": asdict(config.preprocess),
        "label": asdict(config.label),
        "filters": asdict(config.filters),
        "universe": asdict(config.universe),
        "data_root": str(config.data_root),
        "data_version": config.data_version,
        "train_dates": window.train_dates,
        "valid_dates": window.valid_dates,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def save_manifest(path: Path, config, window, dataset) -> None:
    identity = artifact_identity(config, window, dataset)
    payload = {"identity": identity, "fingerprint": hashlib.sha256(_canonical(identity).encode()).hexdigest()}
    destination = path.with_suffix(".manifest.json")
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical(payload), encoding="utf-8")
    temporary.replace(destination)


def validate_manifest(path: Path, config, window, dataset) -> None:
    manifest = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest.exists():
        raise ValueError(f"missing model/manifest; retrain before resume or predict: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected = hashlib.sha256(_canonical(artifact_identity(config, window, dataset)).encode()).hexdigest()
    if payload.get("fingerprint") != expected:
        raise ValueError(f"model config/features/data version changed; retrain: {path}")
