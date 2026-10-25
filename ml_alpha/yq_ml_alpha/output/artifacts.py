from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
from yq_ml_alpha.output import artifact_io


def window_artifact_path(artifact_dir: str | Path, window_id: str) -> Path:
    path = Path(artifact_dir) / window_id / "model.pkl"
    artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
    return path


def artifact_identity(config, window, dataset) -> dict:
    label_identity = asdict(config.label)
    # Keep existing single-label model fingerprints unchanged.
    if not config.label.auxiliary:
        label_identity.pop("auxiliary", None)
    identity = {
        "pipeline_version": 3,
        "training_information_cutoff": window.training_cutoff_date,
        "label_maturity_policy": "label_end_at_or_before_training_close",
        "model": {"class": config.model.class_path, "params": config.model.params, "search": config.model.search},
        "features": asdict(config.features),
        "feature_columns": list(dataset.feature_provider.feature_columns),
        "preprocess": asdict(config.preprocess),
        "label": label_identity,
        "filters": asdict(config.filters),
        "universe": asdict(config.universe),
        "data_root": artifact_io.identity_path(config.data_root),
        "data_version": config.data_version,
        "train_dates": window.train_dates,
        "valid_dates": window.valid_dates,
    }
    if getattr(window, "activation_date", None) is not None:
        identity["prediction_activation"] = {
            "policy": config.train_scheme.prediction_activation,
            "training_cutoff_date": window.training_cutoff_date,
            "activation_date": window.activation_date,
        }
    auxiliary = config.features.params.get("supervision_lookahead_days", 0)
    if auxiliary:
        identity["auxiliary_supervision"] = {
            "lookahead_days": auxiliary,
            "policy": "max_label_and_auxiliary_horizon_at_training_close_and_split",
        }
    if config.train_scheme.validation_unit == "rows":
        identity["validation_split"] = {
            "unit": "rows", "ratio": config.train_scheme.validation_ratio,
            "sort_columns": ["trade_date", "ts_code"],
            "boundary": "floor(total_rows * (1 - ratio))",
            "after": "filters_and_missing_label_removal",
        }
    dataset_identity = getattr(dataset.feature_provider, "dataset_identity", None)
    if callable(dataset_identity):
        identity["derived_dataset"] = dataset_identity()
    return artifact_io.identity_values(identity)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def save_manifest(path: Path, config, window, dataset) -> None:
    identity = artifact_identity(config, window, dataset)
    payload = {"identity": identity, "fingerprint": hashlib.sha256(_canonical(identity).encode()).hexdigest()}
    row_split = getattr(dataset, "row_split_info", None)
    if config.train_scheme.validation_unit == "rows" and row_split and row_split["window_id"] == window.window_id:
        payload["row_split"] = row_split
    destination = path.with_suffix(".manifest.json")
    if artifact_io.backend(destination):
        artifact_io.write_text(destination, _canonical(payload), encoding="utf-8")
        return
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    data_io.write_text(temporary, _canonical(payload), encoding="utf-8")
    data_io.replace(temporary, destination)


def validate_manifest(path: Path, config, window, dataset) -> None:
    manifest = path.with_suffix(".manifest.json")
    if not artifact_io.exists(path) or not artifact_io.exists(manifest):
        raise ValueError(f"missing model/manifest; retrain before resume or predict: {path}")
    payload = json.loads(artifact_io.read_text(manifest, encoding="utf-8"))
    expected = hashlib.sha256(_canonical(artifact_identity(config, window, dataset)).encode()).hexdigest()
    if payload.get("fingerprint") != expected:
        raise ValueError(f"model config/features/data version changed; retrain: {path}")
