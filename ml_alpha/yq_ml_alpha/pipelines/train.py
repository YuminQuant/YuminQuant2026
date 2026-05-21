from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.pipelines import dispatch
from yq_ml_alpha.pipelines.runtime import build_windows, _split_by_validation_ratio


def run(config_path: str | Path) -> list[Path]:
    return dispatch.run(config_path)


def train_only(config_path: str | Path) -> list[Path]:
    return dispatch.train_only(config_path)


def predict_only(config_path: str | Path) -> list[Path]:
    return dispatch.predict_only(config_path)
