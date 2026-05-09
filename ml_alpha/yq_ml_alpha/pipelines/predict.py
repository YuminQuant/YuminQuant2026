from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.pipelines.train import predict_only


def run(config_path: str | Path):
    return predict_only(config_path)
