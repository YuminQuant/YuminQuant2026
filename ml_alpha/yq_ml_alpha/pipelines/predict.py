from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.pipelines import dispatch


def run(config_path: str | Path):
    return dispatch.predict_only(config_path)
