from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.pipelines import factor, materialize, model


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    return _select(config).run_config(config)


def train_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    return _select(config).train_config(config)


def predict_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    return _select(config).predict_config(config)


def materialize_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    if _is_factor(config):
        factor._ensure_factor_config(config)
    else:
        model._ensure_model_config(config)
    return materialize.run_config(config)


def _select(config: MlAlphaConfig):
    if _is_factor(config):
        return factor
    return model


def _is_factor(config: MlAlphaConfig) -> bool:
    return config.factor_id is not None or config.output.kind == "factor"
