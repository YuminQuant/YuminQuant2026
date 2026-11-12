"""Compatibility entry point for the report sentiment factor workflow."""
from yq_ml_alpha.models.report_sentiment_model import (
    _execute,
    _history_config as _history_config,
)


def execute(config, *, train, predict, resume=False, window_ids=None):
    from yq_ml_alpha.models.base import load_model_class

    model_class = load_model_class(config.model.class_path)
    return _execute(model_class, config, train=train, predict=predict, resume=resume, window_ids=window_ids)
