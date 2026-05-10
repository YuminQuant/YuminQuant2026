from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.train import _context, _new_model, _train_frequency


def run(config_path: str | Path):
    config = load_config(config_path)
    if not config.tuning.enabled:
        raise ValueError("[tuning].enabled is false")
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    model = _new_model(config)
    train_frequency = _train_frequency(config)
    train_dates = sample_dates(calendar, config.dates.train, train_frequency)
    valid_dates = sample_dates(calendar, config.dates.valid, train_frequency) if config.dates.valid is not None else []
    train_bundle = dataset.load(train_dates, include_label=True)
    valid_bundle = dataset.load(valid_dates, include_label=True)
    context = _context(config, "tuning", train_bundle.feature_columns)

    def data_factory():
        return train_bundle.frame.copy(), valid_bundle.frame.copy()

    return model.tune(data_factory, context)
