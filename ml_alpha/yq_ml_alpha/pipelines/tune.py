from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.train import _context, _new_model


def run(config_path: str | Path):
    config = load_config(config_path)
    if not config.tuning.enabled:
        raise ValueError("[tuning].enabled is false")
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    model = _new_model(config)
    context = _context(config, "tuning")

    def data_factory():
        train_dates = sample_dates(calendar, config.dates.train, config.sample.frequency)
        valid_dates = sample_dates(calendar, config.dates.valid, config.sample.frequency)
        return (
            dataset.load(train_dates, include_label=True).frame,
            dataset.load(valid_dates, include_label=True).frame,
        )

    return model.tune(data_factory, context)
