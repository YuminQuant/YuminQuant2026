from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.train import _predict_frequency, _train_frequency


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    if not config.materialize.cache_samples:
        raise ValueError("set [materialize].cache_samples = true to write debug sample cache")
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    outputs = []
    splits = [("train", config.dates.train, True, _train_frequency(config))]
    if config.dates.valid is not None:
        splits.append(("valid", config.dates.valid, True, _train_frequency(config)))
    if config.dates.predict is not None:
        splits.append(("predict", config.dates.predict, False, _predict_frequency(config)))
    for split, date_range, include_label, frequency in splits:
        dates = sample_dates(calendar, date_range, frequency)
        if not dates:
            continue
        frame = dataset.load(dates, include_label=include_label).frame
        path = Path(config.materialize.cache_dir) / f"{split}_{dates[0]}_{dates[-1]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        outputs.append(path)
    return outputs
