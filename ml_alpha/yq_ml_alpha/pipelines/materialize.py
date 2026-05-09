from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    if not config.materialize.cache_samples:
        raise ValueError("set [materialize].cache_samples = true to write debug sample cache")
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    outputs = []
    for split, date_range, include_label, frequency in [
        ("train", config.dates.train, True, config.sample.train_frequency or config.sample.frequency),
        ("valid", config.dates.valid, True, config.sample.train_frequency or config.sample.frequency),
        ("predict", config.dates.predict, False, config.sample.predict_frequency or config.sample.frequency),
    ]:
        dates = sample_dates(calendar, date_range, frequency)
        if not dates:
            continue
        frame = dataset.load(dates, include_label=include_label).frame
        path = Path(config.materialize.cache_dir) / f"{split}_{dates[0]}_{dates[-1]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        outputs.append(path)
    return outputs
