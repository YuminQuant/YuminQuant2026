from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.runtime import _load_bundle, _predict_frequency, _train_frequency
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    return run_config(config)


def run_config(config: MlAlphaConfig) -> list[Path]:
    if config.features.type == "derived_logsig":
        raise ValueError("Logsignature is derived data; use Rust derive-logsig before factor-run")
    if config.features.type == "logsig_signature":
        return materialize_logsignature(config)
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
        frame = _load_bundle(config, dataset, calendar, dates, include_label=include_label).frame
        path = Path(config.materialize.cache_dir) / f"{split}_{dates[0]}_{dates[-1]}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        outputs.append(path)
    return outputs


def materialize_logsignature(config: MlAlphaConfig) -> list[Path]:
    """Persist one feature date at a time; never assemble training samples here."""
    calendar = TradingCalendar.load(config.data_root)
    params = dict(config.features.params, read_only=False)
    if not params.get("feature_cache_dir"):
        raise ValueError("logsignature materialization requires feature_cache_dir")
    provider = LogsigSignatureProvider(config.features.root, config.features.columns, params)
    provider.set_calendar_dates(calendar.dates)
    dates = set(sample_dates(calendar, config.dates.train, _train_frequency(config)))
    if config.dates.valid is not None:
        dates.update(sample_dates(calendar, config.dates.valid, _train_frequency(config)))
    if config.dates.predict is not None:
        dates.update(sample_dates(calendar, config.dates.predict, _predict_frequency(config)))
    dates = sorted(dates)
    provider.set_cache_days_for_target_dates(dates)
    outputs = []
    for index, date in enumerate(dates, 1):
        frame = provider.load(date)
        print(f"logsignature materialize [{index}/{len(dates)}] date={date} rows={len(frame)}", flush=True)
        outputs.append(provider._feature_cache_path(date))
        del frame
    return outputs
