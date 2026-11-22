from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.runtime import _predict_frequency, _train_frequency
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    return run_config(config)


def run_config(config: MlAlphaConfig) -> list[Path]:
    if config.features.type == "derived_logsigrnn":
        raise ValueError("Logsig-RNN inputs are derived data; use Rust derive-logsigrnn before factor-run")
    if config.features.type in {"market_style_sequence", "rawbar_sequence"}:
        raise ValueError("daily market sequences use RAM only; factor-materialize is not supported")
    if config.features.type == "analyst_raw40_sequence":
        raise ValueError("Analyst GAT inputs are derived data; use Rust derive-analyst-gat-inputs before factor-run")
    if config.features.type == "derived_chip_age":
        raise ValueError("Chip-age tensors are derived data; use derive-chip-age before factor-run")
    if config.features.type == "derived_logsig":
        raise ValueError("Logsignature is derived data; use Rust derive-logsig before factor-run")
    if config.features.type == "logsig_signature":
        return materialize_logsignature(config)
    raise ValueError("Sample export has been removed; materialize supports only logsig_signature features")


def materialize_logsignature(config: MlAlphaConfig) -> list[Path]:
    """Persist one feature date at a time; never assemble training samples here."""
    calendar = TradingCalendar.for_config(config)
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
