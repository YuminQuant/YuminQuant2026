"""Plan independent fits without slicing configs or changing window identities."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.pipelines.runtime import build_windows


def select_windows(windows, window_ids=None):
    if window_ids is None:
        return windows
    requested = set(window_ids)
    if not requested or len(requested) != len(window_ids):
        raise ValueError("Training window IDs must be nonempty and unique")
    missing = requested - {window.window_id for window in windows}
    if missing:
        raise ValueError(f"Unknown training windows: {sorted(missing)}")
    return [window for window in windows if window.window_id in requested]


def training_windows(config, calendar, window_ids=None):
    if config.features.type == "analyst_report_events":
        from yq_ml_alpha.pipelines.report_sentiment import _history_config
        config = _history_config(config)
    return select_windows(build_windows(config, calendar), window_ids)


def training_config_for_windows(config, window_ids):
    if window_ids is not None and len(window_ids) == 1:
        # Sample caches are operational, not part of model/feature identity.
        config = replace(config, materialize=replace(config.materialize,
                         cache_dir=config.materialize.cache_dir / 'training_windows' / window_ids[0]))
    return config


def training_plan(config, window_ids=None):
    windows = training_windows(config, TradingCalendar.load(config.data_root), window_ids)
    if not windows:
        raise ValueError("No training windows; check calendar and configured date ranges")
    def span(dates):
        return [dates[0], dates[-1]] if dates else None
    return {
        "artifact_dir": str(config.model.artifact_dir),
        "cache_dir": str(config.materialize.cache_dir),
        "diagnostics_dir": str(Path(config.model.artifact_dir).parent / 'diagnostics'),
        "windows": [dict(window_id=w.window_id, train=span(w.train_dates),
                         valid=span(w.valid_dates), predict=span(w.predict_dates),
                         train_count=len(w.train_dates), valid_count=len(w.valid_dates),
                         training_cutoff_date=w.training_cutoff_date,
                         activation_date=w.activation_date) for w in windows],
        "aggregate_diagnostics": bool(config.diagnostics.enabled and
                                     (config.diagnostics.write_loss_history or
                                      config.diagnostics.write_window_summary)),
    }
