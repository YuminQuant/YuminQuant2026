from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.output.alpha_writer import AlphaWriter
from yq_ml_alpha.output.artifacts import window_artifact_path
from yq_ml_alpha.pipelines import materialize
from yq_ml_alpha.pipelines.runtime import (
    _Progress,
    _aggregate_diagnostics,
    _context,
    _load_bundle,
    _model_class,
    _new_model,
    _predict_dates,
    _predict_write_window,
    _write_missing_coverage,
    build_windows,
)


def run(config_path: str | Path) -> list[Path]:
    return run_config(_load_model_config(config_path))


def train_only(config_path: str | Path) -> list[Path]:
    return train_config(_load_model_config(config_path))


def predict_only(config_path: str | Path) -> list[Path]:
    return predict_config(_load_model_config(config_path))


def materialize_only(config_path: str | Path) -> list[Path]:
    config = _load_model_config(config_path)
    return materialize.run_config(config)


def run_config(config: MlAlphaConfig) -> list[Path]:
    _ensure_model_config(config)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    writer = _new_writer(config)
    written: list[Path] = []
    windows = build_windows(config, calendar)
    all_predict_dates = _predict_dates(config, calendar)
    covered_dates: set[int] = set()
    progress = _Progress("ml-alpha model run", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = _load_bundle(config, dataset, calendar, window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = _load_bundle(config, dataset, calendar, window.valid_dates, include_label=True)
        if train_bundle.frame.empty:
            continue
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        if config.diagnostics.enabled:
            progress.step("diagnostics")
            written.extend(model.write_diagnostics(context))
        progress.step("save")
        model.save(window_artifact_path(config.model.artifact_dir, window.window_id))
        del train_bundle, valid_bundle
        written.extend(
            _predict_write_window(
                config,
                dataset,
                writer,
                model,
                window,
                progress,
                calendar,
                context,
                schema_dates=all_predict_dates,
            )
        )
        covered_dates.update(window.predict_dates)
    written.extend(_write_missing_coverage(writer, all_predict_dates, covered_dates))
    written.extend(_aggregate_diagnostics(config))
    progress.done()
    return written


def train_config(config: MlAlphaConfig) -> list[Path]:
    _ensure_model_config(config)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    paths: list[Path] = []
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha model train", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = _load_bundle(config, dataset, calendar, window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = _load_bundle(config, dataset, calendar, window.valid_dates, include_label=True)
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        if config.diagnostics.enabled:
            progress.step("diagnostics")
            paths.extend(model.write_diagnostics(context))
        progress.step("save")
        model.save(path)
        paths.append(path)
    paths.extend(_aggregate_diagnostics(config))
    progress.done()
    return paths


def predict_config(config: MlAlphaConfig) -> list[Path]:
    _ensure_model_config(config)
    if config.dates.predict is None:
        return []
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    writer = _new_writer(config)
    written: list[Path] = []
    model_class = _model_class(config.model.class_path)
    windows = build_windows(config, calendar)
    all_predict_dates = _predict_dates(config, calendar)
    covered_dates: set[int] = set()
    progress = _Progress("ml-alpha model predict", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        progress.step("load_model")
        model = model_class.load(path)
        written.extend(
            _predict_write_window(
                config,
                dataset,
                writer,
                model,
                window,
                progress,
                calendar,
                schema_dates=all_predict_dates,
            )
        )
        covered_dates.update(window.predict_dates)
    written.extend(_write_missing_coverage(writer, all_predict_dates, covered_dates))
    progress.done()
    return written


def _new_writer(config: MlAlphaConfig) -> AlphaWriter:
    if config.output.kind != "signal":
        raise ValueError(f"model pipeline requires output.kind='signal', got {config.output.kind!r}")
    return AlphaWriter(
        config.output.root,
        config.output.id,
        base_root=config.output.base_root,
        write_workers=config.output.write_workers,
    )


def _load_model_config(config_path: str | Path) -> MlAlphaConfig:
    config = load_config(config_path)
    _ensure_model_config(config)
    return config


def _ensure_model_config(config: MlAlphaConfig) -> None:
    if config.factor_id is not None or config.output.kind == "factor":
        raise ValueError("model pipeline requires a model config without factor_id or factor output")
    if config.output.kind != "signal":
        raise ValueError(f"model pipeline requires output.kind='signal', got {config.output.kind!r}")
