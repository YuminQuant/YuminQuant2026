from __future__ import annotations

from pathlib import Path

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.output.artifacts import window_artifact_path
from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter
from yq_ml_alpha.output.factor_metadata import write_factor_metadata
from yq_ml_alpha.pipelines import materialize
from yq_ml_alpha.pipelines.runtime import (
    _Progress,
    _aggregate_diagnostics,
    _context,
    _fit_model,
    _load_bundle,
    _model_class,
    _new_model,
    _predict_dates,
    _predict_write_window,
    _release_accelerator_memory,
    TrainingWindow,
    build_windows,
)


def run(config_path: str | Path, *, resume: bool = False) -> list[Path]:
    return run_config(_load_factor_config(config_path), resume=resume)


def train_only(config_path: str | Path, *, resume: bool = False) -> list[Path]:
    return train_config(_load_factor_config(config_path), resume=resume)


def predict_only(config_path: str | Path) -> list[Path]:
    return predict_config(_load_factor_config(config_path))


def materialize_only(config_path: str | Path) -> list[Path]:
    config = _load_factor_config(config_path)
    return materialize.run_config(config)


def run_config(config: MlAlphaConfig, *, resume: bool = False) -> list[Path]:
    _ensure_factor_config(config)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    writer = _new_writer(config)
    written: list[Path] = []
    metadata_path = write_factor_metadata(config)
    if metadata_path is not None:
        written.append(metadata_path)
    windows = build_windows(config, calendar)
    all_predict_dates = _predict_dates(config, calendar)
    progress = _Progress("ml-alpha factor run", len(windows))
    model_class = _model_class(config.model.class_path) if resume else None
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        resume_predict_dates: list[int] | None = None
        artifact_path = window_artifact_path(config.model.artifact_dir, window.window_id)
        if resume:
            resume_predict_dates = writer.dates_missing_output_column(window.predict_dates)
            if not resume_predict_dates:
                progress.step("resume_skip output_complete")
                continue
            if artifact_path.exists():
                progress.step(f"resume_load_model missing_predict={len(resume_predict_dates)}")
                model = model_class.load(artifact_path)
                resume_window = _window_with_predict_dates(window, resume_predict_dates)
                written.extend(
                    _predict_write_window(
                        config,
                        dataset,
                        writer,
                        model,
                        resume_window,
                        progress,
                        calendar,
                        schema_dates=all_predict_dates,
                    )
                )
                progress.step("cleanup")
                del model
                _release_accelerator_memory()
                continue
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = _load_bundle(config, dataset, calendar, window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = _load_bundle(config, dataset, calendar, window.valid_dates, include_label=True)
        if train_bundle.frame.empty:
            del model, train_bundle, valid_bundle
            _release_accelerator_memory()
            continue
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        _fit_model(model, train_bundle, valid_bundle, context)
        if config.diagnostics.enabled:
            progress.step("diagnostics")
            written.extend(model.write_diagnostics(context))
        progress.step("save")
        model.save(artifact_path)
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
        progress.step("cleanup")
        del model, context
        _release_accelerator_memory()
    progress.step("ensure_output_column")
    written.extend(writer.ensure_output_column(all_predict_dates))
    written.extend(_aggregate_diagnostics(config))
    progress.done()
    return written


def train_config(config: MlAlphaConfig, *, resume: bool = False) -> list[Path]:
    _ensure_factor_config(config)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    paths: list[Path] = []
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha factor train", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        if resume and path.exists():
            progress.step("resume_skip artifact_exists")
            continue
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = _load_bundle(config, dataset, calendar, window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = _load_bundle(config, dataset, calendar, window.valid_dates, include_label=True)
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        _fit_model(model, train_bundle, valid_bundle, context)
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
    _ensure_factor_config(config)
    if config.dates.predict is None:
        return []
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    writer = _new_writer(config)
    written: list[Path] = []
    metadata_path = write_factor_metadata(config)
    if metadata_path is not None:
        written.append(metadata_path)
    model_class = _model_class(config.model.class_path)
    windows = build_windows(config, calendar)
    all_predict_dates = _predict_dates(config, calendar)
    progress = _Progress("ml-alpha factor predict", len(windows))
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
        progress.step("cleanup")
        del model
        _release_accelerator_memory()
    progress.step("ensure_output_column")
    written.extend(writer.ensure_output_column(all_predict_dates))
    progress.done()
    return written


def _window_with_predict_dates(window: TrainingWindow, predict_dates: list[int]) -> TrainingWindow:
    return TrainingWindow(
        window_id=window.window_id,
        train_dates=window.train_dates,
        valid_dates=window.valid_dates,
        predict_dates=predict_dates,
    )


def _new_writer(config: MlAlphaConfig) -> DailyWideWriter:
    if config.output.asset != "stock" or config.output.frequency != "daily":
        raise ValueError("factor output currently supports asset='stock' and frequency='daily'")
    return DailyWideWriter(
        config.output.root,
        config.output.id,
        layout="standard",
        asset=config.output.asset,
        frequency=config.output.frequency,
        base_root=config.output.base_root,
        write_workers=config.output.write_workers,
    )


def _load_factor_config(config_path: str | Path) -> MlAlphaConfig:
    config = load_config(config_path)
    _ensure_factor_config(config)
    return config


def _ensure_factor_config(config: MlAlphaConfig) -> None:
    if config.output.kind != "factor" or config.factor_id is None:
        raise ValueError("factor pipeline requires factor_id and output.kind='factor'")
    if config.output.id != config.factor_id:
        raise ValueError("factor pipeline requires output.id to match factor_id")
    if config.run_id.startswith("mdl_") or config.alpha_id.startswith("mdl_"):
        raise ValueError("factor pipeline must not use mdl_* run_id or alpha_id")
    if config.factor_id.startswith("e2e_fct_"):
        raise ValueError("factor pipeline requires semantic factor_id, not e2e_fct_*")
