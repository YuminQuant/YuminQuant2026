from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig, load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import refit_dates, sample_dates
from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.output.alpha_writer import AlphaWriter
from yq_ml_alpha.output.artifacts import window_artifact_path


@dataclass(frozen=True)
class TrainingWindow:
    window_id: str
    train_range: tuple[int, int]
    valid_range: tuple[int, int]
    predict_dates: list[int]


def run(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    predictions = _fit_predict_all(config)
    writer = AlphaWriter(config.output_root, config.alpha_id)
    return writer.write(predictions)


def train_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    paths = []
    for window in build_windows(config, calendar):
        model = _new_model(config)
        context = _context(config, window.window_id)
        train_bundle = dataset.load(sample_dates(calendar, window.train_range, config.sample.frequency), include_label=True)
        valid_bundle = dataset.load(sample_dates(calendar, window.valid_range, config.sample.frequency), include_label=True)
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        model.save(path)
        paths.append(path)
    return paths


def predict_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    frames = []
    model_class = _model_class(config.model.class_path)
    for window in build_windows(config, calendar):
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        model = model_class.load(path)
        context = _context(config, window.window_id)
        predict_bundle = dataset.load(window.predict_dates, include_label=False)
        if predict_bundle.frame.empty:
            continue
        score = model.predict(predict_bundle.frame, context)
        frames.append(_prediction_frame(predict_bundle.frame, score))
    predictions = pd.concat(frames, ignore_index=True) if frames else _empty_prediction_frame()
    return AlphaWriter(config.output_root, config.alpha_id).write(predictions)


def _fit_predict_all(config: MlAlphaConfig) -> pd.DataFrame:
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    frames = []
    for window in build_windows(config, calendar):
        model = _new_model(config)
        context = _context(config, window.window_id)
        train_dates = sample_dates(calendar, window.train_range, config.sample.frequency)
        valid_dates = sample_dates(calendar, window.valid_range, config.sample.frequency)
        train_bundle = dataset.load(train_dates, include_label=True)
        valid_bundle = dataset.load(valid_dates, include_label=True)
        if train_bundle.frame.empty:
            continue
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        model.save(window_artifact_path(config.model.artifact_dir, window.window_id))
        predict_bundle = dataset.load(window.predict_dates, include_label=False)
        if predict_bundle.frame.empty:
            continue
        frames.append(_prediction_frame(predict_bundle.frame, model.predict(predict_bundle.frame, context)))
    return pd.concat(frames, ignore_index=True) if frames else _empty_prediction_frame()


def build_windows(config: MlAlphaConfig, calendar: TradingCalendar) -> list[TrainingWindow]:
    predict_dates = sample_dates(calendar, config.dates.predict, config.sample.frequency)
    scheme = config.train_scheme.type.lower()
    if scheme == "static":
        return [
            TrainingWindow(
                window_id="static",
                train_range=config.dates.train,
                valid_range=config.dates.valid,
                predict_dates=predict_dates,
            )
        ]

    starts = refit_dates(calendar, predict_dates, config.train_scheme.refit_frequency)
    if predict_dates and (not starts or starts[0] != predict_dates[0]):
        starts = [predict_dates[0], *starts]
    windows = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else None
        segment = [date for date in predict_dates if date >= start and (end is None or date < end)]
        if not segment:
            continue
        train_range, valid_range = _training_ranges(config, calendar, start, scheme)
        if len(calendar.between(train_range[0], train_range[1])) < config.train_scheme.min_train_days:
            continue
        windows.append(
            TrainingWindow(
                window_id=f"{idx + 1:04d}_{segment[0]}_{segment[-1]}",
                train_range=train_range,
                valid_range=valid_range,
                predict_dates=segment,
            )
        )
    return windows


def _training_ranges(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    predict_start: int,
    scheme: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    train_end_anchor = calendar.previous_open(predict_start) or config.dates.train[1]
    valid_days = max(1, int(config.train_scheme.valid_days))
    valid_start = calendar.offset(train_end_anchor, -(valid_days - 1)) or config.dates.valid[0]
    valid_range = (valid_start, train_end_anchor)
    train_end = calendar.previous_open(valid_start) or config.dates.train[1]
    if scheme == "rolling":
        start = calendar.offset(train_end, -(max(1, config.train_scheme.rolling_train_days) - 1)) or config.dates.train[0]
    else:
        start = config.dates.train[0]
    return (start, train_end), valid_range


def _new_model(config: MlAlphaConfig) -> AlphaModel:
    return _model_class(config.model.class_path)()


def _model_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _context(config: MlAlphaConfig, window_id: str) -> ModelContext:
    return ModelContext(
        run_id=config.run_id,
        alpha_id=config.alpha_id,
        feature_columns=config.features.columns,
        label_column=config.label.id,
        artifact_dir=Path(config.model.artifact_dir) / window_id,
        model_params=config.model.params,
        tuning_params=config.tuning.params,
    )


def _prediction_frame(source: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": source["trade_date"].astype("int32").to_numpy(),
            "ts_code": source["ts_code"].to_numpy(),
            "score": score.astype("float32").to_numpy(),
        }
    )


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["trade_date", "ts_code", "score"])
