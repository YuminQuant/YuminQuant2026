from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

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
    train_dates: list[int]
    valid_dates: list[int]
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
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha train", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = dataset.load(window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = dataset.load(window.valid_dates, include_label=True)
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        progress.step("save")
        model.save(path)
        paths.append(path)
    progress.done()
    return paths


def predict_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    frames = []
    model_class = _model_class(config.model.class_path)
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha predict", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        progress.step("load_model")
        model = model_class.load(path)
        progress.step("load_predict")
        predict_bundle = dataset.load(window.predict_dates, include_label=False)
        if predict_bundle.frame.empty:
            continue
        context = _context(config, window.window_id, predict_bundle.feature_columns)
        progress.step("predict")
        score = model.predict(predict_bundle.frame, context)
        frames.append(_prediction_frame(predict_bundle.frame, score))
    progress.done()
    predictions = pd.concat(frames, ignore_index=True) if frames else _empty_prediction_frame()
    return AlphaWriter(config.output_root, config.alpha_id).write(predictions)


def _fit_predict_all(config: MlAlphaConfig) -> pd.DataFrame:
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    frames = []
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha run", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        model = _new_model(config)
        progress.step("load_train")
        train_bundle = dataset.load(window.train_dates, include_label=True)
        progress.step("load_valid")
        valid_bundle = dataset.load(window.valid_dates, include_label=True)
        if train_bundle.frame.empty:
            continue
        context = _context(config, window.window_id, train_bundle.feature_columns)
        progress.step("fit")
        model.fit(train_bundle.frame, valid_bundle.frame, context)
        progress.step("save")
        model.save(window_artifact_path(config.model.artifact_dir, window.window_id))
        progress.step("load_predict")
        predict_bundle = dataset.load(window.predict_dates, include_label=False)
        if predict_bundle.frame.empty:
            continue
        progress.step("predict")
        frames.append(_prediction_frame(predict_bundle.frame, model.predict(predict_bundle.frame, context)))
    progress.done()
    return pd.concat(frames, ignore_index=True) if frames else _empty_prediction_frame()


def build_windows(config: MlAlphaConfig, calendar: TradingCalendar) -> list[TrainingWindow]:
    train_frequency = _train_frequency(config)
    predict_dates = sample_dates(calendar, config.dates.predict, _predict_frequency(config))
    scheme = config.train_scheme.type.lower()
    if scheme == "static":
        return [
            TrainingWindow(
                window_id="static",
                train_dates=sample_dates(calendar, config.dates.train, train_frequency),
                valid_dates=sample_dates(calendar, config.dates.valid, train_frequency),
                predict_dates=predict_dates,
            )
        ]

    if config.train_scheme.train_sample_count > 0:
        return _sample_count_windows(config, calendar, predict_dates, scheme, train_frequency)

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
                train_dates=sample_dates(calendar, train_range, train_frequency),
                valid_dates=sample_dates(calendar, valid_range, train_frequency),
                predict_dates=segment,
            )
        )
    return windows


def _sample_count_windows(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    predict_dates: list[int],
    scheme: str,
    train_frequency: str,
) -> list[TrainingWindow]:
    refits = _actual_refit_dates(calendar, config.dates.predict, config.train_scheme.refit_frequency)
    train_pool = sample_dates(calendar, config.dates.train, train_frequency)
    count = int(config.train_scheme.train_sample_count)
    windows = []
    for idx, refit in enumerate(refits):
        next_refit = refits[idx + 1] if idx + 1 < len(refits) else None
        segment = [date for date in predict_dates if date > refit and (next_refit is None or date <= next_refit)]
        if not segment:
            continue
        candidates = [date for date in train_pool if date < refit]
        if len(candidates) < count:
            continue
        if scheme == "rolling":
            train_dates = candidates[-count:]
        elif scheme == "expanding":
            train_dates = candidates
        else:
            raise ValueError(f"train_sample_count is only supported for rolling/expanding, got {scheme}")
        windows.append(
            TrainingWindow(
                window_id=f"{len(windows) + 1:04d}_{segment[0]}_{segment[-1]}",
                train_dates=train_dates,
                valid_dates=[],
                predict_dates=segment,
            )
        )
    return windows


def _actual_refit_dates(calendar: TradingCalendar, date_range: tuple[int, int], frequency: str) -> list[int]:
    candidates = refit_dates(calendar, calendar.between(date_range[0], date_range[1]), frequency)
    return [date for date in candidates if _is_actual_period_end(calendar, date, frequency)]


def _is_actual_period_end(calendar: TradingCalendar, date: int, frequency: str) -> bool:
    frequency = frequency.lower().strip()
    next_open = next((item for item in calendar.dates if item > date), None)
    if next_open is not None:
        if frequency in {"monthly", "monthly_end"}:
            return next_open // 100 != date // 100
        if frequency in {"weekly", "weekly_end"}:
            return _iso_week_key(next_open) != _iso_week_key(date)
    if frequency in {"monthly", "monthly_end"}:
        import calendar as cal

        text = str(date)
        last_day = cal.monthrange(int(text[:4]), int(text[4:6]))[1]
        return int(text[6:]) >= last_day - 3
    return True


def _train_frequency(config: MlAlphaConfig) -> str:
    return config.sample.train_frequency or config.sample.frequency


def _predict_frequency(config: MlAlphaConfig) -> str:
    return config.sample.predict_frequency or config.sample.frequency


def _iso_week_key(yyyymmdd: int) -> tuple[int, int]:
    import datetime as dt

    text = str(yyyymmdd)
    date = dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
    year, week, _ = date.isocalendar()
    return year, week


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


def _context(config: MlAlphaConfig, window_id: str, feature_columns: list[str] | None = None) -> ModelContext:
    if feature_columns is None:
        if isinstance(config.features.columns, str):
            raise ValueError("features.columns='__all__' must be resolved by DatasetBuilder before creating context")
        feature_columns = list(config.features.columns)
    return ModelContext(
        run_id=config.run_id,
        alpha_id=config.alpha_id,
        feature_columns=feature_columns,
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


class _Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = total
        self.current = 0

    def window(self, current: int, window: TrainingWindow) -> None:
        self.current = current
        percent = 100.0 * current / self.total if self.total else 100.0
        predict_span = _date_span(window.predict_dates)
        print(
            f"{self.label} [{current}/{self.total} {percent:5.1f}%] "
            f"window={window.window_id} train={len(window.train_dates)} "
            f"valid={len(window.valid_dates)} predict={len(window.predict_dates)} "
            f"predict_dates={predict_span}",
            flush=True,
        )

    def step(self, name: str) -> None:
        print(f"{self.label} [{self.current}/{self.total}] step={name}", flush=True)

    def done(self) -> None:
        print(f"{self.label} done windows={self.total}", flush=True)


def _date_span(dates: list[int]) -> str:
    if not dates:
        return "-"
    if len(dates) == 1:
        return str(dates[0])
    return f"{dates[0]}..{dates[-1]}"
