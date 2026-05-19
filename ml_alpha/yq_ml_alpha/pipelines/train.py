from __future__ import annotations

import importlib
import json
import math
import datetime as dt
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
    return _fit_predict_write_all(config)


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


def predict_only(config_path: str | Path) -> list[Path]:
    config = load_config(config_path)
    if config.dates.predict is None:
        return []
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    written: list[Path] = []
    writer = AlphaWriter(config.output_root, config.alpha_id)
    model_class = _model_class(config.model.class_path)
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha predict", len(windows))
    for idx, window in enumerate(windows, start=1):
        progress.window(idx, window)
        path = window_artifact_path(config.model.artifact_dir, window.window_id)
        progress.step("load_model")
        model = model_class.load(path)
        written.extend(
            _predict_write_window(config, dataset, writer, model, window, progress, calendar)
        )
    progress.done()
    return written


def _fit_predict_write_all(config: MlAlphaConfig) -> list[Path]:
    calendar = TradingCalendar.load(config.data_root)
    dataset = DatasetBuilder(config)
    written: list[Path] = []
    writer = AlphaWriter(config.output_root, config.alpha_id)
    windows = build_windows(config, calendar)
    progress = _Progress("ml-alpha run", len(windows))
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
        written.extend(
            _predict_write_window(config, dataset, writer, model, window, progress, calendar, context)
        )
    written.extend(_aggregate_diagnostics(config))
    progress.done()
    return written


def _predict_write_window(
    config: MlAlphaConfig,
    dataset: DatasetBuilder,
    writer: AlphaWriter,
    model: AlphaModel,
    window: TrainingWindow,
    progress: "_Progress",
    calendar: TradingCalendar,
    context: ModelContext | None = None,
) -> list[Path]:
    written: list[Path] = []
    batches = _chunks(window.predict_dates, config.materialize.predict_batch_size)
    for batch_idx, dates in enumerate(batches, start=1):
        progress.step(f"load_predict {batch_idx}/{len(batches)} dates={_date_span(dates)}")
        predict_bundle = _load_bundle(config, dataset, calendar, dates, include_label=False)
        if predict_bundle.frame.empty:
            continue
        batch_context = context or _context(config, window.window_id, predict_bundle.feature_columns)
        progress.step(f"predict {batch_idx}/{len(batches)}")
        score = model.predict(predict_bundle.frame, batch_context)
        progress.step(f"write {batch_idx}/{len(batches)}")
        written.extend(writer.write(_prediction_frame(predict_bundle.frame, score)))
    return written


def build_windows(config: MlAlphaConfig, calendar: TradingCalendar) -> list[TrainingWindow]:
    train_frequency = _train_frequency(config)
    predict_dates = _predict_dates(config, calendar)
    scheme = config.train_scheme.type.lower()
    if not predict_dates:
        return _train_only_windows(config, calendar, scheme, train_frequency)
    if scheme == "static":
        return [
            TrainingWindow(
                window_id="static",
                train_dates=sample_dates(calendar, config.dates.train, train_frequency),
                valid_dates=_valid_dates(config, calendar, train_frequency),
                predict_dates=predict_dates,
            )
        ]

    if config.train_scheme.validation_ratio is not None:
        return _ratio_split_windows(config, calendar, predict_dates, scheme, train_frequency)
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
        ranges = _training_ranges(config, calendar, start, scheme)
        if ranges is None:
            continue
        train_range, valid_range = ranges
        if len(calendar.between(train_range[0], train_range[1])) < config.train_scheme.min_train_days:
            continue
        valid_dates = sample_dates(calendar, valid_range, train_frequency) if valid_range is not None else []
        windows.append(
            TrainingWindow(
                window_id=f"{idx + 1:04d}_{segment[0]}_{segment[-1]}",
                train_dates=sample_dates(calendar, train_range, train_frequency),
                valid_dates=valid_dates,
                predict_dates=segment,
            )
        )
    return windows


def _load_bundle(
    config: MlAlphaConfig,
    dataset: DatasetBuilder,
    calendar: TradingCalendar,
    dates: list[int],
    include_label: bool,
):
    if config.features.type in {"bar_panel", "multi_bar_panel"}:
        return dataset.load_bar_panel(dates, include_label, calendar)
    if _uses_sequence_dataset(config):
        sequence_length = int(config.model.params.get("sequence_length", 6))
        sequence_frequency = str(config.model.params.get("sequence_frequency", config.sample.train_frequency))
        return dataset.load_sequence(dates, include_label, calendar, sequence_length, sequence_frequency)
    return dataset.load(dates, include_label)


def _uses_sequence_dataset(config: MlAlphaConfig) -> bool:
    return config.model.class_path in {
        "yq_ml_alpha.models.elstm_ranknet_model.eLSTMRankNetAlphaModel",
        "yq_ml_alpha.models.sequence_model.RNNAlphaModel",
        "yq_ml_alpha.models.sequence_model.LSTMAlphaModel",
        "yq_ml_alpha.models.sequence_model.GRUAlphaModel",
    }


def _train_only_windows(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    scheme: str,
    train_frequency: str,
) -> list[TrainingWindow]:
    train_pool = _train_only_sample_dates(config, calendar, train_frequency)
    if not train_pool:
        return []
    count = int(config.train_scheme.train_sample_count)
    if config.train_scheme.validation_ratio is not None:
        train_dates, valid_dates = _split_by_validation_ratio(train_pool, float(config.train_scheme.validation_ratio))
        if not train_dates:
            return []
    elif count > 0:
        valid_count = max(0, int(config.train_scheme.validation_sample_count))
        if len(train_pool) < count + valid_count:
            return []
        valid_dates = train_pool[-valid_count:] if valid_count > 0 else _valid_dates(config, calendar, train_frequency)
        train_candidates = train_pool[:-valid_count] if valid_count > 0 else train_pool
        if scheme == "rolling":
            train_dates = train_candidates[-count:]
        elif scheme == "expanding":
            train_dates = train_candidates
        else:
            raise ValueError(f"train_sample_count is only supported for rolling/expanding, got {scheme}")
    else:
        train_dates = train_pool
        valid_dates = _valid_dates(config, calendar, train_frequency)
    return [
        TrainingWindow(
            window_id=f"train_only_{train_dates[-1]}",
            train_dates=train_dates,
            valid_dates=valid_dates,
            predict_dates=[],
        )
    ]


def _train_only_sample_dates(config: MlAlphaConfig, calendar: TradingCalendar, train_frequency: str) -> list[int]:
    frequency = train_frequency.lower().strip()
    if frequency in {"monthly", "monthly_end"}:
        return _actual_refit_dates(calendar, config.dates.train, train_frequency)
    return sample_dates(calendar, config.dates.train, train_frequency)


def _ratio_split_windows(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    predict_dates: list[int],
    scheme: str,
    train_frequency: str,
) -> list[TrainingWindow]:
    if scheme not in {"rolling", "expanding"}:
        raise ValueError(f"validation_ratio is only supported for rolling/expanding, got {scheme}")
    assert config.dates.predict is not None
    refits = _actual_refit_dates(calendar, config.dates.predict, config.train_scheme.refit_frequency)
    ratio = float(config.train_scheme.validation_ratio)
    windows = []
    for idx, refit in enumerate(refits):
        next_refit = refits[idx + 1] if idx + 1 < len(refits) else None
        segment = [date for date in predict_dates if date > refit and (next_refit is None or date <= next_refit)]
        if not segment:
            continue
        train_range = _train_range_for_refit(config, calendar, refit, scheme)
        if train_range is None:
            continue
        eligible = sample_dates(calendar, train_range, train_frequency)
        train_dates, valid_dates = _split_by_validation_ratio(eligible, ratio)
        if not train_dates:
            continue
        windows.append(
            TrainingWindow(
                window_id=f"{len(windows) + 1:04d}_{segment[0]}_{segment[-1]}",
                train_dates=train_dates,
                valid_dates=valid_dates,
                predict_dates=segment,
            )
        )
    return windows


def _split_by_validation_ratio(dates: list[int], ratio: float) -> tuple[list[int], list[int]]:
    if len(dates) < 2:
        return [], []
    valid_count = min(max(1, math.floor(len(dates) * ratio + 0.5)), len(dates) - 1)
    if valid_count <= 0:
        return list(dates), []
    return dates[:-valid_count], dates[-valid_count:]


def _sample_count_windows(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    predict_dates: list[int],
    scheme: str,
    train_frequency: str,
) -> list[TrainingWindow]:
    assert config.dates.predict is not None
    refits = _actual_refit_dates(calendar, config.dates.predict, config.train_scheme.refit_frequency)
    train_pool = sample_dates(calendar, config.dates.train, train_frequency)
    count = int(config.train_scheme.train_sample_count)
    valid_count = max(0, int(config.train_scheme.validation_sample_count))
    windows = []
    for idx, refit in enumerate(refits):
        next_refit = refits[idx + 1] if idx + 1 < len(refits) else None
        segment = [date for date in predict_dates if date > refit and (next_refit is None or date <= next_refit)]
        if not segment:
            continue
        # The refit date anchors the prediction segment that starts after it.
        # For forward-return labels, the refit-date sample itself would use
        # returns inside the prediction segment, so train/valid samples must be
        # strictly earlier than the refit anchor.
        eligible = [date for date in train_pool if date < refit]
        if len(eligible) < count + valid_count:
            continue
        valid_dates = eligible[-valid_count:] if valid_count > 0 else []
        candidates = eligible[:-valid_count] if valid_count > 0 else eligible
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
                valid_dates=valid_dates,
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
        if frequency in {"semiannual", "semiannual_end", "halfyear", "halfyear_end"}:
            return _semiannual_key(next_open) != _semiannual_key(date)
        if frequency in {"weekly", "weekly_end"}:
            return _iso_week_key(next_open) != _iso_week_key(date)
    if frequency in {"monthly", "monthly_end"}:
        import calendar as cal

        text = str(date)
        last_day = cal.monthrange(int(text[:4]), int(text[4:6]))[1]
        return int(text[6:]) >= last_day - 3
    if frequency in {"semiannual", "semiannual_end", "halfyear", "halfyear_end"}:
        import calendar as cal

        text = str(date)
        month = int(text[4:6])
        if month not in {6, 12}:
            return False
        last_day = cal.monthrange(int(text[:4]), month)[1]
        return int(text[6:]) >= last_day - 3
    return True


def _train_frequency(config: MlAlphaConfig) -> str:
    if not config.sample.train_frequency:
        raise ValueError("sample.train_frequency is required")
    return config.sample.train_frequency


def _predict_frequency(config: MlAlphaConfig) -> str:
    if not config.sample.predict_frequency:
        raise ValueError("sample.predict_frequency is required when dates.predict is not empty")
    return config.sample.predict_frequency


def _predict_dates(config: MlAlphaConfig, calendar: TradingCalendar) -> list[int]:
    if config.dates.predict is None:
        return []
    return sample_dates(calendar, config.dates.predict, _predict_frequency(config))


def _valid_dates(config: MlAlphaConfig, calendar: TradingCalendar, train_frequency: str) -> list[int]:
    if config.dates.valid is None:
        return []
    return sample_dates(calendar, config.dates.valid, train_frequency)


def _iso_week_key(yyyymmdd: int) -> tuple[int, int]:
    import datetime as dt

    text = str(yyyymmdd)
    date = dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
    year, week, _ = date.isocalendar()
    return year, week


def _semiannual_key(yyyymmdd: int) -> tuple[int, int]:
    year = yyyymmdd // 10000
    month = (yyyymmdd // 100) % 100
    half = 1 if month <= 6 else 2
    return year, half


def _training_ranges(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    predict_start: int,
    scheme: str,
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    train_end_anchor = calendar.previous_open(predict_start) or config.dates.train[1]
    if config.dates.valid is None:
        train_end = train_end_anchor
        valid_range = None
    else:
        valid_days = max(1, int(config.train_scheme.valid_days))
        valid_start = calendar.offset(train_end_anchor, -(valid_days - 1)) or config.dates.valid[0]
        valid_range = (valid_start, train_end_anchor)
        train_end = calendar.previous_open(valid_start) or config.dates.train[1]
    train_range = _train_range_for_end(config, calendar, train_end, scheme, lookback_anchor=predict_start)
    if train_range is None:
        return None
    return train_range, valid_range


def _train_range_for_refit(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    refit_date: int,
    scheme: str,
) -> tuple[int, int] | None:
    train_end = calendar.previous_open(refit_date) or config.dates.train[1]
    return _train_range_for_end(config, calendar, train_end, scheme, lookback_anchor=refit_date)


def _train_range_for_end(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    train_end: int,
    scheme: str,
    lookback_anchor: int | None = None,
) -> tuple[int, int] | None:
    train_end = min(train_end, config.dates.train[1])
    if train_end < config.dates.train[0]:
        return None
    lookback = config.train_scheme.train_lookback
    if lookback:
        start = _lookback_start(config, calendar, lookback_anchor or train_end, train_end, lookback)
        if start is None or start < config.dates.train[0]:
            return None
    elif scheme in {"rolling", "expanding"}:
        start = config.dates.train[0]
    else:
        raise ValueError(f"unsupported train scheme: {scheme}")
    if start > train_end:
        return None
    return start, train_end


def _lookback_start(
    config: MlAlphaConfig,
    calendar: TradingCalendar,
    anchor_date: int,
    train_end: int,
    lookback: str,
) -> int | None:
    value = lookback.strip().lower()
    if value.endswith("d") and value[:-1].isdigit():
        days = int(value[:-1])
        if days <= 0:
            raise ValueError("train_scheme.train_lookback days must be > 0")
        return calendar.offset(train_end, -(days - 1))
    if value.endswith("y") and value[:-1].isdigit():
        years = int(value[:-1])
        if years <= 0:
            raise ValueError("train_scheme.train_lookback years must be > 0")
        end_date = _int_to_date(anchor_date)
        try:
            shifted = end_date.replace(year=end_date.year - years)
        except ValueError:
            shifted = end_date.replace(year=end_date.year - years, day=28)
        start_date = shifted + dt.timedelta(days=1)
        return int(start_date.strftime("%Y%m%d"))
    raise ValueError("train_scheme.train_lookback must use 'Ny' or 'Nd', for example '3y' or '720d'")


def _int_to_date(value: int) -> dt.date:
    text = str(value)
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))


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
        model_search=config.model.search,
        diagnostics=config.diagnostics.__dict__,
    )


def _prediction_frame(source: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": source["trade_date"].astype("int32").to_numpy(),
            "ts_code": source["ts_code"].to_numpy(),
            "score": score.astype("float32").to_numpy(),
        }
    )


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[idx : idx + size] for idx in range(0, len(values), size)]


def _aggregate_diagnostics(config: MlAlphaConfig) -> list[Path]:
    if not config.diagnostics.enabled:
        return []
    artifacts = Path(config.model.artifact_dir)
    diagnostics_dir = artifacts.parent / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if config.diagnostics.write_loss_history:
        frames = []
        for path in sorted(artifacts.glob("*/loss_history.parquet")):
            frames.append(pd.read_parquet(path))
        if frames:
            out = diagnostics_dir / "loss_history.parquet"
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            written.append(out)

    if config.diagnostics.write_window_summary:
        rows = []
        for path in sorted(artifacts.glob("*/model_info.json")):
            with path.open("r", encoding="utf-8") as file:
                rows.append(json.load(file))
        if rows:
            out = diagnostics_dir / "window_summary.parquet"
            pd.DataFrame(rows).to_parquet(out, index=False)
            written.append(out)
    return written


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
