"""Train KD's three component encoders and compute point-in-time features."""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.data.stores import KEY_COLUMNS
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.models.base import load_model_class
from yq_ml_alpha.output import artifact_io
from yq_ml_alpha.output.parquet_reader import transient_parquet_tables

_branch_run = ContextVar("kd_branch_run", default=None)


def branch_path(config, window_id, stage):
    if stage not in config.features.params["sources"]:
        raise ValueError(f"Unknown KD branch: {stage}")
    return config.model.artifact_dir / window_id / f"{stage}.pkl"


def train_stage(config, window_id, stage, run_id, *, resume=False):
    from yq_ml_alpha.pipelines import factor
    from yq_ml_alpha.pipelines.training_tasks import training_windows

    if stage == "distill":
        token = _branch_run.set(run_id)
        try:
            # A new staged run must package the models from this run.
            return factor.train_config(config, window_ids=[window_id])
        finally:
            _branch_run.reset(token)
    path = branch_path(config, window_id, stage)
    provider = KDComponentsProvider(config)
    try:
        student, = training_windows(config, provider.calendar, [window_id])
        source = next(source for source in provider.sources if source.config.factor_id == stage)
        window = component_window(config, source.config, provider.calendar, student)
        identity = provider.branch_identity(source, window, run_id)
        if resume and artifact_io.exists(path):
            prior = artifact_io.torch_load(path, map_location="cpu")
            if prior["identity"] == identity:
                return [path]
        source.train(window, provider.window_calendar(student))
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(identity=identity, checkpoint=source.model.checkpoint(include_diagnostics=False)), path)
        return [path]
    finally:
        provider.close()


def _split_boundaries(config, window):
    params = config.features.params
    months, valid = params["component_months"], params["component_validation_months"]
    if type(months) is not int or type(valid) is not int or not 0 < valid < months:
        raise ValueError("KD component fit and validation require positive month counts")
    if window.training_start_date is None:
        raise ValueError("KD-Ensemble requires calendar-month training windows")
    start = pd.Timestamp(str(window.training_start_date))
    return tuple(int((start + pd.DateOffset(months=offset)).strftime("%Y%m%d"))
                 for offset in (months - valid, months))


def adjust_student_window(config, calendar, window):
    _, student_start = _split_boundaries(config, window)
    dates = [date for date in window.train_dates if date >= student_start]
    if not dates or not window.valid_dates:
        raise ValueError(f"{window.window_id}: KD requires separate student training and validation dates")
    return replace(window, train_dates=dates)


def component_window(config, source_config, calendar, student):
    from yq_ml_alpha.pipelines.runtime import TrainingWindow, _settled_dates

    start = student.training_start_date
    valid_start, student_start = _split_boundaries(config, student)
    first_student_session = calendar.between(student_start, student.training_cutoff_date)[0]
    cutoff = calendar.previous_open(first_student_session)
    frequency = source_config.sample.train_frequency
    valid_candidates = sample_dates(calendar, (valid_start, cutoff), frequency)
    valid = _settled_dates(source_config, calendar, valid_candidates, first_student_session)
    if not valid:
        raise ValueError(f"{student.window_id}: no mature ASTGNN validation targets")
    train_end = calendar.previous_open(valid_start)
    train_candidates = sample_dates(calendar, (start, train_end), frequency)
    train = _settled_dates(source_config, calendar, train_candidates, valid[0])
    if not train:
        raise ValueError(f"{student.window_id}: no mature ASTGNN training targets")
    return TrainingWindow(student.window_id, train, valid, [], cutoff, first_student_session, start)


def _source_config(kd_config, source):
    # These are KD's own branch training settings, packaged with the code.
    source_dir = Path(__file__).resolve().parents[1] / "resources" / "kd_sources"
    config = load_config(source_dir / f"{source}.toml")
    if config.factor_id != source or config.calendar_exchange != kd_config.calendar_exchange:
        raise ValueError(f"incompatible KD-Ensemble source config: {source}")

    def relocate(value):
        if isinstance(value, Path):
            return kd_config.data_root / value.relative_to(config.data_root)
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    return replace(
        config,
        data_root=kd_config.data_root,
        label=replace(config.label, root=relocate(config.label.root)),
        features=replace(config.features, root=relocate(config.features.root),
                         params=relocate(config.features.params)),
        model=replace(config.model, artifact_dir=kd_config.model.artifact_dir / "branch_diagnostics" / source),
    )


class _ComponentSource:
    def __init__(self, config, calendar, columns):
        self.config = config
        self.calendar = calendar
        self.columns = columns
        self.model_class = load_model_class(config.model.class_path)
        if (config.model.params.get("base_factors") != len(columns)
                or not callable(getattr(self.model_class, "predict_components_bundle", None))):
            raise ValueError(f"{config.factor_id} does not provide {len(columns)} components")
        self.dataset = None
        self.model = None
        self.training_calendar = None
        self.sessions = ExitStack()

    def identity(self):
        config = self.config
        return {"model": asdict(config.model), "features": asdict(config.features),
                "data_version": config.data_version, "sample": asdict(config.sample),
                "label": asdict(config.label), "preprocess": asdict(config.preprocess),
                "filters": asdict(config.filters), "universe": asdict(config.universe)}

    def _dataset(self):
        from yq_ml_alpha.data.dataset import DatasetBuilder

        if self.dataset is None:
            self.dataset = DatasetBuilder(self.config, calendar=self.training_calendar or self.calendar)
        return self.dataset

    @transient_parquet_tables()
    def train(self, window, calendar):
        from yq_ml_alpha.pipelines.runtime import (
            _Progress, _context, _fit_model, _load_train_valid, _new_model,
        )

        self.close()
        self.training_calendar = calendar
        progress = _Progress(f"ml-alpha KD branch {self.config.factor_id}", 1)
        progress.window(1, window)
        train, valid = _load_train_valid(self.config, self._dataset(), calendar, window, progress)
        context = _context(self.config, window.window_id, train.feature_columns)
        model = _new_model(self.config)
        progress.step("fit")
        _fit_model(model, train, valid, context)
        model.suspend_prediction_device()
        self.model = self.sessions.enter_context(model.prediction_session())
        self.dataset.close(clear_read_cache=False)
        self.dataset = None
        progress.done()

    @transient_parquet_tables()
    def predict(self, dates, model):
        from yq_ml_alpha.pipelines.runtime import _context, _load_bundle

        calendar = self.training_calendar or self.calendar
        bundle = _load_bundle(self.config, self._dataset(), calendar, dates, include_label=False)
        try:
            keys = pd.MultiIndex.from_frame(bundle.frame[KEY_COLUMNS])
            if bundle.frame.empty:
                values = np.empty((0, len(self.columns)), dtype=np.float32)
            else:
                context = _context(self.config, "kd_components", bundle.feature_columns)
                scores, values = model.predict_components_bundle(bundle, context)
                if not scores.index.equals(bundle.frame.index):
                    raise ValueError(f"unaligned component rows: {self.config.factor_id}")
            if (keys.has_duplicates or values.shape != (len(keys), len(self.columns))
                    or not np.isfinite(values).all()):
                raise ValueError(f"invalid component output: {self.config.factor_id}/{dates[0]}")
            return pd.DataFrame(values, index=keys, columns=self.columns, copy=False)
        finally:
            try:
                model.suspend_prediction_device()
            finally:
                bundle.close()

    def close(self):
        try:
            self.sessions.close()
        finally:
            self.model = None
            self.training_calendar = None
            if self.dataset is not None:
                self.dataset.close(clear_read_cache=False)
            self.dataset = None


class KDComponentsProvider(FeatureProvider):
    def __init__(self, config):
        sources = config.features.params["sources"]
        if sources != ["astgnn_week", "astgnn_msnew", "astgnn_lfq_daily"]:
            raise ValueError("KD-Ensemble requires week, msnew and LFQ component sources in order")
        self.config = config
        self.calendar = TradingCalendar.for_config(config)
        self.sources = []
        for source in sources:
            columns = [f"{source}__component_{index:02d}" for index in range(1, 65)]
            self.sources.append(_ComponentSource(_source_config(config, source), self.calendar, columns))
        self.feature_columns = [name for source in self.sources for name in source.columns]
        self.batch_size = config.materialize.predict_batch_size
        self._training_window = False
        self._prediction_models = None

    def dataset_identity(self):
        return {"kind": "staged_components_v3",
                "sources": [source.identity() for source in self.sources],
                "join": "complete_three_branches"}

    def window_calendar(self, student):
        return TradingCalendar(self.calendar.between(student.training_start_date, student.training_cutoff_date))

    @staticmethod
    def branch_identity(source, window, run_id):
        return {"run_id": run_id, "branch": source.config.factor_id,
                "window": asdict(window),
                "config": artifact_io.identity_values(source.identity())}

    @contextmanager
    def training_window(self, student):
        calendar = self.window_calendar(student)
        run_id = _branch_run.get()
        self._training_window = True
        try:
            for source in self.sources:
                window = component_window(self.config, source.config, self.calendar, student)
                if run_id is None:
                    source.train(window, calendar)
                else:
                    source.close()
                    source.training_calendar = calendar
                    path = branch_path(self.config, student.window_id, source.config.factor_id)
                    saved = artifact_io.torch_load(path, map_location="cpu")
                    if saved["identity"] != self.branch_identity(source, window, run_id):
                        raise ValueError(f"KD branch checkpoint does not match this training run: {path}")
                    model = source.model_class.from_checkpoint(saved["checkpoint"])
                    source.model = source.sessions.enter_context(model.prediction_session())
            yield
        except BaseException:
            self.close()
            raise
        finally:
            self._training_window = False

    def checkpoint(self):
        """Capture the same fitted encoders that produced the student inputs."""
        try:
            return {
                source.config.factor_id: source.model.checkpoint(include_diagnostics=False)
                for source in self.sources
            }
        finally:
            self.close()

    def restore_components(self, checkpoint, sessions):
        if set(checkpoint) != {source.config.factor_id for source in self.sources}:
            raise ValueError("KD-Ensemble checkpoint does not have all three ASTGNN branches")
        return {
            source.config.factor_id: sessions.enter_context(
                source.model_class.from_checkpoint(checkpoint[source.config.factor_id]).prediction_session()
            )
            for source in self.sources
        }

    @contextmanager
    def prediction_components(self, models):
        previous = self._prediction_models
        self._prediction_models = models
        try:
            yield
        finally:
            self._prediction_models = previous

    def _component_model(self, source):
        if self._training_window:
            return source.model
        try:
            return self._prediction_models[source.config.factor_id]
        except (KeyError, TypeError) as error:
            raise ValueError(f"KD-Ensemble checkpoint missing {source.config.factor_id}") from error

    def load_many(self, dates):
        empty = pd.DataFrame(columns=[*KEY_COLUMNS, *self.feature_columns])
        for start in range(0, len(dates), self.batch_size):
            batch = dates[start:start + self.batch_size]
            branches = [source.predict(batch, self._component_model(source)) for source in self.sources]
            joined = pd.concat(branches, axis=1, join="inner").reset_index()
            daily = {date: frame for date, frame in joined.groupby("trade_date", sort=False)}
            for date in batch:
                yield daily.get(date, empty)

    def load(self, trade_date):
        return next(self.load_many([trade_date]))

    def close(self):
        for source in self.sources:
            source.close()
