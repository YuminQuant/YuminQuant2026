"""Train KD's three component encoders and compute point-in-time features."""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.stores import KEY_COLUMNS
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.models.base import load_model_class


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
        from yq_ml_alpha.pipelines.runtime import build_windows

        self.config = config
        self.calendar = calendar
        self.columns = columns
        self.model_class = load_model_class(config.model.class_path)
        if (config.model.params.get("base_factors") != len(columns)
                or not callable(getattr(self.model_class, "predict_components_bundle", None))):
            raise ValueError(f"{config.factor_id} does not provide {len(columns)} components")
        self.windows = {}
        for window in build_windows(config, calendar):
            for date in window.predict_dates:
                if (date in self.windows or window.training_cutoff_date is None
                        or window.activation_date is None
                        or not window.training_cutoff_date < window.activation_date <= date):
                    raise ValueError(f"invalid KD-Ensemble upstream model window: {config.factor_id}/{date}")
                self.windows[date] = window
        self.dataset = None
        self.models = {}
        self.sessions = ExitStack()

    def identity(self):
        config = self.config
        return {"model": asdict(config.model), "features": asdict(config.features),
                "data_version": config.data_version, "dates": asdict(config.dates),
                "sample": asdict(config.sample), "train_scheme": asdict(config.train_scheme),
                "label": asdict(config.label), "preprocess": asdict(config.preprocess),
                "filters": asdict(config.filters), "universe": asdict(config.universe)}

    def window(self, date):
        try:
            return self.windows[date]
        except KeyError as error:
            raise ValueError(f"no effective KD-Ensemble upstream model: {self.config.factor_id}/{date}") from error

    def _dataset(self):
        from yq_ml_alpha.data.dataset import DatasetBuilder

        if self.dataset is None:
            self.dataset = DatasetBuilder(self.config)
        return self.dataset

    def train(self, window):
        from yq_ml_alpha.pipelines.runtime import (
            _Progress, _context, _fit_model, _load_train_valid, _new_model,
        )

        if window.window_id not in self.models:
            progress = _Progress(f"ml-alpha KD branch {self.config.factor_id}", 1)
            progress.window(1, window)
            train, valid = _load_train_valid(self.config, self._dataset(), self.calendar, window, progress)
            context = _context(self.config, window.window_id, train.feature_columns)
            model = _new_model(self.config)
            progress.step("fit")
            _fit_model(model, train, valid, context)
            model.suspend_prediction_device()
            self.models[window.window_id] = self.sessions.enter_context(model.prediction_session())
            progress.done()
        return self.models[window.window_id]

    def predict(self, window, dates, model):
        from yq_ml_alpha.pipelines.runtime import _context, _load_bundle

        bundle = _load_bundle(self.config, self._dataset(), self.calendar, dates, include_label=False)
        try:
            keys = pd.MultiIndex.from_frame(bundle.frame[KEY_COLUMNS])
            if bundle.frame.empty:
                values = np.empty((0, len(self.columns)), dtype=np.float32)
            else:
                context = _context(self.config, window.window_id, bundle.feature_columns)
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
            self.models.clear()
            if self.dataset is not None:
                self.dataset.close(clear_read_cache=False)
            self.dataset = None


class KDComponentsProvider(FeatureProvider):
    def __init__(self, config):
        sources = config.features.params["sources"]
        if sources != ["astgnn_week", "astgnn_msnew", "astgnn_lfq_daily"]:
            raise ValueError("KD-Ensemble requires week, msnew and LFQ component sources in order")
        calendar = TradingCalendar.for_config(config)
        self.sources = []
        for source in sources:
            columns = [f"{source}__component_{index:02d}" for index in range(1, 65)]
            self.sources.append(_ComponentSource(_source_config(config, source), calendar, columns))
        self.feature_columns = [name for source in self.sources for name in source.columns]
        self.batch_size = config.materialize.predict_batch_size
        self._training_frames = {}
        self._training_window = False
        self._prediction_models = None

    def dataset_identity(self):
        return {"kind": "in_memory_trained_components_v2",
                "sources": [source.identity() for source in self.sources],
                "join": "complete_three_branches"}

    @contextmanager
    def training_window(self, dates):
        # Later rolling windows can reuse the same point-in-time components.
        # Retire dates only when they leave the rolling lookback.
        if dates:
            first = min(dates)
            for date in list(self._training_frames):
                if date < first:
                    del self._training_frames[date]
        self._training_window = True
        try:
            yield
        finally:
            self._training_window = False

    def checkpoint(self, dates):
        """Package only branch windows covering the student prediction dates."""
        components = {}
        for source in self.sources:
            windows = {source.window(date).window_id: source.window(date) for date in dates}
            components[source.config.factor_id] = {
                key: source.train(window).checkpoint(include_diagnostics=False)
                for key, window in windows.items()
            }
        return components

    def restore_components(self, checkpoint, sessions):
        if set(checkpoint) != {source.config.factor_id for source in self.sources}:
            raise ValueError("KD-Ensemble checkpoint does not have all three ASTGNN branches")
        return {
            source.config.factor_id: {
                key: sessions.enter_context(source.model_class.from_checkpoint(weights).prediction_session())
                for key, weights in checkpoint[source.config.factor_id].items()
            }
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

    def _component_model(self, source, window):
        if self._training_window:
            return source.train(window)
        try:
            return self._prediction_models[source.config.factor_id][window.window_id]
        except (KeyError, TypeError) as error:
            raise ValueError(f"KD-Ensemble checkpoint missing {source.config.factor_id}/{window.window_id}") from error

    def _infer_dates(self, dates):
        windows = [tuple(source.window(date).window_id for source in self.sources) for date in dates]
        pairs = zip(windows, dates)
        empty = pd.DataFrame(columns=[*KEY_COLUMNS, *self.feature_columns])
        for _, group in groupby(pairs, key=lambda item: item[0]):
            group_dates = [date for _, date in group]
            for start in range(0, len(group_dates), self.batch_size):
                batch = group_dates[start:start + self.batch_size]
                branches = []
                for source in self.sources:
                    window = source.window(batch[0])
                    branches.append(source.predict(window, batch, self._component_model(source, window)))
                joined = pd.concat(branches, axis=1, join="inner").reset_index()
                daily = {date: frame for date, frame in joined.groupby("trade_date", sort=False)}
                for date in batch:
                    yield daily.get(date, empty)

    def load_many(self, dates):
        if not self._training_window:
            yield from self._infer_dates(dates)
            return
        missing = (date for date in dates if date not in self._training_frames)
        # Evaluate only cache misses in the original date order.
        frames = self._infer_dates(list(missing))
        try:
            for date in dates:
                if date not in self._training_frames:
                    self._training_frames[date] = next(frames)
                yield self._training_frames[date]
        finally:
            frames.close()

    def load(self, trade_date):
        return next(self.load_many([trade_date]))

    def close(self):
        self._training_frames.clear()
        for source in self.sources:
            source.close()
