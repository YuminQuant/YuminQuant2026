"""Build KD features from frozen, point-in-time component models in memory."""
from dataclasses import asdict, replace
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import PROJECT_ROOT, load_config
from yq_ml_alpha.data.stores import KEY_COLUMNS
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.models.base import load_model_class
from yq_ml_alpha.output.artifacts import validate_manifest


def _source_config(kd_config, source):
    config = load_config(PROJECT_ROOT / "ml_alpha" / "factors" / f"{source}.toml")
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
        model=replace(config.model, artifact_dir=kd_config.features.root /
                      config.model.artifact_dir.relative_to(config.data_root / "model_workspace")),
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
        self.model = None
        self.active_window = None

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

    def _model(self, window):
        from yq_ml_alpha.data.dataset import DatasetBuilder

        if self.dataset is None:
            self.dataset = DatasetBuilder(self.config)
        if self.active_window != window.window_id:
            path = self.config.model.artifact_dir / window.window_id / "model.pkl"
            validate_manifest(path, self.config, window, self.dataset)
            self.model = self.model_class.load(path)
            self.active_window = window.window_id
        return self.model

    def predict(self, window, dates):
        from yq_ml_alpha.pipelines.runtime import _context, _load_bundle

        model = self._model(window)
        bundle = _load_bundle(self.config, self.dataset, self.calendar, dates, include_label=False)
        try:
            keys = pd.MultiIndex.from_frame(bundle.frame[KEY_COLUMNS])
            if bundle.frame.empty:
                values = np.empty((0, len(self.columns)), dtype=np.float32)
            else:
                context = _context(self.config, window.window_id, bundle.feature_columns)
                with model.prediction_session():
                    scores, values = model.predict_components_bundle(bundle, context)
                if not scores.index.equals(bundle.frame.index):
                    raise ValueError(f"unaligned component rows: {self.config.factor_id}")
            if (keys.has_duplicates or values.shape != (len(keys), len(self.columns))
                    or not np.isfinite(values).all()):
                raise ValueError(f"invalid component output: {self.config.factor_id}/{dates[0]}")
            return pd.DataFrame(values, index=keys, columns=self.columns, copy=False)
        finally:
            bundle.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close(clear_read_cache=False)
        self.dataset = self.model = self.active_window = None


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

    def dataset_identity(self):
        return {"kind": "in_memory_pit_components_v1",
                "sources": [source.identity() for source in self.sources],
                "join": "complete_three_branches"}

    def load_many(self, dates):
        windows = [tuple(source.window(date).window_id for source in self.sources) for date in dates]
        pairs = zip(windows, dates)
        empty = pd.DataFrame(columns=[*KEY_COLUMNS, *self.feature_columns])
        for _, group in groupby(pairs, key=lambda item: item[0]):
            group_dates = [date for _, date in group]
            for start in range(0, len(group_dates), self.batch_size):
                batch = group_dates[start:start + self.batch_size]
                branches = [source.predict(source.window(batch[0]), batch) for source in self.sources]
                joined = pd.concat(branches, axis=1, join="inner").reset_index()
                daily = {date: frame for date, frame in joined.groupby("trade_date", sort=False)}
                for date in batch:
                    yield daily.get(date, empty)

    def load(self, trade_date):
        return next(self.load_many([trade_date]))

    def close(self):
        for source in self.sources:
            source.close()
