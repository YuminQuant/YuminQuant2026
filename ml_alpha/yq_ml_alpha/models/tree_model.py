from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class RandomForestAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None
        self.params: dict = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional local package
            raise ImportError("RandomForestAlphaModel requires installing scikit-learn") from exc

        self.params = _params(context.model_params)
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(
            _features(train_data, context.feature_columns),
            train_data[context.label_column].astype(float).to_numpy(),
        )

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        score = self.model.predict(_features(data, context.feature_columns))
        return pd.Series(score, index=data.index, dtype="float32")


def _params(raw: dict) -> dict:
    params = dict(raw)
    params.setdefault("n_estimators", 300)
    params.setdefault("max_depth", None)
    params.setdefault("min_samples_leaf", 20)
    params.setdefault("min_samples_split", 40)
    params.setdefault("max_features", "sqrt")
    params.setdefault("bootstrap", True)
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)
    return params


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
