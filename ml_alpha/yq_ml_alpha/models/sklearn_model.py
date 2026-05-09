from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class SklearnRegressorAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        params = dict(context.model_params)
        self.model = HistGradientBoostingRegressor(**params)
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype(float).to_numpy()
        self.model.fit(x, y)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        score = self.model.predict(_features(data, context.feature_columns))
        return pd.Series(score, index=data.index, dtype="float32")


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
