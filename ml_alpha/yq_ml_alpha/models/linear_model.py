from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class LinearRegressionAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype("float64").to_numpy()
        x = np.column_stack([np.ones(len(x), dtype="float64"), x.astype("float64", copy=False)])
        self.coef_, *_ = np.linalg.lstsq(x, y, rcond=None)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        x = _features(data, context.feature_columns).astype("float64", copy=False)
        x = np.column_stack([np.ones(len(x), dtype="float64"), x])
        score = x @ self.coef_
        return pd.Series(score, index=data.index, dtype="float32")


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
