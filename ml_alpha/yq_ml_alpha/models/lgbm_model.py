from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class LightGBMAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        import lightgbm as lgb

        params = dict(context.model_params)
        self.model = lgb.LGBMRegressor(**params)
        fit_kwargs = {}
        if not valid_data.empty:
            fit_kwargs["eval_set"] = [
                (_features(valid_data, context.feature_columns), valid_data[context.label_column].astype(float).to_numpy())
            ]
        self.model.fit(
            _features(train_data, context.feature_columns),
            train_data[context.label_column].astype(float).to_numpy(),
            **fit_kwargs,
        )

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
