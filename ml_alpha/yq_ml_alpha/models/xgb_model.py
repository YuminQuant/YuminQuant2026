from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class XGBoostAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - depends on optional local package
            raise ImportError("XGBoostAlphaModel requires installing the optional xgboost package") from exc

        params = dict(context.model_params)
        fit_on_validation = bool(params.pop("fit_on_validation", False))
        if fit_on_validation:
            if valid_data.empty or context.label_column not in valid_data.columns:
                raise ValueError("fit_on_validation requires explicit labeled validation data")
            # Fixed parameters need only one final fit on the combined pool.
            # Do not evaluate on rows that are also used for fitting.
            train_data = pd.concat([train_data, valid_data], ignore_index=True)
        self.model = xgb.XGBRegressor(**params)
        fit_kwargs = {}
        if not fit_on_validation and not valid_data.empty:
            fit_kwargs["eval_set"] = [
                (_features(valid_data, context.feature_columns), valid_data[context.label_column].astype(float).to_numpy())
            ]
            fit_kwargs["verbose"] = False
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
