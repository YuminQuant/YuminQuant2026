from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class LinearRegressionAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.train_rows_: int = 0
        self.n_features_: int = 0

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype("float64").to_numpy()
        self.train_rows_ = int(x.shape[0])
        self.n_features_ = int(x.shape[1])
        x = np.column_stack([np.ones(len(x), dtype="float64"), x.astype("float64", copy=False)])
        self.coef_, *_ = np.linalg.lstsq(x, y, rcond=None)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("model is not fitted")
        x = _features(data, context.feature_columns).astype("float64", copy=False)
        x = np.column_stack([np.ones(len(x), dtype="float64"), x])
        score = x @ self.coef_
        return pd.Series(score, index=data.index, dtype="float32")

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        diagnostics = context.diagnostics
        if not diagnostics.get("enabled", False) or not diagnostics.get("write_model_info", False):
            return []
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        coef = self.coef_ if self.coef_ is not None else np.array([], dtype="float64")
        info = {
            "window_id": context.artifact_dir.name,
            "run_id": context.run_id,
            "alpha_id": context.alpha_id,
            "model_class": self.__class__.__name__,
            "train_rows": self.train_rows_,
            "valid_rows": 0,
            "n_features": self.n_features_,
            "search_enabled": False,
            "intercept": float(coef[0]) if coef.size else None,
            "coef_abs_mean": float(np.mean(np.abs(coef[1:]))) if coef.size > 1 else None,
            "coef_abs_max": float(np.max(np.abs(coef[1:]))) if coef.size > 1 else None,
        }
        path = context.artifact_dir / "model_info.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2)
        return [path]


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
