from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class PCAOLSAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.pca_ = None
        self.coef_: np.ndarray | None = None
        self.n_original_features_: int = 0
        self.n_components_: int = 0
        self.explained_variance_ratio_sum_: float | None = None
        self.train_rows_: int = 0

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        pca_cls = _pca_class()
        explained_variance = float(context.model_params.get("explained_variance", 0.95))
        if not 0.0 < explained_variance <= 1.0:
            raise ValueError("explained_variance must be in (0, 1]")
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype("float64").to_numpy()
        self.train_rows_ = int(x.shape[0])
        self.n_original_features_ = int(x.shape[1])
        n_components = explained_variance if explained_variance < 1.0 else None
        self.pca_ = pca_cls(n_components=n_components, svd_solver="full")
        x_pca = self.pca_.fit_transform(x.astype("float64", copy=False))
        self.n_components_ = int(x_pca.shape[1])
        self.explained_variance_ratio_sum_ = float(np.sum(self.pca_.explained_variance_ratio_))
        x_design = np.column_stack([np.ones(len(x_pca), dtype="float64"), x_pca])
        self.coef_, *_ = np.linalg.lstsq(x_design, y, rcond=None)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.pca_ is None or self.coef_ is None:
            raise RuntimeError("model is not fitted")
        x = _features(data, context.feature_columns).astype("float64", copy=False)
        x_pca = self.pca_.transform(x)
        x_design = np.column_stack([np.ones(len(x_pca), dtype="float64"), x_pca])
        score = x_design @ self.coef_
        return pd.Series(score, index=data.index, dtype="float32")

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        diagnostics = context.diagnostics
        if not diagnostics.get("enabled", False) or not diagnostics.get("write_model_info", False):
            return []
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        ratio = []
        if self.pca_ is not None:
            ratio = [float(value) for value in self.pca_.explained_variance_ratio_]
        coef = self.coef_ if self.coef_ is not None else np.array([], dtype="float64")
        info = {
            "window_id": context.artifact_dir.name,
            "run_id": context.run_id,
            "alpha_id": context.alpha_id,
            "model_class": self.__class__.__name__,
            "train_rows": self.train_rows_,
            "valid_rows": 0,
            "n_original_features": self.n_original_features_,
            "n_components": self.n_components_,
            "explained_variance_ratio_sum": self.explained_variance_ratio_sum_,
            "explained_variance_ratio_json": json.dumps(ratio, ensure_ascii=False),
            "intercept": float(coef[0]) if coef.size else None,
            "coef_abs_mean": float(np.mean(np.abs(coef[1:]))) if coef.size > 1 else None,
            "coef_abs_max": float(np.max(np.abs(coef[1:]))) if coef.size > 1 else None,
        }
        path = context.artifact_dir / "model_info.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2, default=_json_default)
        return [path]


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )


def _pca_class():
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("PCAOLSAlphaModel requires installing scikit-learn") from exc
    return PCA


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
