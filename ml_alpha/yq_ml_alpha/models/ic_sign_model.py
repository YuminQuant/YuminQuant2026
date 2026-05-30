from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SIGN_CACHE: dict[tuple[str, str, tuple[str, ...]], dict[str, float]] = {}


class ICSignEqualWeightAlphaModel(AlphaModel):
    """Equal-weight feature combiner after orienting each feature by RankIC sign."""

    def __init__(self) -> None:
        self.signs: dict[str, float] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        params = _params(context.model_params)
        backtest_root = _project_path(params["backtest_root"])
        metric = str(params["ic_metric"])
        cache_key = (str(backtest_root.resolve()), metric, tuple(context.feature_columns))
        signs = _SIGN_CACHE.get(cache_key)
        if signs is None:
            signs = {}
            for feature in context.feature_columns:
                sign = _feature_ic_sign(backtest_root / feature / "ic.parquet", metric)
                if sign is not None:
                    signs[feature] = sign
            _SIGN_CACHE[cache_key] = dict(signs)
        if not signs:
            raise ValueError(f"no valid IC signs found under {backtest_root} using metric={metric}")
        self.signs = dict(signs)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if not self.signs:
            raise RuntimeError("model is not fitted")
        features = [feature for feature in context.feature_columns if feature in self.signs and feature in data.columns]
        if not features:
            raise ValueError("none of the fitted IC-sign features are present in prediction data")
        values = data[features].replace([np.inf, -np.inf], np.nan)
        signs = pd.Series({feature: self.signs[feature] for feature in features}, dtype="float64")
        score = values.mul(signs, axis="columns").mean(axis=1)
        return score.fillna(0.0).astype("float32")


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    if "ic_root" in params:
        raise ValueError("ic_root is removed; use backtest_root pointing to data/backtest/stock/daily")
    params["backtest_root"] = params.get("backtest_root", "data/backtest/stock/daily")
    params["ic_metric"] = str(params.get("ic_metric", "rank_ic"))
    return params


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _feature_ic_sign(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    try:
        table = pd.read_parquet(path, columns=[metric])
    except Exception:
        return None
    values = pd.to_numeric(table[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    mean_value = float(values.mean())
    if not np.isfinite(mean_value) or mean_value == 0.0:
        return None
    return 1.0 if mean_value > 0.0 else -1.0
