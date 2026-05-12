from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.optuna_space import suggest_params


class XGBoostOptunaAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None
        self.best_params_: dict[str, Any] | None = None
        self.best_value_: float | None = None
        self.study_summary_: dict[str, Any] | None = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        xgb = _xgboost_module()
        optuna = _optuna_module()
        base_params = _base_params(context.model_params)
        search = _search_params(context.model_params)
        x_train_all = _features(train_data, context.feature_columns)
        y_train_all = _label(train_data, context.label_column)
        x_fit, y_fit, x_valid, y_valid = _validation_split(
            x_train_all,
            y_train_all,
            valid_data,
            context,
            float(search["valid_fraction"]),
        )

        sampler = optuna.samplers.TPESampler(seed=int(search["random_state"]))
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial):
            params = {**base_params, **suggest_params(trial, search.get("space"), _default_xgb_space())}
            model = xgb.XGBRegressor(**params)
            fit_kwargs = {}
            if x_valid is not None:
                fit_kwargs["eval_set"] = [(x_valid, y_valid)]
                fit_kwargs["verbose"] = False
            model.fit(x_fit, y_fit, **fit_kwargs)
            pred = model.predict(x_valid if x_valid is not None else x_fit)
            target = y_valid if x_valid is not None else y_fit
            return _mse(target, pred)

        study.optimize(
            objective,
            n_trials=int(search["n_trials"]),
            timeout=search.get("timeout"),
            show_progress_bar=bool(search["show_progress_bar"]),
        )
        self.best_params_ = {**base_params, **dict(study.best_params)}
        self.best_value_ = float(study.best_value)
        self.study_summary_ = {
            "best_value": self.best_value_,
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
        }
        self.model = xgb.XGBRegressor(**self.best_params_)
        self.model.fit(x_train_all, y_train_all)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        score = self.model.predict(_features(data, context.feature_columns))
        return pd.Series(score, index=data.index, dtype="float32")

    def tune(self, data_factory, context: ModelContext) -> dict[str, Any]:
        train_data, valid_data = data_factory()
        self.fit(train_data, valid_data, context)
        return {
            "best_params": self.best_params_,
            "best_value": self.best_value_,
            "study_summary": self.study_summary_,
        }


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )


def _label(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].astype("float32").to_numpy()


def _base_params(raw: dict[str, Any]) -> dict[str, Any]:
    params = {key: value for key, value in dict(raw).items() if key != "search"}
    params.setdefault("objective", "reg:squarederror")
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)
    return params


def _search_params(raw: dict[str, Any]) -> dict[str, Any]:
    search = dict(raw.get("search", {}))
    search.setdefault("n_trials", 50)
    search.setdefault("timeout", None)
    search.setdefault("valid_fraction", 0.2)
    search.setdefault("random_state", 42)
    search.setdefault("show_progress_bar", False)
    return search


def _default_xgb_space() -> dict[str, Any]:
    return {
        "n_estimators": {"type": "int", "low": 100, "high": 800, "step": 50},
        "max_depth": {"type": "int", "low": 2, "high": 8},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.2, "log": True},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "min_child_weight": {"type": "float", "low": 1e-2, "high": 100.0, "log": True},
        "gamma": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
    }


def _validation_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    valid_data: pd.DataFrame,
    context: ModelContext,
    valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not valid_data.empty and context.label_column in valid_data.columns:
        return (
            x_train,
            y_train,
            _features(valid_data, context.feature_columns),
            _label(valid_data, context.label_column),
        )
    if x_train.shape[0] < 5 or valid_fraction <= 0.0:
        return x_train, y_train, None, None
    rng = np.random.default_rng(42)
    order = rng.permutation(x_train.shape[0])
    valid_size = max(1, min(x_train.shape[0] - 1, int(round(x_train.shape[0] * valid_fraction))))
    valid_idx = order[:valid_size]
    fit_idx = order[valid_size:]
    return x_train[fit_idx], y_train[fit_idx], x_train[valid_idx], y_train[valid_idx]


def _mse(y_true, y_pred) -> float:
    diff = np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64")
    return float(np.mean(diff * diff))


def _xgboost_module():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("XGBoostOptunaAlphaModel requires installing xgboost") from exc
    return xgb


def _optuna_module():
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("XGBoostOptunaAlphaModel requires installing optuna") from exc
    return optuna
