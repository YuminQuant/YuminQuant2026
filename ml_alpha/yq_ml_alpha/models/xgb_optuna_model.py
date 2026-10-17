from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.optuna_space import suggest_params
from yq_ml_alpha.models.xgb_training import make_matrix, fit_matrix


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
        search = _search_params(context.model_search)
        stopping = _SearchEarlyStopping(
            int(search["early_stopping_patience"]), float(search["min_relative_improvement"])
        )
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
        d_fit = make_matrix(xgb, x_fit, y_fit, base_params["n_jobs"])
        d_valid = (
            make_matrix(xgb, x_valid, y_valid, base_params["n_jobs"])
            if x_valid is not None else d_fit
        )

        def objective(trial):
            params = {**base_params, **suggest_params(trial, search.get("space"), _default_xgb_space())}
            model = fit_matrix(xgb, params, d_fit)
            pred = model.get_booster().predict(d_valid)
            target = y_valid if x_valid is not None else y_fit
            return _mse(target, pred)

        study.optimize(
            objective,
            n_trials=int(search["n_trials"]),
            timeout=search.get("timeout"),
            show_progress_bar=bool(search["show_progress_bar"]),
            callbacks=[stopping],
        )
        self.best_params_ = {**base_params, **dict(study.best_params)}
        self.best_value_ = float(study.best_value)
        self.study_summary_ = {
            "best_value": self.best_value_,
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
            "stopped_early": stopping.stopped,
            "early_stopping_patience": stopping.patience,
            "min_relative_improvement": stopping.min_relative_improvement,
        }
        if search.get("refit_on_validation", False):
            if valid_data.empty or context.label_column not in valid_data.columns:
                raise ValueError("refit_on_validation requires explicit labeled validation data")
            x_train_all = np.concatenate([x_train_all, x_valid])
            y_train_all = np.concatenate([y_train_all, y_valid])
        if x_train_all is x_fit:
            d_final = d_fit
        else:
            # Release trial matrices before allocating the full refit matrix.
            del d_fit, d_valid
            d_final = make_matrix(xgb, x_train_all, y_train_all, base_params["n_jobs"])
        self.model = fit_matrix(xgb, self.best_params_, d_final)

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
    raw = dict(raw)
    search = dict(raw.get("search", raw))
    search.setdefault("n_trials", 20)
    search.setdefault("early_stopping_patience", 5)
    search.setdefault("min_relative_improvement", 0.0001)
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


class _SearchEarlyStopping:
    """Stop after consecutive completed trials without material relative MSE improvement."""

    def __init__(self, patience: int, min_relative_improvement: float) -> None:
        if patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if not np.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be finite and in [0, 1)")
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.reference = None
        self.stale_trials = 0
        self.stopped = False

    def __call__(self, study, trial) -> None:
        if trial.state.name != "COMPLETE" or trial.value is None or not np.isfinite(trial.value):
            return
        value = float(trial.value)
        if self.reference is None:
            self.reference = value
            return
        improvement = self.reference - value
        if improvement > 0 and improvement >= abs(self.reference) * self.min_relative_improvement:
            self.reference = value
            self.stale_trials = 0
        else:
            self.stale_trials += 1
        if self.stale_trials >= self.patience:
            self.stopped = True
            print(
                f"Optuna early stop: {self.stale_trials} consecutive completed trials "
                f"without relative MSE improvement >= {self.min_relative_improvement:.6g}; "
                f"best_value={study.best_value:.10g}",
                flush=True,
            )
            study.stop()
