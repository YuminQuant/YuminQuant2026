from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class _RegularizedLinearAlphaModel(AlphaModel):
    estimator_name = ""

    def __init__(self) -> None:
        self.model = None
        self.best_params_: dict[str, Any] | None = None
        self.cv_results_: dict[str, Any] | None = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        sklearn_linear = _sklearn_linear_module()
        estimator_cls = getattr(sklearn_linear, self.estimator_name)
        params = _estimator_params(context.model_params)
        search_config = _search_config(context.model_params)
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype(float).to_numpy()
        validation = _explicit_validation(valid_data, context)
        if bool(search_config.get("enabled", False)):
            search_model = _fit_search(estimator_cls, params, search_config, x, y, self.estimator_name, validation)
            self.best_params_ = dict(getattr(search_model, "best_params_", {}))
            self.cv_results_ = _compact_cv_results(getattr(search_model, "cv_results_", {}))
            if hasattr(search_model, "best_estimator_"):
                self.model = search_model.best_estimator_
            else:
                self.model = estimator_cls(**{**params, **self.best_params_})
                self.model.fit(x, y)
        else:
            self.model = estimator_cls(**params)
            self.model.fit(x, y)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        score = self.model.predict(_features(data, context.feature_columns))
        return pd.Series(score, index=data.index, dtype="float32")

    def tune(self, data_factory, context: ModelContext) -> dict[str, Any]:
        train_data, _ = data_factory()
        self.fit(train_data, pd.DataFrame(), context)
        return {
            "best_params": self.best_params_,
            "cv_results": self.cv_results_,
        }


class LassoAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "Lasso"


class RidgeAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "Ridge"


class ElasticNetAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "ElasticNet"


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )


def _sklearn_linear_module():
    try:
        import sklearn.linear_model as linear_model
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("regularized linear alpha models require installing scikit-learn") from exc
    return linear_model


def _search_modules():
    try:
        from sklearn.model_selection import GridSearchCV, PredefinedSplit, RandomizedSearchCV
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("regularized linear alpha model search requires installing scikit-learn") from exc
    return GridSearchCV, PredefinedSplit, RandomizedSearchCV


def _estimator_params(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dict(raw).items() if key != "search"}


def _search_config(raw: dict[str, Any]) -> dict[str, Any]:
    search = dict(raw.get("search", {}))
    if "enabled" not in search:
        search["enabled"] = False
    search.setdefault("method", "random")
    search.setdefault("cv", 3)
    search.setdefault("n_iter", 40)
    search.setdefault("scoring", "neg_mean_squared_error")
    search.setdefault("n_jobs", -1)
    search.setdefault("random_state", 42)
    search.setdefault("verbose", 0)
    return search


def _fit_search(
    estimator_cls,
    params: dict[str, Any],
    search: dict[str, Any],
    x,
    y,
    estimator_name: str,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
):
    GridSearchCV, PredefinedSplit, RandomizedSearchCV = _search_modules()
    method = str(search.get("method", "random")).lower()
    param_space = _param_space(search, estimator_name)
    base = estimator_cls(**params)
    fit_x = x
    fit_y = y
    cv = int(search["cv"])
    refit = True
    if validation is not None:
        valid_x, valid_y = validation
        fit_x = np.vstack([x, valid_x])
        fit_y = np.concatenate([y, valid_y])
        cv = PredefinedSplit(
            np.concatenate(
                [
                    np.full(x.shape[0], -1, dtype="int32"),
                    np.zeros(valid_x.shape[0], dtype="int32"),
                ]
            )
        )
        refit = False
    common = {
        "estimator": base,
        "scoring": search["scoring"],
        "cv": cv,
        "n_jobs": int(search["n_jobs"]),
        "verbose": int(search["verbose"]),
        "error_score": search.get("error_score", np.nan),
        "refit": refit,
    }
    if method == "grid":
        model = GridSearchCV(param_grid=param_space, **common)
    elif method == "random":
        n_iter = min(int(search["n_iter"]), _grid_size(param_space))
        model = RandomizedSearchCV(
            param_distributions=param_space,
            n_iter=max(1, n_iter),
            random_state=int(search["random_state"]),
            **common,
        )
    else:
        raise ValueError(f"unsupported regularized linear search.method: {method}")
    model.fit(fit_x, fit_y)
    return model


def _explicit_validation(valid_data: pd.DataFrame, context: ModelContext) -> tuple[np.ndarray, np.ndarray] | None:
    if valid_data.empty or context.label_column not in valid_data.columns:
        return None
    x_valid = _features(valid_data, context.feature_columns)
    y_valid = valid_data[context.label_column].astype(float).to_numpy()
    if x_valid.shape[0] == 0 or y_valid.shape[0] == 0:
        return None
    return x_valid, y_valid


def _param_space(search: dict[str, Any], estimator_name: str) -> dict[str, list[Any]]:
    configured = search.get("param_grid") or search.get("param_distributions") or search.get("params")
    if isinstance(configured, dict) and configured:
        return {key: _as_list(value) for key, value in configured.items()}
    if estimator_name == "Lasso":
        return {
            "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "fit_intercept": [True, False],
            "max_iter": [3000, 5000, 10000],
            "tol": [1e-5, 1e-4, 1e-3],
            "selection": ["cyclic", "random"],
            "positive": [False, True],
        }
    if estimator_name == "Ridge":
        return {
            "alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0],
            "fit_intercept": [True, False],
            "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
            "tol": [1e-5, 1e-4, 1e-3],
            "max_iter": [None, 3000, 10000],
        }
    if estimator_name == "ElasticNet":
        return {
            "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
            "fit_intercept": [True, False],
            "max_iter": [3000, 5000, 10000],
            "tol": [1e-5, 1e-4, 1e-3],
            "selection": ["cyclic", "random"],
            "positive": [False, True],
        }
    raise ValueError(f"unsupported regularized estimator: {estimator_name}")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _grid_size(param_space: dict[str, list[Any]]) -> int:
    size = 1
    for values in param_space.values():
        size *= max(1, len(values))
    return size


def _compact_cv_results(cv_results: dict[str, Any]) -> dict[str, Any]:
    if not cv_results:
        return {}
    output: dict[str, Any] = {}
    for key in ["mean_test_score", "std_test_score", "rank_test_score", "params"]:
        if key not in cv_results:
            continue
        value = cv_results[key]
        if hasattr(value, "tolist"):
            value = value.tolist()
        output[key] = value
    return output
