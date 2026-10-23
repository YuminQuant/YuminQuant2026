"""Huatai classical estimators using the shared model train/predict workflow."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.data.targets import cross_section_tail_labels
from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.tree_model import _features


def _estimator(kind: str, raw: dict[str, Any]):
    from sklearn import discriminant_analysis, ensemble, linear_model, naive_bayes, svm, tree

    specs = {
        "logistic": (linear_model.LogisticRegression,
                     dict(C=0.001, solver="lbfgs", max_iter=1000, random_state=42)),
        "sgd_hinge": (linear_model.SGDClassifier,
                      dict(loss="hinge", alpha=0.1, penalty="l2", max_iter=10000, tol=0.001, random_state=42)),
        "sgd_modified_huber": (linear_model.SGDClassifier,
                               dict(loss="modified_huber", alpha=1.0, penalty="l2", max_iter=10000,
                                    tol=0.001, random_state=42)),
        "svr": (svm.SVR, dict(kernel="rbf", C=1.0, gamma=0.01, epsilon=0.001, cache_size=512)),
        "gaussian_nb": (naive_bayes.GaussianNB, dict(var_smoothing=1e-9)),
        "lda": (discriminant_analysis.LinearDiscriminantAnalysis, dict(solver="lsqr", shrinkage="auto")),
        "qda": (discriminant_analysis.QuadraticDiscriminantAnalysis, dict(reg_param=0.1)),
        "adaboost": (ensemble.AdaBoostRegressor,
                     dict(n_estimators=100, learning_rate=0.05, loss="linear", random_state=42)),
        "gbdt": (ensemble.GradientBoostingRegressor,
                 dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_samples_leaf=20,
                      subsample=0.8, random_state=42)),
    }
    estimator_class, defaults = specs[kind]
    params = {**defaults, **raw}
    if kind.startswith("sgd_") and params["loss"] != defaults["loss"]:
        raise ValueError("select the matching SGD model class instead of changing its loss")
    if kind == "adaboost":
        base = tree.DecisionTreeRegressor(
            max_depth=int(params.pop("base_max_depth", 3)),
            min_samples_leaf=int(params.pop("base_min_samples_leaf", 10)),
            random_state=params["random_state"],
        )
        # sklearn 0.23 (local Python 3.8) and modern server environments.
        key = "estimator" if "estimator" in inspect.signature(estimator_class).parameters else "base_estimator"
        if "estimator" in raw or "base_estimator" in raw:
            raise ValueError("configure AdaBoost trees with base_max_depth/base_min_samples_leaf")
        params[key] = base
    return estimator_class(**params)


class _ClassicalAlphaModel(AlphaModel):
    estimator_kind = ""
    classification = False

    def __init__(self) -> None:
        self.model = None
        self.pca = None
        self.feature_columns = []
        self.model_info = {}
        self.prediction_batch_size = 10000

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        if context.model_search:
            raise ValueError("classical baselines use fixed model.params; model.search is unsupported")
        if train_data.empty:
            raise ValueError("classical baseline requires labeled training data")
        if not valid_data.empty and train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("classical baselines require chronological, disjoint train/validation dates")
        self.feature_columns = list(context.feature_columns)
        params = dict(context.model_params)
        self.tail_fraction = float(params.pop("tail_fraction", 0.3))
        if not 0 < self.tail_fraction < 0.5:
            raise ValueError("tail_fraction must be strictly between zero and 0.5")
        components = params.pop("pca_components", None)
        self.prediction_batch_size = int(params.pop("prediction_batch_size", 10000))
        if self.prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")
        train_target = self._target(train_data, context.label_column)
        keep = np.isfinite(train_target.to_numpy())
        if not keep.any():
            raise ValueError("no usable baseline targets")
        y = train_target.loc[keep].to_numpy()
        if self.classification and set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("tail classification requires both observed return classes")
        x = _features(train_data.loc[keep], self.feature_columns)
        if components is not None:
            from sklearn.decomposition import PCA
            # Fit solely on training samples, persist for validation/prediction.
            self.pca = PCA(n_components=components, svd_solver="full")
            x = self.pca.fit_transform(x)
        else:
            self.pca = None
        self.model = _estimator(self.estimator_kind, params)
        started = perf_counter()
        print(f"baseline fit kind={self.estimator_kind} rows={len(y)} features={x.shape[1]}", flush=True)
        self.model.fit(x, y.astype(np.int64) if self.classification else y)
        del x
        self.model_info = dict(
            model_class=self.__class__.__name__, estimator_kind=self.estimator_kind,
            task="tail_classification" if self.classification else "return_regression",
            feature_columns=self.feature_columns, train_rows=int(keep.sum()),
            train_pool_rows=len(train_data), valid_rows=len(valid_data),
            params=context.model_params, pca_components=components,
            output="decision_margin" if self.estimator_kind == "sgd_hinge" else (
                "positive_class_probability" if self.classification else "predicted_return"),
            fit_seconds=perf_counter() - started,
        )
        if not valid_data.empty:
            target = self._target(valid_data, context.label_column)
            score = self.predict(valid_data, context)
            finite = np.isfinite(target.to_numpy())
            self.model_info["valid_scored_rows"] = int(finite.sum())
            if finite.any():
                from sklearn.metrics import mean_squared_error, roc_auc_score
                if self.classification:
                    if len(np.unique(target.loc[finite])) == 2:
                        self.model_info["valid_auc"] = float(roc_auc_score(target.loc[finite], score.loc[finite]))
                else:
                    self.model_info["valid_mse"] = float(mean_squared_error(target.loc[finite], score.loc[finite]))
        print(f"baseline fit done kind={self.estimator_kind} seconds={self.model_info['fit_seconds']:.2f}", flush=True)

    def _target(self, frame: pd.DataFrame, column: str) -> pd.Series:
        if self.classification:
            return cross_section_tail_labels(frame, column, self.tail_fraction)
        return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("baseline prediction feature columns/order changed")
        scores = np.empty(len(data), dtype=np.float32)
        for start in range(0, len(data), self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, len(data))
            x = _features(data.iloc[start:stop], self.feature_columns)
            if self.pca is not None:
                x = self.pca.transform(x)
            if self.classification:
                if self.estimator_kind == "sgd_hinge":
                    values = self.model.decision_function(x)
                else:
                    positive = list(self.model.classes_).index(1)
                    values = self.model.predict_proba(x)[:, positive]
            else:
                values = self.model.predict(x)
            if not np.isfinite(values).all():
                raise ValueError(f"nonfinite {self.estimator_kind} predictions")
            scores[start:stop] = values
        return pd.Series(scores, index=data.index)

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        if not context.diagnostics.get("enabled") or not context.diagnostics.get("write_model_info"):
            return []
        artifact_io.mkdir(context.artifact_dir, parents=True, exist_ok=True)
        path = context.artifact_dir / "model_info.json"
        artifact_io.write_text(path, json.dumps(self.model_info, ensure_ascii=False, indent=2), encoding="utf-8")
        return [path]


class LogisticRegressionAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "logistic"
    classification = True


class SGDLinearSVMAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "sgd_hinge"
    classification = True


class SGDModifiedHuberAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "sgd_modified_huber"
    classification = True


class SVRAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "svr"


class GaussianNBAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "gaussian_nb"
    classification = True


class LDAAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "lda"
    classification = True


class QDAAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "qda"
    classification = True


class AdaBoostAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "adaboost"


class GradientBoostingAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "gbdt"
