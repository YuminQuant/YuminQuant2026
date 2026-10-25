"""Huatai AI61 event-count ternary classifier, shared by own/related inputs."""
from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.xgb_training import make_matrix
from yq_ml_alpha.output import artifact_io


def _terciles(frame, column):
    """Keep tied returns together; do not use labels when loading predictions."""
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = values.groupby(frame.trade_date)
    count = grouped.transform("count")
    percentile = (grouped.rank(method="average") - 0.5) / count
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid = values.notna() & count.ge(3)
    result.loc[valid] = np.minimum(np.floor(3 * percentile.loc[valid]), 2)
    return result


def _positive_integer(params, name, default):
    value = params.pop(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _features(frame, columns):
    values = np.ascontiguousarray(frame[columns].to_numpy(dtype="float32"))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("event features must be complete, finite nonnegative occurrence counts")
    return values


def _progress_callback(xgb, enabled):
    if not enabled:
        return []

    def report(iteration):
        if (iteration + 1) % 50 == 0:
            print(f"event-xgb round={iteration + 1}", flush=True)

    if hasattr(xgb.callback, "TrainingCallback"):
        class Progress(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                report(epoch)
                return False
        return [Progress()]

    def callback(environment):
        report(environment.iteration)
    return [callback]


def _log_loss(booster, matrix):
    probabilities = np.asarray(booster.predict(matrix))
    labels = matrix.get_label().astype("int64")
    if probabilities.shape != (len(labels), 3) or not np.isfinite(probabilities).all():
        raise ValueError("invalid event validation probabilities")
    correct = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(correct, np.finfo(np.float32).tiny, 1.0)).mean())


class EventXGBoostAlphaModel(AlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def __init__(self):
        self.model = None
        self.feature_columns = []
        self.model_info = {}
        self.loss_history = []
        self.prediction_batch_size = 10000
        self.n_jobs = 4

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        import xgboost as xgb

        if context.model_search:
            raise ValueError("event XGBoost uses fixed parameters, without model.search")
        if train_data.empty or valid_data.empty:
            raise ValueError("event XGBoost requires explicit labeled train and validation pools")
        if train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("event XGBoost requires chronological, disjoint train/validation dates")
        if len(context.feature_columns) not in {28, 56} or len(set(context.feature_columns)) != len(context.feature_columns):
            raise ValueError("event XGBoost requires 28 own or 56 own/related event features")
        self.feature_columns = list(context.feature_columns)
        params = dict(context.model_params)
        rounds = _positive_integer(params, "n_estimators", 600)
        self.prediction_batch_size = _positive_integer(params, "prediction_batch_size", 10000)
        self.n_jobs = _positive_integer(params, "n_jobs", 4)
        seed = params.pop("seed", 42)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if params.pop("device", "cuda") != "cuda":
            raise ValueError("event XGBoost training requires device=cuda; CPU fallback is disabled")
        allowed = {"max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"}
        if set(params).difference(allowed):
            raise ValueError(f"unsupported event XGBoost parameters: {sorted(set(params).difference(allowed))}")
        common = dict(
            objective="multi:softprob", num_class=3, eval_metric="mlogloss", seed=seed,
            nthread=self.n_jobs, max_depth=3, eta=0.03, min_child_weight=1.0,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        )
        if "learning_rate" in params:
            params["eta"] = params.pop("learning_rate")
        common.update(params)
        version = tuple(int(part) for part in re.match(r"^(\d+)\.(\d+)", xgb.__version__).groups())
        if callable(getattr(xgb, "build_info", None)) and not xgb.build_info().get("USE_CUDA", True):
            raise RuntimeError("installed XGBoost lacks CUDA support; event model was not trained")
        common.update(dict(tree_method="hist", device="cuda") if version >= (2, 0)
                      else dict(tree_method="gpu_hist", gpu_id=0))
        started = perf_counter()
        train_target = _terciles(train_data, context.label_column)
        valid_target = _terciles(valid_data, context.label_column)
        train_keep, valid_keep = train_target.notna(), valid_target.notna()
        if set(train_target.loc[train_keep].unique()) != {0, 1, 2} or not valid_keep.any():
            raise ValueError("event training needs all three return classes and usable validation targets")
        train_matrix = make_matrix(
            xgb, _features(train_data.loc[train_keep], self.feature_columns),
            train_target.loc[train_keep].to_numpy(dtype="float32"), self.n_jobs,
        )
        valid_matrix = make_matrix(
            xgb, _features(valid_data.loc[valid_keep], self.feature_columns),
            valid_target.loc[valid_keep].to_numpy(dtype="float32"), self.n_jobs,
        )
        print(f"event-xgb fit rows={int(train_keep.sum())} valid_rows={int(valid_keep.sum())} "
              f"features={len(self.feature_columns)} rounds={rounds} device=cuda", flush=True)
        # Modern XGBoost can warn and silently choose CPU when CUDA is absent.
        # Convert those warnings to failures, rather than running a CPU fallback.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="(?is).*(no visible gpu|not compiled with (cuda|gpu)|falling back to cpu).*")
            self.model = xgb.train(
                common, train_matrix, num_boost_round=rounds,
                # Fixed rounds have no model-selection step. Evaluate once at
                # the end instead of rescoring both pools on every round.
                evals=[], verbose_eval=False,
                callbacks=_progress_callback(xgb, context.diagnostics.get("print_epoch")),
            )
        if version >= (2, 0):
            device = json.loads(self.model.save_config())["learner"]["generic_param"].get("device", "cpu")
            if not device.startswith("cuda"):
                self.model = None
                raise RuntimeError("XGBoost did not use CUDA; refusing a CPU-trained event model")
        self.loss_history = [dict(round=rounds, train_mlogloss=_log_loss(self.model, train_matrix),
                                  valid_mlogloss=_log_loss(self.model, valid_matrix))]
        self.model_info = dict(
            model_class=self.__class__.__name__, feature_columns=self.feature_columns,
            objective="multi:softprob", class_order=["bottom", "middle", "top"],
            target="daily_midpoint_return_rank_terciles_ties_together_minimum_3_labels",
            output="probability_top_minus_probability_bottom", output_range=[-1, 1],
            train_rows=int(train_keep.sum()), valid_rows=int(valid_keep.sum()),
            train_class_counts={str(key): int(value) for key, value in train_target.loc[train_keep].value_counts().items()},
            valid_class_counts={str(key): int(value) for key, value in valid_target.loc[valid_keep].value_counts().items()},
            boosting_rounds=rounds, trees_per_round=3, individual_trees=3 * rounds,
            validation_policy="final_round_only_no_early_stopping_or_refit_on_validation",
            params=common, xgboost_version=xgb.__version__, fit_seconds=perf_counter() - started,
            train_mlogloss=self.loss_history[-1]["train_mlogloss"],
            valid_mlogloss=self.loss_history[-1]["valid_mlogloss"],
        )
        print(f"event-xgb fit done rounds={rounds} valid_mlogloss={self.model_info['valid_mlogloss']:.6f} "
              f"seconds={self.model_info['fit_seconds']:.2f}", flush=True)

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        import xgboost as xgb

        if self.model is None:
            raise RuntimeError("event model is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("event prediction feature columns/order changed")
        scores = np.empty(len(data), dtype="float32")
        for start in range(0, len(data), self.prediction_batch_size):
            stop = min(start + self.prediction_batch_size, len(data))
            matrix = make_matrix(xgb, _features(data.iloc[start:stop], self.feature_columns), n_jobs=self.n_jobs)
            probabilities = np.asarray(self.model.predict(matrix))
            if (probabilities.shape != (stop - start, 3) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
                raise ValueError("invalid event class probabilities")
            scores[start:stop] = probabilities[:, 2] - probabilities[:, 0]
        return pd.Series(scores, index=data.index)

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        if not context.diagnostics.get("enabled"):
            return []
        artifact_io.mkdir(context.artifact_dir, parents=True, exist_ok=True)
        paths = []
        if context.diagnostics.get("write_model_info"):
            path = context.artifact_dir / "model_info.json"
            artifact_io.write_text(path, json.dumps(self.model_info, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(path)
        if context.diagnostics.get("write_loss_history"):
            path = context.artifact_dir / "loss_history.parquet"
            artifact_io.write_parquet(pd.DataFrame(self.loss_history), path, index=False)
            paths.append(path)
        return paths
