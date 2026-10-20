"""Date-grouped pairwise ranking; DMatrix construction is shared by all trials."""
import inspect
from functools import partial

import numpy as np
import pandas as pd

from yq_ml_alpha.models.xgb_training import make_matrix


def _metric(predictions, matrix, groups):
    labels = matrix.get_label()
    # XGBoost 0.90 cannot read group_ptr back from DMatrix.
    boundaries = np.concatenate([[0], np.cumsum(groups)])
    values = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        x = pd.Series(predictions[left:right]).rank().to_numpy()
        y = pd.Series(labels[left:right]).rank().to_numpy()
        if len(x) > 1:
            values.append(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0)
    return "date_rank_ic", float(np.mean(values)) if values else 0.0


def _matrix(xgb, data, threads):
    features, labels, groups = data
    if sum(groups) != len(features) or len(labels) != len(features) or min(groups) < 2:
        raise ValueError("ranker groups must cover complete labeled cross-sections")
    # Pairwise ranking depends only on order; integer ranks also support newer XGBoost validation.
    relevance = np.empty(len(labels), dtype="float32")
    left = 0
    for count in groups:
        relevance[left:left + count] = pd.Series(labels[left:left + count]).rank(method="dense").to_numpy() - 1
        left += count
    matrix = make_matrix(xgb, features, relevance, threads)
    matrix.set_group(groups)
    return matrix


def predict_ranker(booster, features, rounds, threads):
    import xgboost as xgb
    matrix = make_matrix(xgb, features, n_jobs=threads)
    if "iteration_range" in inspect.signature(booster.predict).parameters:
        return booster.predict(matrix, iteration_range=(0, int(rounds)))
    return booster.predict(matrix, ntree_limit=int(rounds))


def fit_ranker(train, valid, params, branch, encoder_device):
    import optuna
    import xgboost as xgb

    train_matrix = _matrix(xgb, train, params["ranker_n_jobs"])
    valid_matrix = _matrix(xgb, valid, params["ranker_n_jobs"])
    requested = params["ranker_device"]
    version = tuple(int(part) for part in xgb.__version__.split(".")[:2])
    use_gpu = requested == "cuda" or (requested == "auto" and encoder_device == "cuda")
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("ranker_device must be auto, cpu or cuda")
    common = dict(objective="rank:pairwise", seed=params["seed"], nthread=params["ranker_n_jobs"],
                  disable_default_eval_metric=1, tree_method="hist")
    if use_gpu:
        if version >= (2, 0):
            common["device"] = "cuda"
        else:
            common["tree_method"] = "gpu_hist"
    best = {}
    rows = []
    metric_arg = "custom_metric" if "custom_metric" in inspect.signature(xgb.train).parameters else "feval"

    def objective(trial):
        chosen = dict(max_depth=trial.suggest_int("max_depth", 3, 8),
                      eta=trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                      min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                      subsample=trial.suggest_float("subsample", 0.7, 1.0),
                      colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                      reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True))
        rounds = trial.suggest_int("n_estimators", 300, 800, step=100)
        booster = xgb.train(
            {**common, **chosen}, train_matrix, num_boost_round=rounds,
            evals=[(valid_matrix, "valid")], maximize=True,
            early_stopping_rounds=params["ranker_early_stopping"], verbose_eval=False,
            **{metric_arg: partial(_metric, groups=valid[2])},
        )
        score, used = float(booster.best_score), int(booster.best_iteration) + 1
        if not np.isfinite(score):
            raise ValueError("nonfinite ranker validation score")
        rows.append(dict(trial=trial.number, valid_rank_ic=score, best_rounds=used, params=trial.params))
        if not best or score > best["score"]:
            best.update(booster=booster, score=score, rounds=used, params=trial.params.copy())
        print(f"graph-temporal branch={branch} optuna={trial.number + 1}/{params['ranker_trials']} "
              f"valid_rank_ic={score:.6f} best_rounds={used}", flush=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=params["seed"]))
    study.optimize(objective, n_trials=params["ranker_trials"], n_jobs=1)
    details = dict(best_rounds=best["rounds"], best_valid_rank_ic=best["score"],
                   best_params=best["params"], trials=rows, device="cuda" if use_gpu else "cpu",
                   xgboost_version=xgb.__version__, objective="rank:pairwise")
    return best["booster"], details
