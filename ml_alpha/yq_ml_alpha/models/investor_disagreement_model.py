from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import apply_cross_section_transform
from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.tree_model import _features


class InvestorDisagreementAlphaModel(AlphaModel):
    """Random-feature LightGBM investors; standardized agreement/disagreement score."""

    def __init__(self) -> None:
        self.models = []
        self.feature_columns: list[str] = []
        self.feature_subsets: list[list[int]] = []
        self.params: dict[str, Any] = {}
        self.model_info: dict[str, Any] = {}
        self.investor_history: list[dict[str, Any]] = []

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        import lightgbm as lgb

        if context.model_search:
            raise ValueError("investor disagreement uses fixed parameters, not model.search")
        if train_data.empty:
            raise ValueError("investor disagreement requires labeled training rows")
        if not valid_data.empty and train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("investor disagreement requires disjoint chronological validation dates")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        if len(set(self.feature_columns)) != len(self.feature_columns) or len(self.feature_columns) < 2:
            raise ValueError("investor disagreement requires at least two distinct factors")
        n_investors = self.params["n_investors"]
        subset_size = int(math.ceil(len(self.feature_columns) * self.params["investor_feature_fraction"]))
        if subset_size >= len(self.feature_columns):
            raise ValueError("each investor must see a strict subset of the factors")
        y_train = train_data[context.label_column].to_numpy(dtype=np.float32)
        keep = np.isfinite(y_train)
        if not keep.any():
            raise ValueError("no finite investor return targets")
        frame = train_data if keep.all() else train_data.loc[keep]
        # Construct the full matrix once; only one investor's feature slice is live at a time.
        x_train = _features(frame, self.feature_columns)
        y_train = y_train[keep]
        rng = np.random.RandomState(self.params["seed"])
        self.feature_subsets = [
            sorted(rng.choice(len(self.feature_columns), size=subset_size, replace=False).tolist())
            for _ in range(n_investors)
        ]
        self.models = []
        self.investor_history = []
        started = perf_counter()
        for investor, columns in enumerate(self.feature_subsets):
            seed = self.params["seed"] + investor
            estimator_params = {**self.params["estimator"], "random_state": seed}
            model = lgb.LGBMRegressor(**estimator_params)
            investor_started = perf_counter()
            print(
                f"window={context.artifact_dir.name} investor={investor + 1}/{n_investors} "
                f"step=fit rows={len(y_train)} features={subset_size} seed={seed}", flush=True,
            )
            # No per-round validation, tuning, row resampling or silent CPU fallback.
            model.fit(x_train[:, columns], y_train)
            self.models.append(model)
            row = dict(investor=investor + 1, seed=seed, feature_count=subset_size,
                       fit_seconds=perf_counter() - investor_started)
            self.investor_history.append(row)
            print(
                f"window={context.artifact_dir.name} investor={investor + 1}/{n_investors} "
                f"step=done seconds={row['fit_seconds']:.2f} "
                f"total_seconds={perf_counter() - started:.2f}", flush=True,
            )
        del x_train
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id,
            lightgbm_version=lgb.__version__, params=self.params,
            train_rows=len(y_train), valid_rows=len(valid_data),
            feature_count=len(self.feature_columns), feature_columns=self.feature_columns,
            features_per_investor=subset_size, n_investors=n_investors,
            investor_features=[[self.feature_columns[i] for i in subset] for subset in self.feature_subsets],
            investor_seeds=[self.params["seed"] + i for i in range(n_investors)],
            fit_seconds=perf_counter() - started,
            score="0.5 * cs_zscore(mean_prediction) + 0.5 * cs_zscore(-population_std)",
        )
        if not valid_data.empty:
            # Validation diagnoses the fixed ensemble once; it does not choose subsets or weights.
            components = self.predict_components(valid_data, context)
            target = valid_data[context.label_column].to_numpy(dtype=np.float64)
            finite = np.isfinite(target)
            self.model_info["valid_labeled_rows"] = int(finite.sum())
            if finite.any():
                recognition = components.recognition.to_numpy()
                self.model_info["valid_recognition_mse"] = float(np.mean((recognition[finite] - target[finite]) ** 2))
                score = _composite_score(components).to_numpy()
                daily = pd.DataFrame(dict(trade_date=valid_data.trade_date.to_numpy(), score=score, target=target))
                correlations = []
                for _, group in daily.loc[finite].groupby("trade_date", sort=False):
                    if len(group) > 1 and group.score.nunique() > 1 and group.target.nunique() > 1:
                        value = group.score.corr(group.target)
                        if np.isfinite(value):
                            correlations.append(float(value))
                self.model_info["valid_mean_daily_ic"] = float(np.mean(correlations)) if correlations else None

    def predict_components(self, data: pd.DataFrame, context: ModelContext) -> pd.DataFrame:
        """Raw mean/std for diagnostics; prediction never reads a return label."""
        if not self.models or len(self.models) != len(self.feature_subsets):
            raise RuntimeError("investor ensemble is not completely fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("investor prediction feature columns/order changed")
        if data.trade_date.isna().any():
            raise ValueError("investor scores require a trade_date for every stock")
        mean = np.empty(len(data), dtype=np.float64)
        std = np.empty(len(data), dtype=np.float64)
        batch_size = self.params["prediction_batch_size"]
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            x = _features(data.iloc[start:stop], self.feature_columns)
            batch_mean = np.zeros(stop - start, dtype=np.float64)
            m2 = np.zeros_like(batch_mean)
            for count, (model, columns) in enumerate(zip(self.models, self.feature_subsets), start=1):
                values = np.asarray(model.predict(x[:, columns]), dtype=np.float64)
                if values.shape != batch_mean.shape or not np.isfinite(values).all():
                    raise ValueError("investor returned invalid predictions")
                # Welford accumulation: avoid both an N x investors table and cancellation.
                delta = values - batch_mean
                batch_mean += delta / count
                m2 += delta * (values - batch_mean)
            mean[start:stop] = batch_mean
            std[start:stop] = np.sqrt(np.maximum(m2 / len(self.models), 0.0))
        return pd.DataFrame(
            dict(trade_date=data.trade_date.to_numpy(), recognition=mean, negative_disagreement=-std),
            index=data.index,
        )

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        # Normalize complete date cross-sections AFTER all internal stock batches.
        return _composite_score(self.predict_components(data, context)).astype("float32")

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        if not context.diagnostics.get("enabled") or not context.diagnostics.get("write_model_info"):
            return []
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        info_path = context.artifact_dir / "model_info.json"
        history_path = context.artifact_dir / "investor_history.parquet"
        info_path.write_text(json.dumps(self.model_info, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(self.investor_history).to_parquet(history_path, index=False)
        return [info_path, history_path]


def _composite_score(components: pd.DataFrame) -> pd.Series:
    standardized = apply_cross_section_transform(
        components, "zscore", ["recognition", "negative_disagreement"], feature_fill_value=0.0,
    )
    return 0.5 * (standardized.recognition + standardized.negative_disagreement)


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    allowed = {"n_investors", "investor_feature_fraction", "seed", "prediction_batch_size", "estimator"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unsupported investor settings: {sorted(unknown)}; estimator options belong in estimator")
    params = dict(raw)
    params["n_investors"] = int(params.get("n_investors", 50))
    params["investor_feature_fraction"] = float(params.get("investor_feature_fraction", 0.5))
    params["seed"] = int(params.get("seed", 42))
    params["prediction_batch_size"] = int(params.get("prediction_batch_size", 10000))
    if params["n_investors"] < 2 or params["prediction_batch_size"] <= 0:
        raise ValueError("require at least two investors and a positive prediction_batch_size")
    if not 0 < params["investor_feature_fraction"] < 1:
        raise ValueError("investor_feature_fraction must be strictly between zero and one")
    if not 0 <= params["seed"] < 2**32 - params["n_investors"]:
        raise ValueError("investor seeds must fit in uint32")
    estimator = dict(params.get("estimator", {}))
    if "random_state" in estimator or "seed" in estimator:
        raise ValueError("use the ensemble seed; individual investor seeds are assigned deterministically")
    params["estimator"] = {
        "objective": "regression", "n_estimators": 100, "num_leaves": 31, "max_depth": -1,
        "learning_rate": 0.1, "min_child_samples": 20, "subsample": 1.0,
        "subsample_freq": 0, "colsample_bytree": 1.0, "reg_alpha": 0.0, "reg_lambda": 0.0,
        "device_type": "gpu", "max_bin": 255, "n_jobs": 8, "verbosity": -1, **estimator,
    }
    if params["estimator"]["objective"] != "regression":
        raise ValueError("investor predictions must be return regressions on the same scale")
    return params
