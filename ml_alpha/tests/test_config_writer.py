from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.features.factor_frame import FactorFrameProvider
from yq_ml_alpha.features.transforms import (
    apply_cross_section_transform,
    available_transforms,
    cross_section_zscore_erfinv_rank,
    cross_section_zscore_log_rank,
)
from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.cnn_model import CNNAlphaModel
from yq_ml_alpha.models import elstm_ranknet_model as elstm_module
from yq_ml_alpha.models.elstm_ranknet_model import (
    eLSTM,
    eLSTMCell,
    eLSTMRankNetAlphaModel,
    ranknet_loss,
)
from yq_ml_alpha.models.ic_sign_model import ICSignEqualWeightAlphaModel
from yq_ml_alpha.models.linear_model import LinearRegressionAlphaModel
from yq_ml_alpha.models.lgbm_optuna_model import LightGBMOptunaAlphaModel
from yq_ml_alpha.models.mlp_model import MLPAlphaModel
from yq_ml_alpha.models.optuna_space import suggest_params
from yq_ml_alpha.models.regularized_linear_model import (
    ElasticNetAlphaModel,
    LassoAlphaModel,
    RidgeAlphaModel,
)
from yq_ml_alpha.models.sequence_model import GRUAlphaModel, LSTMAlphaModel, RNNAlphaModel
from yq_ml_alpha.models.tree_model import RandomForestAlphaModel
from yq_ml_alpha.models.xgb_optuna_model import XGBoostOptunaAlphaModel
from yq_ml_alpha.models.xgb_model import XGBoostAlphaModel
from yq_ml_alpha.output.alpha_writer import AlphaWriter
from yq_ml_alpha.pipelines.train import build_windows


class ConfigAndWriterTests(unittest.TestCase):
    def test_config_preserves_model_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20200101, 20200131]
valid = [20200201, 20200228]
predict = [20200301, 20200331]

[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "static"
train_sample_count = 36

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"

[model.params]
learning_rate = 0.03

""",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.train_scheme.type, "static")
            self.assertEqual(config.sample.train_frequency, "monthly_end")
            self.assertEqual(config.sample.predict_frequency, "daily")
            self.assertEqual(config.train_scheme.train_sample_count, 36)
            self.assertEqual(config.train_scheme.validation_sample_count, 0)
            self.assertEqual(config.model.params["learning_rate"], 0.03)

    def test_valid_and_predict_dates_can_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20200101, 20200131]
valid = []
predict = []

[sample]
train_frequency = "monthly_end"

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertIsNone(config.dates.valid)
            self.assertIsNone(config.dates.predict)
            self.assertIsNone(config.sample.predict_frequency)

    def test_train_frequency_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20200101, 20200131]

[sample]

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sample.train_frequency"):
                load_config(path)

    def test_sample_frequency_accepts_rebalance_style_fixed_days(self) -> None:
        class Calendar:
            def between(self, start: int, end: int) -> list[int]:
                return [20260105, 20260106, 20260107, 20260108, 20260109, 20260112]

        self.assertEqual(
            sample_dates(Calendar(), (20260105, 20260112), "5"),
            [20260105, 20260112],
        )
        self.assertEqual(
            sample_dates(Calendar(), (20260105, 20260112), "every_5_days"),
            [20260105, 20260112],
        )

    def test_zscore_log_rank_fills_features_but_not_label(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": [20260131, 20260131, 20260131, 20260131],
                "ts_code": ["a", "b", "c", "d"],
                "f": [1.0, 2.0, np.nan, 4.0],
                "label": [0.1, 0.2, 0.3, np.nan],
            }
        )
        output = cross_section_zscore_log_rank(
            frame,
            ["f", "label"],
            fill_columns=["f"],
            fill_value=0.0,
        )
        self.assertEqual(output.loc[2, "f"], 0.0)
        self.assertTrue(pd.isna(output.loc[3, "label"]))
        self.assertAlmostEqual(float(output["label"].dropna().mean()), 0.0, places=7)
        self.assertAlmostEqual(float(output["label"].dropna().std(ddof=0)), 1.0, places=7)

    def test_zscore_erfinv_rank_maps_rank_to_gaussian_scores(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": [20260131, 20260131, 20260131, 20260131, 20260131],
                "ts_code": ["a", "b", "c", "d", "e"],
                "f": [1.0, 2.0, 3.0, 4.0, np.nan],
                "label": [0.1, 0.2, 0.3, 0.4, np.nan],
            }
        )
        output = cross_section_zscore_erfinv_rank(
            frame,
            ["f", "label"],
            fill_columns=["f"],
            fill_value=0.0,
        )
        finite = output.loc[:3, "f"]
        self.assertTrue(np.isfinite(finite).all())
        self.assertTrue(finite.is_monotonic_increasing)
        self.assertAlmostEqual(float(finite.mean()), 0.0, places=7)
        self.assertAlmostEqual(float(finite.std(ddof=0)), 1.0, places=7)
        self.assertEqual(output.loc[4, "f"], 0.0)
        self.assertTrue(pd.isna(output.loc[4, "label"]))

    def test_transform_registry_dispatches_aliases(self) -> None:
        frame = pd.DataFrame(
            {
                "trade_date": [20260131, 20260131, 20260131],
                "ts_code": ["a", "b", "c"],
                "f": [1.0, 2.0, np.nan],
                "label": [0.1, 0.2, 0.3],
            }
        )
        self.assertIn("zscore_log_rank", available_transforms())
        self.assertIn("zscore_erfinv_rank", available_transforms())
        output = apply_cross_section_transform(
            frame,
            "zscore_inverf_rank",
            ["f"],
            label_columns=["label"],
            feature_fill_value=0.0,
        )
        self.assertEqual(output.loc[2, "f"], 0.0)
        self.assertAlmostEqual(float(output["label"].mean()), 0.0, places=7)

    def test_monthly_rolling_sample_count_predicts_after_refit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20260101, 20260424]
valid = [20260101, 20260424]
predict = [20260301, 20260424]

[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "monthly_end"
train_sample_count = 2

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            calendar = TradingCalendar([20260130, 20260227, 20260331, 20260401, 20260402, 20260430])
            windows = build_windows(config, calendar)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].train_dates, [20260130, 20260227])
            self.assertEqual(windows[0].valid_dates, [])
            self.assertEqual(windows[0].predict_dates, [20260401, 20260402])

    def test_monthly_rolling_sample_count_can_use_next_sample_as_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20260101, 20260529]
valid = []
predict = [20260301, 20260504]

[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "monthly_end"
train_sample_count = 2
validation_sample_count = 1

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            calendar = TradingCalendar(
                [20260130, 20260227, 20260331, 20260401, 20260402, 20260430, 20260501, 20260504, 20260529]
            )
            windows = build_windows(config, calendar)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].train_dates, [20260130, 20260227])
            self.assertEqual(windows[0].valid_dates, [20260331])
            self.assertEqual(windows[0].predict_dates, [20260501, 20260504])

    def test_train_only_window_when_predict_dates_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20260101, 20260424]
valid = []
predict = []

[sample]
train_frequency = "monthly_end"

[train_scheme]
type = "rolling"
refit_frequency = "monthly_end"
train_sample_count = 2

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["utd"]

[label]
id = "future_vwap_return_20d"

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "data/model_workspace/r1/artifacts"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            calendar = TradingCalendar([20260130, 20260227, 20260331, 20260424])
            windows = build_windows(config, calendar)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0].window_id, "train_only_20260331")
            self.assertEqual(windows[0].train_dates, [20260227, 20260331])
            self.assertEqual(windows[0].valid_dates, [])
            self.assertEqual(windows[0].predict_dates, [])

    def test_alpha_writer_preserves_existing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer_a = AlphaWriter(root, "alpha_a")
            writer_b = AlphaWriter(root, "alpha_b")
            writer_a.write(
                pd.DataFrame(
                    {
                        "trade_date": [20260105, 20260105],
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "score": [1.0, 2.0],
                    }
                )
            )
            writer_b.write(
                pd.DataFrame(
                    {
                        "trade_date": [20260105],
                        "ts_code": ["000001.SZ"],
                        "score": [3.0],
                    }
                )
            )
            table = pd.read_parquet(root / "2026" / "20260105.parquet")
            self.assertIn("alpha_a", table.columns)
            self.assertIn("alpha_b", table.columns)
            self.assertEqual(str(table["alpha_a"].dtype), "float32")

    def test_factor_frame_all_columns_discovers_union_and_fills_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2026").mkdir()
            pd.DataFrame(
                {
                    "trade_date": [20260105],
                    "ts_code": ["000001.SZ"],
                    "factor_a": [1.0],
                }
            ).to_parquet(root / "2026" / "20260105.parquet", index=False)
            pd.DataFrame(
                {
                    "trade_date": [20260106],
                    "ts_code": ["000001.SZ"],
                    "factor_a": [2.0],
                    "factor_b": [3.0],
                }
            ).to_parquet(root / "2026" / "20260106.parquet", index=False)

            provider = FactorFrameProvider(root, "__all__")
            self.assertEqual(provider.feature_columns, ["factor_a", "factor_b"])
            frame = provider.load(20260105)
            self.assertEqual(list(frame.columns), ["trade_date", "ts_code", "factor_a", "factor_b"])
            self.assertTrue(pd.isna(frame.loc[0, "factor_b"]))

    def test_sequence_dataset_uses_past_sample_dates_as_timesteps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            factor_root = root / "factors"
            label_root = root / "labels"
            (factor_root / "2026").mkdir(parents=True)
            (label_root / "2026").mkdir(parents=True)
            for trade_date, values in [
                (20260130, [1.0, 2.0, 10.0, 20.0]),
                (20260227, [3.0, 4.0, 30.0, 40.0]),
                (20260331, [5.0, 6.0, 50.0, 60.0]),
            ]:
                pd.DataFrame(
                    {
                        "trade_date": [trade_date, trade_date],
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "f1": values[:2],
                        "f2": values[2:],
                    }
                ).to_parquet(factor_root / "2026" / f"{trade_date}.parquet", index=False)
            pd.DataFrame(
                {
                    "trade_date": [20260331, 20260331],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "y": [0.1, 0.2],
                }
            ).to_parquet(label_root / "2026" / "20260331.parquet", index=False)

            config_path = root / "run.toml"
            config_path.write_text(
                f"""
run_id = "r1"
alpha_id = "a1"
data_root = "{root.as_posix()}"

[dates]
train = [20260101, 20260331]
valid = []
predict = []

[sample]
train_frequency = "monthly_end"

[features]
type = "factor_frame"
root = "{factor_root.as_posix()}"
columns = ["f1", "f2"]

[label]
id = "y"
root = "{label_root.as_posix()}"

[filters]
exclude_limit = false
exclude_st = false
exclude_bj = true

[preprocess]
cross_section_transform = "none"
feature_fill_value = 0.0

[model]
name = "lstm"
class = "yq_ml_alpha.models.sequence_model.LSTMAlphaModel"
artifact_dir = "{(root / "artifacts").as_posix()}"
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            builder = DatasetBuilder(config)
            calendar = TradingCalendar([20260130, 20260227, 20260331])
            bundle = builder.load_sequence([20260331], True, calendar, 3, "monthly_end")
            self.assertEqual(
                bundle.feature_columns,
                ["f1__seq0", "f2__seq0", "f1__seq1", "f2__seq1", "f1__seq2", "f2__seq2"],
            )
            row = bundle.frame.loc[bundle.frame["ts_code"] == "000001.SZ"].iloc[0]
            self.assertEqual(
                [row[column] for column in bundle.feature_columns],
                [1.0, 10.0, 3.0, 30.0, 5.0, 50.0],
            )
            self.assertEqual(len(bundle.frame), 2)

    def test_linear_regression_model_fits_simple_relation(self) -> None:
        model = LinearRegressionAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={},
            model_search={},
        )
        train = pd.DataFrame({"trade_date": [1, 1, 1], "ts_code": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]})
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertTrue(np.allclose(pred.to_numpy(), [3.0, 5.0, 7.0], atol=1e-5))

    def test_regularized_linear_models_fit_search_and_save(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d", "e", "f"],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "y": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
            }
        )
        base = dict(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2"],
            label_column="y",
            artifact_dir=Path("tmp"),
        )
        models = [
            (
                LassoAlphaModel(),
                {
                    "search": {
                        "enabled": True,
                        "method": "random",
                        "cv": 2,
                        "n_iter": 2,
                        "space": {"alpha": [0.001, 0.01], "fit_intercept": [True]},
                    }
                },
            ),
            (
                RidgeAlphaModel(),
                {
                    "search": {
                        "enabled": True,
                        "method": "grid",
                        "cv": 2,
                        "space": {"alpha": [0.001, 0.01], "fit_intercept": [True]},
                    }
                },
            ),
            (
                ElasticNetAlphaModel(),
                {
                    "search": {
                        "enabled": True,
                        "method": "random",
                        "cv": 2,
                        "n_iter": 2,
                        "space": {"alpha": [0.001, 0.01], "l1_ratio": [0.2, 0.8]},
                    }
                },
            ),
        ]
        for model, params in models:
            context = ModelContext(model_params={}, model_search=params["search"], **base)
            model.fit(train, pd.DataFrame(), context)
            self.assertIsNotNone(model.best_params_)
            pred = model.predict(train, context)
            self.assertEqual(len(pred), len(train))
            self.assertTrue(np.isfinite(pred.to_numpy()).all())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pkl"
                model.save(path)
                loaded = type(model).load(path)
                loaded_pred = loaded.predict(train, context)
                self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_regularized_linear_search_uses_explicit_validation_when_available(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d"],
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        valid = pd.DataFrame(
            {
                "trade_date": [2, 2],
                "ts_code": ["e", "f"],
                "x": [5.0, 6.0],
                "y": [5.0, 6.0],
            }
        )
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={},
            model_search={
                "enabled": True,
                "method": "grid",
                "cv": 3,
                "space": {"alpha": [0.001, 0.01], "fit_intercept": [True]},
            },
        )
        model = RidgeAlphaModel()
        model.fit(train, valid, context)
        self.assertIsNotNone(model.best_params_)
        self.assertEqual(len(model.cv_results_["mean_test_score"]), 2)
        pred = model.predict(valid, context)
        self.assertEqual(len(pred), len(valid))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_random_forest_model_smoke_when_installed(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")
        model = RandomForestAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={"n_estimators": 5, "min_samples_leaf": 1, "min_samples_split": 2, "random_state": 7},
            model_search={},
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [4.0, 3.0, 2.0, 1.0],
                "y": [0.1, 0.2, 0.3, 0.4],
            }
        )
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_xgboost_model_smoke_when_installed(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost is not installed")
        model = XGBoostAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={"n_estimators": 2, "max_depth": 1},
            model_search={},
        )
        train = pd.DataFrame({"trade_date": [1, 1, 1], "ts_code": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]})
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 3)

    def test_xgboost_optuna_model_smoke_when_installed(self) -> None:
        try:
            import optuna  # noqa: F401
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("optuna or xgboost is not installed")
        model = XGBoostOptunaAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "n_estimators": 2,
                "max_depth": 1,
                "tree_method": "hist",
            },
            model_search={"n_trials": 1, "valid_fraction": 0.25, "random_state": 7},
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d", "e", "f"],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "y": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
            }
        )
        model.fit(train, pd.DataFrame(), context)
        self.assertIsNotNone(model.best_params_)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_lightgbm_optuna_model_smoke_when_installed(self) -> None:
        try:
            import lightgbm  # noqa: F401
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("lightgbm or optuna is not installed")
        model = LightGBMOptunaAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "n_estimators": 2,
                "num_leaves": 3,
                "min_child_samples": 1,
                "verbosity": -1,
            },
            model_search={"n_trials": 1, "valid_fraction": 0.25, "random_state": 7},
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d", "e", "f"],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                "y": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
            }
        )
        model.fit(train, pd.DataFrame(), context)
        self.assertIsNotNone(model.best_params_)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_ic_sign_equal_weight_uses_rankic_signs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ic_root = Path(tmp) / "ic"
            ic_root.mkdir()
            pd.DataFrame({"rank_ic": [0.1, 0.2, np.nan]}).to_parquet(ic_root / "pos.parquet", index=False)
            pd.DataFrame({"rank_ic": [-0.1, -0.3]}).to_parquet(ic_root / "neg.parquet", index=False)
            pd.DataFrame({"rank_ic": [np.nan, np.nan]}).to_parquet(ic_root / "nan.parquet", index=False)
            pd.DataFrame({"rank_ic": [0.1, -0.1]}).to_parquet(ic_root / "zero.parquet", index=False)
            context = ModelContext(
                run_id="r",
                alpha_id="a",
                feature_columns=["pos", "neg", "missing", "nan", "zero"],
                label_column="y",
                artifact_dir=Path("tmp"),
                model_params={"ic_root": str(ic_root), "ic_metric": "rank_ic"},
                model_search={},
            )
            model = ICSignEqualWeightAlphaModel()
            model.fit(pd.DataFrame(), pd.DataFrame(), context)
            self.assertEqual(model.signs, {"pos": 1.0, "neg": -1.0})

            data = pd.DataFrame(
                {
                    "trade_date": [1, 1],
                    "ts_code": ["a", "b"],
                    "pos": [1.0, 2.0],
                    "neg": [10.0, 20.0],
                    "missing": [100.0, 200.0],
                    "nan": [3.0, 4.0],
                    "zero": [5.0, 6.0],
                }
            )
            pred = model.predict(data, context)
            self.assertTrue(np.allclose(pred.to_numpy(), [-4.5, -9.0], atol=1e-7))

            path = Path(tmp) / "model.pkl"
            model.save(path)
            loaded = ICSignEqualWeightAlphaModel.load(path)
            loaded_pred = loaded.predict(data, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_ic_sign_equal_weight_errors_when_no_valid_ic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ic_root = Path(tmp) / "ic"
            ic_root.mkdir()
            pd.DataFrame({"rank_ic": [np.nan]}).to_parquet(ic_root / "a.parquet", index=False)
            context = ModelContext(
                run_id="r",
                alpha_id="a",
                feature_columns=["a", "b"],
                label_column="y",
                artifact_dir=Path("tmp"),
                model_params={"ic_root": str(ic_root), "ic_metric": "rank_ic"},
                model_search={},
            )
            with self.assertRaisesRegex(ValueError, "no valid IC signs"):
                ICSignEqualWeightAlphaModel().fit(pd.DataFrame(), pd.DataFrame(), context)

    def test_torch_mlp_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        model = MLPAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "hidden_layers": [4, 2],
                "epochs": 3,
                "batch_size": 2,
                "patience": 0,
                "seed": 7,
            },
            model_search={},
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [4.0, 3.0, 2.0, 1.0],
                "y": [0.1, 0.2, 0.3, 0.4],
            }
        )
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = MLPAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_sequence_models_smoke_and_pad_features_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [4.0, 3.0, 2.0, 1.0],
                "x3": [0.5, 0.4, 0.3, 0.2],
                "x4": [0.1, 0.2, 0.3, 0.4],
                "x5": [2.0, 2.0, 2.0, 2.0],
                "y": [0.1, 0.2, 0.3, 0.4],
            }
        )
        base_context = dict(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2", "x3", "x4", "x5"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "sequence_length": 6,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 2,
                "patience": 0,
                "seed": 7,
            },
            model_search={},
        )
        for model_cls in [RNNAlphaModel, LSTMAlphaModel, GRUAlphaModel]:
            model = model_cls()
            context = ModelContext(**base_context)
            model.fit(train, pd.DataFrame(), context)
            self.assertGreater(len(model.loss_history), 0)
            self.assertIn("train_loss", model.loss_history[-1])
            self.assertEqual(model.model_info["epochs_run"], len(model.loss_history))
            pred = model.predict(train, context)
            self.assertEqual(len(pred), 4)
            self.assertTrue(np.isfinite(pred.to_numpy()).all())
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.pt"
                model.save(path)
                loaded = model_cls.load(path)
                self.assertEqual(len(loaded.loss_history), len(model.loss_history))
                self.assertEqual(loaded.model_info["rnn_type"], model.rnn_type)
                loaded_pred = loaded.predict(train, context)
                self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_elstm_cell_and_layer_shapes_when_installed(self) -> None:
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skipTest("torch is not installed")
        self.assertNotIn("nn.LSTM", inspect.getsource(elstm_module))
        cell = eLSTMCell(nn, input_size=3, hidden_size=5)
        x = torch.randn(4, 3, requires_grad=True)
        state = tuple(torch.zeros(4, 5) for _ in range(4))
        h, next_state = cell(x, state)
        self.assertEqual(tuple(h.shape), (4, 5))
        for item in next_state:
            self.assertEqual(tuple(item.shape), (4, 5))
        self.assertFalse(next_state[3].requires_grad)

        layer = eLSTM(nn, input_size=3, hidden_size=5, num_layers=2, dropout=0.0, batch_first=True)
        output, h_last = layer(torch.randn(4, 6, 3))
        self.assertEqual(tuple(output.shape), (4, 6, 5))
        self.assertEqual(tuple(h_last.shape), (4, 5))

    def test_ranknet_loss_uses_same_date_pairs_only_when_installed(self) -> None:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch is not installed")
        pred = torch.tensor([0.0, 1.0, 10.0])
        target = torch.tensor([2.0, 1.0, 100.0])
        date_id = torch.tensor([1, 1, 2])
        loss = ranknet_loss(pred, target, date_id, sigma=1.0, max_pairs_per_date=0)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.allclose(loss, F.softplus(torch.tensor(1.0))))

        tied = ranknet_loss(
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([1, 1]),
        )
        self.assertIsNone(tied)

        sampled = ranknet_loss(
            torch.linspace(0.0, 1.0, 50),
            torch.arange(50.0),
            torch.ones(50, dtype=torch.int64),
            max_pairs_per_date=5,
        )
        self.assertIsNotNone(sampled)
        self.assertTrue(torch.isfinite(sampled))

    def test_elstm_ranknet_model_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        model = eLSTMRankNetAlphaModel()
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1, 2, 2, 2, 2],
                "ts_code": ["a", "b", "c", "d", "a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1],
                "x2": [4.0, 3.0, 2.0, 1.0, 4.1, 3.1, 2.1, 1.1],
                "x3": [0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3],
                "x4": [0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.5],
                "y": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
            }
        )
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2", "x3", "x4"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "sequence_length": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 4,
                "patience": 0,
                "seed": 7,
                "max_pairs_per_date": 10,
            },
            model_search={},
        )
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 8)
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = eLSTMRankNetAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_cnn_model_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        model = CNNAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2", "x3", "x4"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "channels": [4],
                "kernel_size": 3,
                "hidden_size": 4,
                "epochs": 2,
                "batch_size": 2,
                "patience": 0,
                "seed": 7,
            },
            model_search={},
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1],
                "ts_code": ["a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [4.0, 3.0, 2.0, 1.0],
                "x3": [0.5, 0.4, 0.3, 0.2],
                "x4": [0.1, 0.2, 0.3, 0.4],
                "y": [0.1, 0.2, 0.3, 0.4],
            }
        )
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = CNNAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_monthly_mlp_config_parses(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "monthly_mlp_36.toml")
        self.assertEqual(config.alpha_id, "ml_alpha_mlp")
        self.assertEqual(config.features.columns, "__all__")
        self.assertEqual(config.model.class_path, "yq_ml_alpha.models.mlp_model.MLPAlphaModel")
        self.assertEqual(config.model.params["hidden_layers"], [256, 128, 64])
        self.assertTrue(config.diagnostics.enabled)
        self.assertTrue(config.diagnostics.print_epoch)
        self.assertTrue(config.diagnostics.write_loss_history)
        self.assertTrue(config.diagnostics.write_model_info)
        self.assertTrue(config.diagnostics.write_window_summary)

    def test_monthly_ic_sign_config_parses(self) -> None:
        config = load_config(
            Path(__file__).resolve().parents[1] / "configs" / "monthly_ic_sign_equal_weight.toml"
        )
        self.assertEqual(config.alpha_id, "ml_alpha_ic_sign_ew")
        self.assertEqual(config.features.columns, "__all__")
        self.assertEqual(config.model.class_path, "yq_ml_alpha.models.ic_sign_model.ICSignEqualWeightAlphaModel")
        self.assertEqual(config.model.params["ic_metric"], "rank_ic")
        self.assertFalse(config.diagnostics.enabled)

    def test_new_model_configs_parse(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        expected = {
            "mdl_000001.toml": ("mdl_000001", "LinearRegressionAlphaModel"),
            "mdl_000002.toml": ("mdl_000002", "LassoAlphaModel"),
            "mdl_000003.toml": ("mdl_000003", "RidgeAlphaModel"),
            "mdl_000004.toml": ("mdl_000004", "ElasticNetAlphaModel"),
            "monthly_rf_36.toml": ("ml_alpha_rf", "RandomForestAlphaModel"),
            "monthly_rnn_36.toml": ("ml_alpha_rnn", "RNNAlphaModel"),
            "monthly_gru_36.toml": ("ml_alpha_gru", "GRUAlphaModel"),
            "monthly_elstm_ranknet_36.toml": ("ml_alpha_elstm_ranknet", "eLSTMRankNetAlphaModel"),
            "monthly_cnn_36.toml": ("ml_alpha_cnn", "CNNAlphaModel"),
            "monthly_xgb_optuna_36.toml": ("ml_alpha_xgb_optuna", "XGBoostOptunaAlphaModel"),
            "monthly_lgbm_optuna_36.toml": ("ml_alpha_lgbm_optuna", "LightGBMOptunaAlphaModel"),
        }
        for filename, (alpha_id, class_name) in expected.items():
            config = load_config(config_dir / filename)
            self.assertEqual(config.alpha_id, alpha_id)
            self.assertTrue(config.model.class_path.endswith(class_name))
            self.assertEqual(config.features.columns, "__all__")
            if filename in {
                "mdl_000002.toml",
                "mdl_000003.toml",
                "mdl_000004.toml",
                "monthly_xgb_optuna_36.toml",
                "monthly_lgbm_optuna_36.toml",
                "monthly_mlp_36.toml",
                "monthly_rnn_36.toml",
                "monthly_gru_36.toml",
                "monthly_elstm_ranknet_36.toml",
                "monthly_cnn_36.toml",
            }:
                self.assertEqual(config.train_scheme.validation_sample_count, 1)
            else:
                self.assertEqual(config.train_scheme.validation_sample_count, 0)
            if class_name in {"RNNAlphaModel", "GRUAlphaModel", "eLSTMRankNetAlphaModel"}:
                self.assertEqual(config.model.params["sequence_length"], 6)
            if class_name == "eLSTMRankNetAlphaModel":
                self.assertEqual(config.model.params["max_pairs_per_date"], 20000)
                self.assertEqual(config.model.params["sigma"], 1.0)

    def test_tuned_configs_expose_search_space(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        lasso = load_config(config_dir / "mdl_000002.toml")
        self.assertIn("alpha", lasso.model.search["space"])

        xgb = load_config(config_dir / "monthly_xgb_optuna_36.toml")
        self.assertIn("space", xgb.model.search)
        self.assertEqual(xgb.model.search["space"]["n_estimators"]["type"], "int")
        self.assertTrue(xgb.model.search["space"]["learning_rate"]["log"])

        lgbm = load_config(config_dir / "monthly_lgbm_optuna_36.toml")
        self.assertIn("num_leaves", lgbm.model.search["space"])

    def test_optuna_space_supports_toml_distributions(self) -> None:
        class FakeTrial:
            def suggest_int(self, name, low, high, **kwargs):
                return ("int", name, low, high, kwargs)

            def suggest_float(self, name, low, high, **kwargs):
                return ("float", name, low, high, kwargs)

            def suggest_categorical(self, name, choices):
                return ("categorical", name, choices)

        params = suggest_params(
            FakeTrial(),
            {
                "a": {"type": "int", "low": 1, "high": 5, "step": 2},
                "b": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
                "c": {"choices": ["x", "y"]},
            },
            {},
        )
        self.assertEqual(params["a"], ("int", "a", 1, 5, {"step": 2}))
        self.assertEqual(params["b"], ("float", "b", 0.1, 1.0, {"log": True}))
        self.assertEqual(params["c"], ("categorical", "c", ["x", "y"]))


if __name__ == "__main__":
    unittest.main()
