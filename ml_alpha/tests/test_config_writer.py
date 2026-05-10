from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.features.transforms import cross_section_zscore_log_rank
from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.linear_model import LinearRegressionAlphaModel
from yq_ml_alpha.models.mlp_model import MLPAlphaModel
from yq_ml_alpha.models.xgb_model import XGBoostAlphaModel
from yq_ml_alpha.output.alpha_writer import AlphaWriter
from yq_ml_alpha.pipelines.train import build_windows


class ConfigAndWriterTests(unittest.TestCase):
    def test_config_preserves_model_and_tuning_params(self) -> None:
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
frequency = "monthly_end"
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

[tuning]
enabled = true
method = "optuna"
n_trials = 10
""",
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.train_scheme.type, "static")
            self.assertEqual(config.sample.train_frequency, "monthly_end")
            self.assertEqual(config.sample.predict_frequency, "daily")
            self.assertEqual(config.train_scheme.train_sample_count, 36)
            self.assertEqual(config.model.params["learning_rate"], 0.03)
            self.assertEqual(config.tuning.params["n_trials"], 10)

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
frequency = "monthly_end"
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
            self.assertEqual(windows[0].predict_dates, [20260401, 20260402])

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

    def test_linear_regression_model_fits_simple_relation(self) -> None:
        model = LinearRegressionAlphaModel()
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["x"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={},
            tuning_params={},
        )
        train = pd.DataFrame({"trade_date": [1, 1, 1], "ts_code": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]})
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertTrue(np.allclose(pred.to_numpy(), [3.0, 5.0, 7.0], atol=1e-5))

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
            tuning_params={},
        )
        train = pd.DataFrame({"trade_date": [1, 1, 1], "ts_code": ["a", "b", "c"], "x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]})
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), 3)

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
            tuning_params={},
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

    def test_monthly_mlp_config_parses(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "configs" / "examples" / "monthly_mlp_36.toml")
        self.assertEqual(config.alpha_id, "ml_alpha_mlp")
        self.assertEqual(config.model.class_path, "yq_ml_alpha.models.mlp_model.MLPAlphaModel")
        self.assertEqual(config.model.params["hidden_layers"], [128, 64])


if __name__ == "__main__":
    unittest.main()
