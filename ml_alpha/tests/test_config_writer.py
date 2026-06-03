from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import DatasetBuilder, DatasetBundle
from yq_ml_alpha.data.sampler import refit_dates, sample_dates
from yq_ml_alpha.features.factor_frame import FactorFrameProvider
from yq_ml_alpha.features.bar_panel import BarPanelProvider, MultiBarPanelProvider
from yq_ml_alpha.features.logsig_signature import (
    LogsigSignatureProvider,
    _logsignature_from_volume_fallback,
    _signature_from_volume,
    signature_width,
)
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
from yq_ml_alpha.models.logsig_orthogonal_mlp_model import LogsigOrthogonalMLPAlphaModel
from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel
from yq_ml_alpha.models.multi_bar_gru_model import MultiBarGRUAlphaModel
from yq_ml_alpha.models.optuna_space import suggest_params
from yq_ml_alpha.models.pca_ols_model import PCAOLSAlphaModel
from yq_ml_alpha.models.residual_multi_bar_gru_model import ResidualMultiBarGRUAlphaModel
from yq_ml_alpha.models.regularized_linear_model import (
    ElasticNetAlphaModel,
    LassoAlphaModel,
    RidgeAlphaModel,
)
from yq_ml_alpha.models.sequence_model import GRUAlphaModel, LSTMAlphaModel, RNNAlphaModel, _negative_ic_loss
from yq_ml_alpha.models.tree_model import RandomForestAlphaModel
from yq_ml_alpha.models.xgb_optuna_model import XGBoostOptunaAlphaModel
from yq_ml_alpha.models.xgb_model import XGBoostAlphaModel
from yq_ml_alpha.output.alpha_writer import AlphaWriter
from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter
from yq_ml_alpha.output.factor_metadata import write_factor_metadata
from yq_ml_alpha.pipelines.runtime import build_windows, _split_by_validation_ratio


def _skip_unless_sklearn_ready(testcase: unittest.TestCase) -> None:
    try:
        from sklearn.decomposition import PCA  # noqa: F401
    except Exception as exc:
        testcase.skipTest(f"scikit-learn runtime is not available: {exc}")


def _skip_unless_xgboost_sklearn_ready(testcase: unittest.TestCase) -> None:
    try:
        import xgboost as xgb

        xgb.XGBRegressor(n_estimators=1)
    except Exception as exc:
        testcase.skipTest(f"xgboost sklearn runtime is not available: {exc}")


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

    def test_semiannual_end_frequency(self) -> None:
        calendar = TradingCalendar([20260629, 20260630, 20260701, 20261230, 20261231, 20270104])
        self.assertEqual(
            sample_dates(calendar, (20260629, 20261231), "semiannual_end"),
            [20260630, 20261231],
        )
        self.assertEqual(
            refit_dates(calendar, [20260629, 20260630, 20260701, 20261230, 20261231], "semiannual_end"),
            [20260630, 20261231],
        )

    def test_annual_end_frequency(self) -> None:
        calendar = TradingCalendar([20251231, 20260105, 20261230, 20261231, 20270104])
        self.assertEqual(
            sample_dates(calendar, (20260105, 20261231), "annual_end"),
            [20261231],
        )
        self.assertEqual(
            refit_dates(calendar, [20260105, 20261230, 20261231], "annual_end"),
            [20261231],
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

    def test_validation_ratio_splits_sampled_dates_by_expected_counts(self) -> None:
        train, valid = _split_by_validation_ratio(list(range(36)), 0.2)
        self.assertEqual(len(train), 29)
        self.assertEqual(len(valid), 7)

        train, valid = _split_by_validation_ratio(list(range(35)), 0.2)
        self.assertEqual(len(train), 28)
        self.assertEqual(len(valid), 7)

        train, valid = _split_by_validation_ratio(list(range(34)), 0.2)
        self.assertEqual(len(train), 27)
        self.assertEqual(len(valid), 7)

        train, valid = _split_by_validation_ratio(list(range(2)), 0.2)
        self.assertEqual(len(train), 1)
        self.assertEqual(len(valid), 1)

    def test_rolling_validation_ratio_uses_fixed_step_sample_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [1, 100]
valid = []
predict = [62, 100]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "every_10_days"
train_lookback = "60d"
validation_ratio = 0.25

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
            calendar = TradingCalendar(list(range(1, 101)))
            windows = build_windows(config, calendar)
            self.assertGreaterEqual(len(windows), 1)
            self.assertEqual(windows[0].train_dates, [2, 22])
            self.assertEqual(windows[0].valid_dates, [42])
            self.assertEqual(windows[0].predict_dates, list(range(63, 73)))

    def test_rolling_validation_ratio_uses_train_lookback_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.toml"
            path.write_text(
                """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20110101, 20141231]
valid = []
predict = [20110101, 20141231]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "semiannual_end"
train_lookback = "3y"
validation_ratio = 0.2

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
            dates = [int(item.strftime("%Y%m%d")) for item in pd.bdate_range("2011-01-03", "2014-12-31")]
            calendar = TradingCalendar(dates)
            windows = build_windows(config, calendar)
            self.assertGreaterEqual(len(windows), 1)
            self.assertEqual(windows[0].predict_dates[0], 20140101)
            self.assertGreaterEqual(windows[0].train_dates[0], 20110101)
            self.assertLess(windows[0].valid_dates[-1], 20131231)
            self.assertTrue(all(20110101 <= date < 20131231 for date in windows[0].train_dates + windows[0].valid_dates))

    def test_validation_ratio_conflicts_with_fixed_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20260101, 20260424]
valid = []
predict = [20260301, 20260424]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "semiannual_end"
train_lookback = "3y"
validation_ratio = 0.2
{extra}

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
"""
            path = Path(tmp) / "run.toml"
            path.write_text(base.format(extra="validation_sample_count = 1"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_ratio cannot be used with validation_sample_count"):
                load_config(path)

            path.write_text(base.format(extra="train_sample_count = 36"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_ratio cannot be used with train_sample_count"):
                load_config(path)

            path.write_text(base.format(extra=""), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.train_scheme.train_lookback, "3y")

            path.write_text(base.format(extra="").replace('train_lookback = "3y"\n', ""), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_ratio requires train_lookback for rolling"):
                load_config(path)

    def test_train_lookback_conflicts_with_static_and_sample_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [20260101, 20260424]
valid = []
predict = [20260301, 20260424]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "{scheme}"
refit_frequency = "semiannual_end"
train_lookback = "3y"
{extra}

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
"""
            path = Path(tmp) / "run.toml"
            path.write_text(base.format(scheme="static", extra=""), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_lookback is not supported for static"):
                load_config(path)

            path.write_text(base.format(scheme="rolling", extra="train_sample_count = 36"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_lookback cannot be used with train_sample_count"):
                load_config(path)

    def test_expanding_validation_ratio_can_use_full_history_or_lookback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = """
run_id = "r1"
alpha_id = "a1"

[dates]
train = [1, 100]
valid = []
predict = [62, 100]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "expanding"
refit_frequency = "every_10_days"
validation_ratio = 0.25
{lookback}

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
"""
            calendar = TradingCalendar(list(range(1, 101)))
            path = Path(tmp) / "run.toml"
            path.write_text(base.format(lookback=""), encoding="utf-8")
            windows = build_windows(load_config(path), calendar)
            self.assertEqual(windows[0].train_dates, [1, 21, 41])
            self.assertEqual(windows[0].valid_dates, [61])

            path.write_text(base.format(lookback='train_lookback = "40d"'), encoding="utf-8")
            windows = build_windows(load_config(path), calendar)
            self.assertEqual(windows[0].train_dates, [22])
            self.assertEqual(windows[0].valid_dates, [42])

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

    def test_factor_config_uses_factor_identity(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "factors" / "multi_bar_gru_daily_15m.toml")
        self.assertEqual(config.factor_id, "multi_bar_gru_daily_15m")
        self.assertEqual(config.run_id, "multi_bar_gru_daily_15m")
        self.assertEqual(config.alpha_id, "multi_bar_gru_daily_15m")
        self.assertEqual(config.output.kind, "factor")
        self.assertEqual(config.output.id, "multi_bar_gru_daily_15m")
        self.assertIn("data\\factors", str(config.output.root))
        self.assertNotIn("mdl_", str(config.model.artifact_dir))

    def test_logsig_alpha_v_config_uses_semantic_identity(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "factors" / "logsig_alpha_v.toml")
        self.assertEqual(config.factor_id, "logsig_alpha_v")
        self.assertEqual(config.output.id, "logsig_alpha_v")
        self.assertEqual(config.features.type, "logsig_signature")
        self.assertIn("data\\derived\\stock\\bar\\5m", str(config.features.root))
        self.assertEqual(config.features.columns, "__all__")
        self.assertEqual(config.features.params["lookback_days"], 20)
        self.assertEqual(config.features.params["bar_size"], 5)
        self.assertEqual(config.features.params["order"], 10)
        self.assertEqual(config.features.params["volume_column"], "volume")
        self.assertEqual(config.features.params["cache_days"], "auto")
        self.assertEqual(config.label.id, "future_vwap_return_5d")
        self.assertEqual(config.sample.train_frequency, "5")
        self.assertEqual(config.sample.predict_frequency, "daily")
        self.assertEqual(config.train_scheme.refit_frequency, "annual_end")
        self.assertEqual(config.train_scheme.train_lookback, "4y")
        self.assertEqual(config.train_scheme.validation_ratio, 0.25)
        self.assertEqual(config.model.params["base_factors"], 8)
        self.assertEqual(config.model.params["orthogonal_lambda"], 0.05)
        self.assertEqual(config.model.params["neutralize"], "barra:SIZE+sector")

    def test_logsig_signature_provider_exposes_expected_feature_columns(self) -> None:
        provider = LogsigSignatureProvider(
            "data/derived/stock/bar/5m",
            "__all__",
            {"lookback_days": 20, "bar_size": 5, "order": 10},
        )
        self.assertEqual(len(provider.feature_columns), 226)
        self.assertEqual(provider.feature_columns[0], "logsig_0001")
        self.assertEqual(provider.feature_columns[-1], "logsig_0226")
        self.assertEqual(signature_width(10), 226)

    def test_logsig_signature_provider_auto_cache_uses_target_stride(self) -> None:
        provider = LogsigSignatureProvider(
            "data/derived/stock/bar/5m",
            "__all__",
            {"lookback_days": 20, "bar_size": 5, "order": 1, "cache_days": "auto"},
        )
        provider.set_calendar_dates(list(range(1, 41)))
        provider.set_cache_days_for_target_dates([20, 25, 30])
        self.assertEqual(provider.cache_days, 15)
        provider.set_cache_days_for_target_dates([20, 21, 22])
        self.assertEqual(provider.cache_days, 19)

    def test_logsig_signature_provider_explicit_cache_overrides_auto_stride(self) -> None:
        provider = LogsigSignatureProvider(
            "data/derived/stock/bar/5m",
            "__all__",
            {"lookback_days": 20, "bar_size": 5, "order": 1, "cache_days": 7},
        )
        provider.set_calendar_dates(list(range(1, 41)))
        provider.set_cache_days_for_target_dates([20, 21, 22])
        self.assertEqual(provider.cache_days, 7)

    def test_logsig_signature_matches_reference_lead_lag_path(self) -> None:
        volumes = np.array([1.0, 10.0], dtype=np.float64)
        actual = _signature_from_volume(volumes, 3)
        reference = self._reference_lead_lag_signature(np.log(np.maximum(volumes, 1.0)), 3)
        self.assertTrue(np.allclose(actual, reference, atol=1e-12))

    def test_logsig_signature_clips_zero_volume_before_log(self) -> None:
        with_zero = _signature_from_volume(np.array([0.0, 10.0], dtype=np.float64), 2)
        clipped = _signature_from_volume(np.array([1.0, 10.0], dtype=np.float64), 2)
        self.assertTrue(np.allclose(with_zero, clipped, atol=1e-12))

    def test_logsig_signature_provider_reuses_cached_bar_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bar" / "5m"
            for trade_date, base in [(20260101, 1.0), (20260102, 2.0), (20260105, 3.0)]:
                self._write_bar_day(
                    root,
                    trade_date,
                    [
                        ("000001.SZ", [base, base + 1.0]),
                        ("000002.SZ", [base + 2.0, base + 3.0]),
                    ],
                )
            provider = LogsigSignatureProvider(
                root,
                "__all__",
                {"lookback_days": 2, "bar_size": 120, "order": 2, "cache_days": 4},
            )
            provider.set_calendar_dates([20260101, 20260102, 20260105])
            original = pd.read_parquet
            read_paths: list[str] = []
            read_columns: list[list[str]] = []

            def counting_read_parquet(path, *args, **kwargs):
                read_paths.append(str(path))
                read_columns.append(list(kwargs.get("columns", [])))
                return original(path, *args, **kwargs)

            with mock.patch("pandas.read_parquet", side_effect=counting_read_parquet):
                first = provider.load(20260102)
                second = provider.load(20260105)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertEqual(len(read_paths), 3)
            self.assertEqual(len(set(read_paths)), 3)
            self.assertTrue(
                all(columns == ["trade_date", "ts_code", "bar_index", "volume"] for columns in read_columns)
            )

    def test_logsig_signature_provider_calls_rust_batch_once_per_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bar" / "5m"
            for trade_date, base in [(20260101, 1.0), (20260102, 2.0)]:
                self._write_bar_day(
                    root,
                    trade_date,
                    [
                        ("000001.SZ", [base, base + 1.0]),
                        ("000002.SZ", [base + 2.0, base + 3.0]),
                    ],
                )
            provider = LogsigSignatureProvider(
                root,
                "__all__",
                {"lookback_days": 2, "bar_size": 120, "order": 2},
            )
            provider.set_calendar_dates([20260101, 20260102])
            calls = []

            class FakeRust:
                @staticmethod
                def logsig_signature_batch(volume, order):
                    calls.append((volume.copy(), order))
                    return np.ones((volume.shape[0], signature_width(order)), dtype=np.float32)

            logs: list[str] = []
            with mock.patch.dict(sys.modules, {"yq_factor_engine_py": FakeRust}):
                with mock.patch("pandas.concat", side_effect=AssertionError("pd.concat should not be used")):
                    output = provider.load(20260102, progress=logs.append)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0].shape, (2, 4))
            self.assertEqual(calls[0][1], 2)
            self.assertEqual(len(output), 2)
            self.assertTrue(any("backend=rust" in line for line in logs))

    def test_logsig_signature_provider_falls_back_to_numba_when_rust_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bar" / "5m"
            self._write_bar_day(root, 20260101, [("000001.SZ", [1.0, 10.0])])
            provider = LogsigSignatureProvider(
                root,
                "__all__",
                {"lookback_days": 1, "bar_size": 120, "order": 3},
            )
            provider.set_calendar_dates([20260101])
            logs: list[str] = []
            with mock.patch.dict(sys.modules, {"yq_factor_engine_py": None}):
                output = provider.load(20260101, progress=logs.append)
            expected = _logsignature_from_volume_fallback(np.array([1.0, 10.0], dtype=np.float64), 3)
            actual = output[provider.feature_columns].to_numpy(dtype="float64")[0]
            self.assertTrue(np.allclose(actual, expected, atol=1e-6))
            self.assertTrue(any("backend=numba_signature_fallback" in line for line in logs))

    def test_logsig_signature_provider_returns_empty_for_incomplete_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "bar" / "5m"
            self._write_bar_day(root, 20260101, [("000001.SZ", [1.0, 2.0])])
            self._write_bar_day(root, 20260102, [("000001.SZ", [3.0])])
            provider = LogsigSignatureProvider(
                root,
                "__all__",
                {"lookback_days": 2, "bar_size": 120, "order": 2},
            )
            provider.set_calendar_dates([20260101, 20260102])
            output = provider.load(20260102)
            self.assertTrue(output.empty)
            self.assertEqual(list(output.columns), ["trade_date", "ts_code", *provider.feature_columns])

    @staticmethod
    def _write_bar_day(root: Path, trade_date: int, symbols: list[tuple[str, list[float]]]) -> None:
        path = root / str(trade_date)[:4] / f"{trade_date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for ts_code, volumes in symbols:
            for bar_index, volume in enumerate(volumes):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "bar_index": bar_index,
                        "volume": volume,
                    }
                )
        pd.DataFrame(rows).to_parquet(path, index=False)

    @staticmethod
    def _reference_lead_lag_signature(values: np.ndarray, order: int) -> np.ndarray:
        state: dict[tuple[int, ...], float] = {(): 1.0}

        def append_axis_segment(axis: int, delta: float) -> None:
            segment: dict[tuple[int, ...], float] = {(): 1.0}
            value = 1.0
            for length in range(1, order + 1):
                value *= delta / length
                segment[(axis,) * length] = value
            updated: dict[tuple[int, ...], float] = {}
            for word, word_value in state.items():
                for suffix, suffix_value in segment.items():
                    combined = word + suffix
                    if len(combined) <= order:
                        updated[combined] = updated.get(combined, 0.0) + word_value * suffix_value
            state.clear()
            state.update(updated)

        for idx in range(1, len(values)):
            delta = float(values[idx] - values[idx - 1])
            append_axis_segment(0, delta)
            append_axis_segment(1, delta)

        output = []
        for level in range(1, order + 1):
            for word in range(2**level):
                letters = tuple((word >> bit) & 1 for bit in range(level - 1, -1, -1))
                output.append(state.get(letters, 0.0))
        return np.asarray(output, dtype=np.float64)

    def test_daily_wide_writer_overwrites_target_and_unions_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            path = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105, 20260105],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "bar_gru_15m": [100.0, 200.0],
                    "old_factor": [1.0, 2.0],
                }
            ).to_parquet(path, index=False)

            writer = DailyWideWriter(output_root, "bar_gru_15m", layout="standard", write_workers=1)
            writer.write(
                pd.DataFrame(
                    {
                        "trade_date": [20260105, 20260105],
                        "ts_code": ["000002.SZ", "000003.SZ"],
                        "score": [9.0, 10.0],
                    }
                ),
                coverage_dates=[20260105],
            )

            table = pd.read_parquet(path)
            self.assertEqual(list(table["ts_code"]), ["000001.SZ", "000002.SZ", "000003.SZ"])
            self.assertTrue(pd.isna(table.loc[0, "bar_gru_15m"]))
            self.assertEqual(float(table.loc[1, "bar_gru_15m"]), 9.0)
            self.assertEqual(float(table.loc[2, "bar_gru_15m"]), 10.0)
            self.assertEqual(float(table.loc[0, "old_factor"]), 1.0)
            self.assertEqual(str(table["bar_gru_15m"].dtype), "float32")

    def test_daily_wide_writer_keeps_coverage_schema_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            first = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            second = output_root / "stock" / "daily" / "2026" / "20260106.parquet"
            first.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105],
                    "ts_code": ["000001.SZ"],
                    "old_factor": [1.0],
                }
            ).to_parquet(first, index=False)
            pd.DataFrame(
                {
                    "trade_date": [20260106],
                    "ts_code": ["000002.SZ"],
                }
            ).to_parquet(second, index=False)

            writer = DailyWideWriter(output_root, "bar_gru_15m", layout="standard", write_workers=1)
            writer.write(
                pd.DataFrame(
                    {
                        "trade_date": [20260105],
                        "ts_code": ["000001.SZ"],
                        "score": [2.0],
                    }
                ),
                coverage_dates=[20260105, 20260106],
            )

            first_table = pd.read_parquet(first)
            second_table = pd.read_parquet(second)
            self.assertEqual(list(first_table.columns), list(second_table.columns))
            self.assertEqual(list(first_table.columns), ["trade_date", "ts_code", "bar_gru_15m", "old_factor"])
            self.assertTrue(pd.isna(second_table.loc[0, "bar_gru_15m"]))
            self.assertTrue(pd.isna(second_table.loc[0, "old_factor"]))

    def test_daily_wide_writer_uses_full_schema_dates_across_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            first = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            second = output_root / "stock" / "daily" / "2026" / "20260106.parquet"
            first.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105],
                    "ts_code": ["000001.SZ"],
                    "old_a": [1.0],
                }
            ).to_parquet(first, index=False)
            pd.DataFrame(
                {
                    "trade_date": [20260106],
                    "ts_code": ["000002.SZ"],
                    "old_b": [2.0],
                }
            ).to_parquet(second, index=False)

            writer = DailyWideWriter(output_root, "bar_gru_15m", layout="standard", write_workers=1)
            schema_dates = [20260105, 20260106]
            writer.write(
                pd.DataFrame({"trade_date": [20260105], "ts_code": ["000001.SZ"], "score": [3.0]}),
                coverage_dates=[20260105],
                schema_dates=schema_dates,
            )
            writer.write(
                pd.DataFrame(columns=["trade_date", "ts_code", "score"]),
                coverage_dates=[20260106],
                schema_dates=schema_dates,
            )

            first_table = pd.read_parquet(first)
            second_table = pd.read_parquet(second)
            expected_columns = ["trade_date", "ts_code", "bar_gru_15m", "old_a", "old_b"]
            self.assertEqual(list(first_table.columns), expected_columns)
            self.assertEqual(list(second_table.columns), expected_columns)
            self.assertEqual(float(first_table.loc[0, "bar_gru_15m"]), 3.0)
            self.assertTrue(pd.isna(second_table.loc[0, "bar_gru_15m"]))

    def test_daily_wide_writer_uses_daily_pv_for_empty_coverage_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            base_root = Path(tmp) / "daily_pv"
            (base_root / "2026").mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260106, 20260106],
                    "ts_code": ["000004.SZ", "000005.SZ"],
                }
            ).to_parquet(base_root / "2026" / "20260106.parquet", index=False)

            writer = DailyWideWriter(
                output_root,
                "bar_gru_15m",
                layout="standard",
                base_root=base_root,
                write_workers=1,
            )
            writer.write(pd.DataFrame(columns=["trade_date", "ts_code", "score"]), coverage_dates=[20260106])

            table = pd.read_parquet(output_root / "stock" / "daily" / "2026" / "20260106.parquet")
            self.assertEqual(list(table.columns), ["trade_date", "ts_code", "bar_gru_15m"])
            self.assertEqual(list(table["ts_code"]), ["000004.SZ", "000005.SZ"])
            self.assertTrue(table["bar_gru_15m"].isna().all())

    def test_daily_wide_writer_ensure_output_column_skips_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            path = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105],
                    "ts_code": ["000001.SZ"],
                    "bar_gru_15m": [7.0],
                    "old_factor": [1.0],
                }
            ).to_parquet(path, index=False)

            writer = DailyWideWriter(output_root, "bar_gru_15m", layout="standard", write_workers=1)
            self.assertEqual(writer.dates_missing_output_column([20260105]), [])
            self.assertEqual(writer.ensure_output_column([20260105]), [])

            table = pd.read_parquet(path)
            self.assertEqual(float(table.loc[0, "bar_gru_15m"]), 7.0)
            self.assertEqual(float(table.loc[0, "old_factor"]), 1.0)

    def test_daily_wide_writer_ensure_output_column_adds_missing_target_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            path = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            path.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105],
                    "ts_code": ["000001.SZ"],
                    "old_factor": [1.0],
                }
            ).to_parquet(path, index=False)

            writer = DailyWideWriter(output_root, "bar_gru_15m", layout="standard", write_workers=1)
            self.assertEqual(writer.dates_missing_output_column([20260105]), [20260105])
            written = writer.ensure_output_column([20260105])

            self.assertEqual(written, [path])
            table = pd.read_parquet(path)
            self.assertEqual(list(table.columns), ["trade_date", "ts_code", "bar_gru_15m", "old_factor"])
            self.assertTrue(table["bar_gru_15m"].isna().all())
            self.assertEqual(float(table.loc[0, "old_factor"]), 1.0)

    def test_daily_wide_writer_ensure_output_column_uses_daily_pv_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "factors"
            base_root = Path(tmp) / "daily_pv"
            (base_root / "2026").mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105, 20260105],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                }
            ).to_parquet(base_root / "2026" / "20260105.parquet", index=False)

            writer = DailyWideWriter(
                output_root,
                "bar_gru_15m",
                layout="standard",
                base_root=base_root,
                write_workers=1,
            )
            written = writer.ensure_output_column([20260105])

            path = output_root / "stock" / "daily" / "2026" / "20260105.parquet"
            self.assertEqual(written, [path])
            table = pd.read_parquet(path)
            self.assertEqual(list(table.columns), ["trade_date", "ts_code", "bar_gru_15m"])
            self.assertEqual(list(table["ts_code"]), ["000001.SZ", "000002.SZ"])
            self.assertTrue(table["bar_gru_15m"].isna().all())

    def test_factor_metadata_writer_uses_factor_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_factor.toml"
            path.write_text(
                f"""
factor_id = "semantic_factor"
data_root = "data"

[output]
kind = "factor"
id = "semantic_factor"
root = "{Path(tmp).as_posix()}/factors"
asset = "stock"
frequency = "daily"

[dates]
train = [20200101, 20200131]
valid = []
predict = [20200201, 20200228]

[sample]
train_frequency = "20"
predict_frequency = "daily"

[train_scheme]
type = "rolling"
refit_frequency = "semiannual_end"
train_lookback = "3y"
validation_ratio = 0.2

[label]
id = "future_vwap_return_20d"

[features]
type = "bar_panel"
root = "data/derived/stock/bar/15m"
columns = ["open", "high", "low", "close", "vwap", "volume"]

[features.params]
source_frequency = "minute_bar"
bar_size = 15
lookback_sessions = 20

[model]
name = "bar_gru"
class = "yq_ml_alpha.models.bar_gru_model.BarGRUAlphaModel"
""",
                encoding="utf-8",
            )
            config = load_config(path)
            metadata_path = write_factor_metadata(config)
            self.assertIsNotNone(metadata_path)
            table = pd.read_parquet(metadata_path)
            row = table.iloc[0].to_dict()
            self.assertEqual(row["factor_id"], "semantic_factor")
            self.assertEqual(row["output_column"], "semantic_factor")
            self.assertEqual(row["name"], "semantic_factor")
            self.assertNotIn("mdl_", "".join(str(value) for value in row.values()))

    def test_postprocess_neutralize_config_raises_migration_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "plain_factor.toml"
            path.write_text(
                f"""
factor_id = "plain_factor"
data_root = "data"

[output]
kind = "factor"
id = "plain_factor"
root = "{root.as_posix()}/factors"

[dates]
train = [20260101, 20260131]

[sample]
train_frequency = "daily"

[label]
id = "future_vwap_return_5d"

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = ["x"]

[model]
name = "mean"
class = "yq_ml_alpha.models.base.MeanFeatureAlphaModel"
artifact_dir = "{root.as_posix()}/artifacts"

[postprocess]
neutralize = "none"
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "model.params.neutralize"):
                load_config(path)

    def test_logsig_rust_neutralization_adapter_restores_order(self) -> None:
        from yq_ml_alpha.models.logsig_orthogonal_mlp_model import _apply_rust_neutralization

        class FakeNeutralizer:
            calls: list[tuple] = []

            @staticmethod
            def neutralize_daily(*args):
                FakeNeutralizer.calls.append(args)
                return [100.0, None, 300.0]

        source = pd.DataFrame(
            {"trade_date": [20260105, 20260105, 20260106], "ts_code": ["a", "b", "c"]},
            index=[10, 11, 12],
        )
        score = pd.Series([1.0, np.nan, 3.0], index=source.index, dtype="float32")
        with mock.patch.dict(sys.modules, {"yq_factor_engine_py": FakeNeutralizer}):
            output = _apply_rust_neutralization(source, score, "barra:SIZE+sector")

        self.assertEqual(output.index.tolist(), [10, 11, 12])
        self.assertEqual(output.tolist()[0], 100.0)
        self.assertTrue(np.isnan(output.tolist()[1]))
        self.assertEqual(output.tolist()[2], 300.0)
        call = FakeNeutralizer.calls[0]
        self.assertEqual(call[0], [20260105, 20260105, 20260106])
        self.assertEqual(call[1], ["a", "b", "c"])
        self.assertEqual(call[2], [1.0, None, 3.0])
        self.assertEqual(call[3], "barra:SIZE+sector")
        self.assertEqual(call[6], 20260105)
        self.assertEqual(call[7], 20260106)

    def test_logsig_rust_neutralization_missing_extension_has_build_hint(self) -> None:
        from yq_ml_alpha.models.logsig_orthogonal_mlp_model import _apply_rust_neutralization

        source = pd.DataFrame({"trade_date": [20260105], "ts_code": ["a"]})
        score = pd.Series([1.0], dtype="float32")
        with mock.patch.dict(sys.modules, {"yq_factor_engine_py": None}):
            with self.assertRaisesRegex(ImportError, "maturin develop"):
                _apply_rust_neutralization(source, score, "sector")

    def test_pipeline_rejects_wrong_config_layer(self) -> None:
        from yq_ml_alpha.pipelines import factor, model

        root = Path(__file__).resolve().parents[1]
        model_path = root / "models" / "mdl_000001.toml"
        factor_path = root / "factors" / "bar_gru_15m.toml"
        with self.assertRaisesRegex(ValueError, "model pipeline requires"):
            model.run(factor_path)
        with self.assertRaisesRegex(ValueError, "factor pipeline requires"):
            factor.run(model_path)

    def test_cli_exposes_explicit_model_and_factor_commands(self) -> None:
        from yq_ml_alpha import cli

        with mock.patch("yq_ml_alpha.pipelines.model.run", return_value=[]) as model_run, mock.patch(
            "yq_ml_alpha.pipelines.factor.run", return_value=[]
        ) as factor_run, mock.patch("yq_ml_alpha.pipelines.factor.train_only", return_value=[]) as factor_train, mock.patch(
            "yq_ml_alpha.pipelines.factor.metadata_only", return_value=[]
        ) as factor_metadata, mock.patch("yq_ml_alpha.pipelines.factor.all_metadata", return_value=[]) as factor_metadata_all:
            cli.main(["model-run", "--config", "models/mdl_000001.toml"])
            cli.main(["factor-run", "--config", "factors/bar_gru_15m.toml"])
            cli.main(["factor-run", "--config", "factors/bar_gru_15m.toml", "--resume"])
            cli.main(["factor-train", "--config", "factors/bar_gru_15m.toml", "--resume"])
            cli.main(["factor-metadata"])
            cli.main(["factor-metadata", "--config", "factors/bar_gru_15m.toml"])
            cli.main(["factor-metadata-all"])
        model_run.assert_called_once_with(Path("models/mdl_000001.toml"))
        self.assertEqual(factor_run.call_args_list[0], mock.call(Path("factors/bar_gru_15m.toml"), resume=False))
        self.assertEqual(factor_run.call_args_list[1], mock.call(Path("factors/bar_gru_15m.toml"), resume=True))
        factor_train.assert_called_once_with(Path("factors/bar_gru_15m.toml"), resume=True)
        self.assertEqual(factor_metadata.call_args_list[0], mock.call(None, None))
        self.assertEqual(factor_metadata.call_args_list[1], mock.call([Path("factors/bar_gru_15m.toml")], None))
        factor_metadata_all.assert_called_once_with(None, None, None)
        with self.assertRaises(SystemExit):
            cli.main(["run", "--config", "factors/bar_gru_15m.toml"])

    def test_factor_metadata_all_runs_rust_then_python_metadata(self) -> None:
        from yq_ml_alpha.pipelines import factor as factor_pipeline

        with mock.patch.object(factor_pipeline.subprocess, "run") as run, mock.patch.object(
            factor_pipeline, "metadata_only", return_value=[Path("data/factors/factor_metadata.parquet")]
        ) as metadata_only:
            paths = factor_pipeline.all_metadata([Path("factors/bar_gru_15m.toml")], rust_manifest="factor_engine/Cargo.toml")

        self.assertEqual(paths, [Path("data/factors/factor_metadata.parquet")])
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0][-1], "metadata")
        self.assertTrue(str(args[0][4]).endswith("factor_engine\\Cargo.toml") or str(args[0][4]).endswith("factor_engine/Cargo.toml"))
        self.assertTrue(kwargs["check"])
        metadata_only.assert_called_once_with([Path("factors/bar_gru_15m.toml")], None)

    def test_factor_train_resume_skips_existing_artifact(self) -> None:
        from yq_ml_alpha.pipelines import factor as factor_pipeline
        from yq_ml_alpha.pipelines.runtime import TrainingWindow

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            existing = artifact_dir / "w1" / "model.pkl"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"done")
            config = mock.Mock()
            config.data_root = Path(tmp)
            config.run_id = "bar_gru_15m"
            config.alpha_id = "bar_gru_15m"
            config.factor_id = "bar_gru_15m"
            config.output.kind = "factor"
            config.output.id = "bar_gru_15m"
            config.model.artifact_dir = artifact_dir
            config.diagnostics.enabled = False
            windows = [
                TrainingWindow("w1", [20260105], [], []),
                TrainingWindow("w2", [20260106], [], []),
            ]
            train_bundle = DatasetBundle(
                pd.DataFrame({"trade_date": [20260106], "ts_code": ["000001.SZ"], "label": [1.0]}),
                ["f"],
                "label",
            )
            valid_bundle = DatasetBundle(pd.DataFrame(columns=["trade_date", "ts_code", "label"]), ["f"], "label")
            model_instance = mock.Mock()

            with mock.patch.object(factor_pipeline.TradingCalendar, "load", return_value=mock.Mock()), mock.patch.object(
                factor_pipeline, "DatasetBuilder", return_value=mock.Mock()
            ), mock.patch.object(factor_pipeline, "build_windows", return_value=windows), mock.patch.object(
                factor_pipeline, "_load_bundle", side_effect=[train_bundle, valid_bundle]
            ) as load_bundle, mock.patch.object(
                factor_pipeline, "_context", return_value=mock.Mock()
            ), mock.patch.object(
                factor_pipeline, "_new_model", return_value=model_instance
            ), mock.patch.object(
                factor_pipeline, "_fit_model"
            ) as fit_model, mock.patch.object(
                factor_pipeline, "_aggregate_diagnostics", return_value=[]
            ):
                factor_pipeline.train_config(config, resume=True)

            self.assertEqual(load_bundle.call_count, 2)
            fit_model.assert_called_once()
            model_instance.save.assert_called_once()

    def test_factor_run_resume_skips_complete_window_and_predicts_missing_with_artifact(self) -> None:
        from yq_ml_alpha.pipelines import factor as factor_pipeline
        from yq_ml_alpha.pipelines.runtime import TrainingWindow

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = Path(tmp) / "artifacts"
            existing = artifact_dir / "w2" / "model.pkl"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"done")
            config = mock.Mock()
            config.data_root = Path(tmp)
            config.run_id = "bar_gru_15m"
            config.alpha_id = "bar_gru_15m"
            config.factor_id = "bar_gru_15m"
            config.output.kind = "factor"
            config.output.id = "bar_gru_15m"
            config.model.artifact_dir = artifact_dir
            config.model.class_path = "dummy.Model"
            config.diagnostics.enabled = False
            windows = [
                TrainingWindow("w1", [20250101], [], [20260105]),
                TrainingWindow("w2", [20250102], [], [20260106, 20260107]),
            ]
            writer = mock.Mock()
            writer.dates_missing_output_column.side_effect = [[], [20260107]]
            writer.ensure_output_column.return_value = []
            loaded_model = mock.Mock()
            model_class = mock.Mock()
            model_class.load.return_value = loaded_model

            with mock.patch.object(factor_pipeline.TradingCalendar, "load", return_value=mock.Mock()), mock.patch.object(
                factor_pipeline, "DatasetBuilder", return_value=mock.Mock()
            ), mock.patch.object(factor_pipeline, "_new_writer", return_value=writer), mock.patch.object(
                factor_pipeline, "write_factor_metadata", return_value=None
            ), mock.patch.object(
                factor_pipeline, "build_windows", return_value=windows
            ), mock.patch.object(
                factor_pipeline, "_predict_dates", return_value=[20260105, 20260106, 20260107]
            ), mock.patch.object(
                factor_pipeline, "_model_class", return_value=model_class
            ), mock.patch.object(
                factor_pipeline, "_new_model"
            ) as new_model, mock.patch.object(
                factor_pipeline, "_load_bundle"
            ) as load_bundle, mock.patch.object(
                factor_pipeline, "_predict_write_window", return_value=[Path(tmp) / "out.parquet"]
            ) as predict_write, mock.patch.object(
                factor_pipeline, "_aggregate_diagnostics", return_value=[]
            ):
                paths = factor_pipeline.run_config(config, resume=True)

            self.assertEqual(paths, [Path(tmp) / "out.parquet"])
            model_class.load.assert_called_once_with(existing)
            new_model.assert_not_called()
            load_bundle.assert_not_called()
            predict_window = predict_write.call_args.args[4]
            self.assertEqual(predict_window.window_id, "w2")
            self.assertEqual(predict_window.predict_dates, [20260107])
            writer.ensure_output_column.assert_called_once_with([20260105, 20260106, 20260107])

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

    def test_bar_panel_dataset_returns_tensor_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_root = root / "daily"
            label_root = root / "labels"
            (daily_root / "2026").mkdir(parents=True)
            (label_root / "2026").mkdir(parents=True)
            for trade_date, offset in [(20260105, 0.0), (20260106, 1.0)]:
                pd.DataFrame(
                    {
                        "trade_date": [trade_date, trade_date],
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "open": [10.0 + offset, 20.0 + offset],
                        "high": [11.0 + offset, 21.0 + offset],
                        "low": [9.0 + offset, 19.0 + offset],
                        "close": [10.5 + offset, 20.5 + offset],
                        "vol": [100.0, 200.0],
                        "amount": [1000.0, 4000.0],
                    }
                ).to_parquet(daily_root / "2026" / f"{trade_date}.parquet", index=False)
            pd.DataFrame(
                {
                    "trade_date": [20260106, 20260106],
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "y": [1.0, 2.0],
                }
            ).to_parquet(label_root / "2026" / "20260106.parquet", index=False)

            config_path = root / "run.toml"
            config_path.write_text(
                f"""
run_id = "r1"
alpha_id = "a1"
data_root = "{root.as_posix()}"

[dates]
train = [20260105, 20260106]
valid = []
predict = []

[sample]
train_frequency = "daily"
predict_frequency = "daily"

[features]
type = "bar_panel"
root = "{daily_root.as_posix()}"
columns = ["open", "volume"]

[features.params]
source_frequency = "daily"
bar_size = 1
lookback_sessions = 2
time_series_scale = "none"
strict = true
max_cache_sessions = "auto"

[label]
id = "y"
root = "{label_root.as_posix()}"

[filters]
exclude_limit = false
exclude_st = false
exclude_bj = true

[preprocess]
cross_section_transform = "zscore"
feature_fill_value = 0.0

[model]
name = "bar_gru"
class = "yq_ml_alpha.models.bar_gru_model.BarGRUAlphaModel"
artifact_dir = "{(root / "artifacts").as_posix()}"
""",
                encoding="utf-8",
            )
            config = load_config(config_path)
            builder = DatasetBuilder(config)
            bundle = builder.load_bar_panel([20260106], True, TradingCalendar([20260105, 20260106]))
            self.assertEqual(list(bundle.frame.columns), ["trade_date", "ts_code", "y"])
            self.assertEqual(bundle.tensors["bar"].shape, (2, 2, 2))
            self.assertEqual(bundle.tensors["bar"].dtype, np.float32)
            self.assertEqual(float(bundle.tensors["bar"][0, 0, 0]), -1.0)
            self.assertEqual(float(bundle.tensors["bar"][1, 0, 0]), 1.0)
            self.assertEqual(bundle.frame["y"].tolist(), [-1.0, 1.0])

    def test_bar_panel_provider_aggregates_minute_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "minute"
            (root / "2026").mkdir(parents=True)
            rows = []
            times = ["2026-01-05 09:30:00"]
            for hour, start, end in [(9, 31, 60), (10, 0, 60), (11, 0, 31), (13, 1, 60), (14, 0, 60), (15, 0, 1)]:
                for minute in range(start, end):
                    times.append(f"2026-01-05 {hour:02d}:{minute:02d}:00")
            for symbol, offset in [("000001.SZ", 0.0), ("000002.SZ", 10.0)]:
                for idx, trade_time in enumerate(times):
                    price = 100.0 + offset + idx * 0.01
                    rows.append(
                        {
                            "ts_code": symbol,
                            "trade_time": trade_time,
                            "open": price,
                            "high": price + 0.2,
                            "low": price - 0.2,
                            "close": price + 0.1,
                            "vol": 100.0 + idx,
                            "amount": (100.5 + offset) * (100.0 + idx),
                        }
                    )
            pd.DataFrame(rows).to_parquet(root / "2026" / "20260105.parquet", index=False)

            provider = BarPanelProvider(
                root,
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "minute", "bar_size": 15, "lookback_sessions": 1},
            )
            daily = provider._load_minute_session(20260105)
            self.assertEqual(len(daily), 2)
            self.assertIn("vwap__b000", daily.columns)
            first = daily.loc[daily["ts_code"] == "000001.SZ"].iloc[0]
            self.assertAlmostEqual(float(first["vwap__b000"]), 100.5, places=7)
            self.assertAlmostEqual(float(first["open__b000"]), 100.01, places=7)

            window = provider.load_window(20260105, [20260105])
            self.assertEqual(window.shape, (2, 2 + 16 * 6))
            self.assertEqual(window["trade_date"].tolist(), [20260105, 20260105])
            self.assertTrue(np.isfinite(window[provider.feature_columns].to_numpy()).all())

    def test_bar_panel_provider_filters_bj_and_st_before_resample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "minute"
            (root / "2026").mkdir(parents=True)
            rows = []
            times = []
            for hour, start, end in [(9, 31, 60), (10, 0, 60), (11, 0, 31), (13, 1, 60), (14, 0, 60), (15, 0, 1)]:
                for minute in range(start, end):
                    times.append(f"2026-01-05 {hour:02d}:{minute:02d}:00")
            for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0), ("830001.BJ", 30.0)]:
                for idx, trade_time in enumerate(times):
                    price = base + idx * 0.01
                    rows.append(
                        {
                            "ts_code": symbol,
                            "trade_time": trade_time,
                            "open": price,
                            "high": price + 0.1,
                            "low": price - 0.1,
                            "close": price + 0.05,
                            "vol": 100.0,
                            "amount": price * 100.0,
                        }
                    )
            pd.DataFrame(rows).to_parquet(root / "2026" / "20260105.parquet", index=False)

            provider = BarPanelProvider(
                root,
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "minute", "bar_size": 15, "lookback_sessions": 1},
            )
            window = provider.load_window(
                20260105,
                [20260105],
                exclude_bj=True,
                st_symbols_by_date={20260105: {"000002.SZ"}},
            )
            self.assertEqual(window["ts_code"].tolist(), ["000001.SZ"])

    def test_bar_panel_rejects_too_large_minute_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "1 <= bar_size <= 120"):
                BarPanelProvider(
                    Path(tmp),
                    ["open", "high", "low", "close", "vwap", "volume"],
                    {"source_frequency": "minute", "bar_size": 121, "lookback_sessions": 1},
                )

    def test_bar_panel_accepts_derived_120_minute_bar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = BarPanelProvider(
                Path(tmp),
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "minute_bar", "bar_size": 120, "lookback_sessions": 1},
            )
            self.assertEqual(provider.steps_per_session, 2)

    def test_bar_panel_provider_reads_derived_minute_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "derived" / "stock" / "bar" / "15m"
            (root / "2026").mkdir(parents=True)
            pd.DataFrame(
                {
                    "trade_date": [20260105, 20260105, 20260105],
                    "trade_time": ["09:45:00", "10:00:00", "09:45:00"],
                    "bar_index": [0, 1, 0],
                    "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                    "open": [10.0, 11.0, 20.0],
                    "high": [10.5, 11.5, 20.5],
                    "low": [9.5, 10.5, 19.5],
                    "close": [10.2, 11.2, 20.2],
                    "volume": [100.0, 200.0, 100.0],
                    "amount": [1000.0, 2200.0, 2000.0],
                    "vwap": [10.0, 11.0, 20.0],
                    "minute_count": [15, 15, 15],
                }
            ).to_parquet(root / "2026" / "20260105.parquet", index=False)

            provider = BarPanelProvider(
                root,
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "minute_bar", "bar_size": 15, "lookback_sessions": 1, "strict": False},
            )
            window = provider.load_window(
                20260105,
                [20260105],
                st_symbols_by_date={20260105: {"000002.SZ"}},
            )
            self.assertEqual(window["ts_code"].tolist(), ["000001.SZ"])
            self.assertAlmostEqual(float(window.iloc[0]["open__t000"]), 10.0, places=7)
            self.assertAlmostEqual(float(window.iloc[0]["volume__t001"]), 200.0, places=7)

    def test_bar_panel_provider_auto_cache_and_tensor_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "derived" / "stock" / "bar" / "120m"
            (root / "2026").mkdir(parents=True)
            for trade_date, base in [(20260102, 10.0), (20260105, 20.0)]:
                rows = []
                for symbol, offset in [("000001.SZ", 0.0), ("000002.SZ", 100.0)]:
                    for bar_idx in [0, 1]:
                        if symbol == "000002.SZ" and trade_date == 20260105 and bar_idx == 1:
                            continue
                        price = base + offset + bar_idx
                        rows.append(
                            {
                                "trade_date": trade_date,
                                "trade_time": "11:30:00" if bar_idx == 0 else "15:00:00",
                                "bar_index": bar_idx,
                                "ts_code": symbol,
                                "open": price,
                                "high": price + 1.0,
                                "low": price - 1.0,
                                "close": price + 0.5,
                                "volume": 100.0 + bar_idx,
                                "amount": price * (100.0 + bar_idx),
                                "vwap": price,
                                "minute_count": 120,
                            }
                        )
                pd.DataFrame(rows).to_parquet(root / "2026" / f"{trade_date}.parquet", index=False)

            provider = BarPanelProvider(
                root,
                ["open", "volume"],
                {
                    "source_frequency": "minute_bar",
                    "bar_size": 120,
                    "lookback_sessions": 2,
                    "time_series_scale": "none",
                    "strict": True,
                    "max_cache_sessions": "auto",
                },
            )
            provider.set_cache_sessions_for_target_dates([20260105, 20260106], [20260102, 20260105, 20260106])
            self.assertEqual(provider.max_cache_sessions, 1)
            window = provider.load_window_tensor(20260105, [20260102, 20260105])
            self.assertEqual(window.frame["ts_code"].tolist(), ["000001.SZ"])
            self.assertEqual(window.tensors["bar"].shape, (1, 4, 2))
            self.assertEqual(window.tensors["bar"].dtype, np.float32)
            self.assertEqual(float(window.tensors["bar"][0, 0, 0]), 10.0)
            self.assertEqual(float(window.tensors["bar"][0, 3, 1]), 101.0)
            self.assertEqual(window.skipped_incomplete, 1)

            provider_20 = BarPanelProvider(
                root,
                ["open"],
                {"source_frequency": "minute_bar", "bar_size": 120, "lookback_sessions": 20, "max_cache_sessions": "auto"},
            )
            calendar_dates = list(range(20260101, 20260131))
            provider_20.set_cache_sessions_for_target_dates([20260106, 20260111], calendar_dates)
            self.assertEqual(provider_20.max_cache_sessions, 15)

    def test_bar_panel_provider_aggregates_daily_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "daily"
            (root / "2026").mkdir(parents=True)
            for offset, trade_date in enumerate([20260102, 20260105, 20260106, 20260107, 20260108]):
                pd.DataFrame(
                    {
                        "trade_date": [trade_date],
                        "ts_code": ["000001.SZ"],
                        "open": [10.0 + offset],
                        "high": [11.0 + offset],
                        "low": [9.0 + offset],
                        "close": [10.5 + offset],
                        "vol": [100.0 + offset],
                        "amount": [1000.0 + offset],
                    }
                ).to_parquet(root / "2026" / f"{trade_date}.parquet", index=False)
            provider = BarPanelProvider(
                root,
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "daily", "bar_size": 5, "lookback_sessions": 5, "time_series_scale": "none"},
            )
            window = provider.load_window(20260108, [20260102, 20260105, 20260106, 20260107, 20260108])
            self.assertEqual(window.shape, (1, 2 + 6))
            row = window.iloc[0]
            self.assertEqual(float(row["open__t000"]), 10.0)
            self.assertEqual(float(row["high__t000"]), 15.0)
            self.assertEqual(float(row["low__t000"]), 9.0)
            self.assertEqual(float(row["close__t000"]), 14.5)
            self.assertEqual(float(row["volume__t000"]), sum(100.0 + i for i in range(5)))
            expected_vwap = sum(1000.0 + i for i in range(5)) * 10.0 / sum(100.0 + i for i in range(5))
            self.assertAlmostEqual(float(row["vwap__t000"]), expected_vwap, places=7)

    def test_bar_panel_last_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "daily"
            (root / "2026").mkdir(parents=True)
            for trade_date, open_value in [(20260105, 10.0), (20260106, 20.0)]:
                pd.DataFrame(
                    {
                        "trade_date": [trade_date],
                        "ts_code": ["000001.SZ"],
                        "open": [open_value],
                        "high": [open_value + 1.0],
                        "low": [open_value - 1.0],
                        "close": [open_value + 0.5],
                        "vol": [100.0],
                        "amount": [1000.0],
                    }
                ).to_parquet(root / "2026" / f"{trade_date}.parquet", index=False)
            provider = BarPanelProvider(
                root,
                ["open", "high", "low", "close", "vwap", "volume"],
                {"source_frequency": "daily", "bar_size": 1, "lookback_sessions": 2, "time_series_scale": "last"},
            )
            window = provider.load_window(20260106, [20260105, 20260106])
            row = window.iloc[0]
            self.assertAlmostEqual(float(row["open__t000"]), 0.5, places=7)
            self.assertAlmostEqual(float(row["open__t001"]), 1.0, places=7)

    def test_multi_bar_panel_provider_merges_prefixed_panels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            daily_root = Path(tmp) / "daily"
            minute_root = Path(tmp) / "minute"
            (daily_root / "2026").mkdir(parents=True)
            (minute_root / "2026").mkdir(parents=True)
            for offset, trade_date in enumerate([20260102, 20260105]):
                pd.DataFrame(
                    {
                        "trade_date": [trade_date, trade_date],
                        "ts_code": ["000001.SZ", "000002.SZ"],
                        "open": [10.0 + offset, 20.0 + offset],
                        "high": [11.0 + offset, 21.0 + offset],
                        "low": [9.0 + offset, 19.0 + offset],
                        "close": [10.5 + offset, 20.5 + offset],
                        "vol": [100.0, 200.0],
                        "amount": [1000.0, 4000.0],
                    }
                ).to_parquet(daily_root / "2026" / f"{trade_date}.parquet", index=False)

            times = []
            for hour, start, end in [(9, 31, 60), (10, 0, 60), (11, 0, 31), (13, 1, 60), (14, 0, 60), (15, 0, 1)]:
                for minute in range(start, end):
                    times.append(f"2026-01-05 {hour:02d}:{minute:02d}:00")
            rows = []
            for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 20.0)]:
                for idx, trade_time in enumerate(times):
                    price = base + idx * 0.01
                    rows.append(
                        {
                            "ts_code": symbol,
                            "trade_time": trade_time,
                            "open": price,
                            "high": price + 0.1,
                            "low": price - 0.1,
                            "close": price + 0.05,
                            "vol": 100.0,
                            "amount": price * 100.0,
                        }
                    )
            pd.DataFrame(rows).to_parquet(minute_root / "2026" / "20260105.parquet", index=False)

            provider = MultiBarPanelProvider(
                {
                    "panels": {
                        "daily": {
                            "root": daily_root,
                            "source_frequency": "daily",
                            "bar_size": 1,
                            "lookback_sessions": 2,
                            "time_series_scale": "last",
                            "columns": ["open", "high", "low", "close", "vwap", "volume"],
                        },
                        "minute": {
                            "root": minute_root,
                            "source_frequency": "minute",
                            "bar_size": 60,
                            "lookback_sessions": 1,
                            "time_series_scale": "mean",
                            "columns": ["open", "high", "low", "close", "vwap", "volume"],
                        },
                    }
                }
            )
            window = provider.load_window(20260105, [20260102, 20260105])
            self.assertEqual(len(window), 2)
            self.assertIn("daily__open__t000", provider.feature_columns)
            self.assertIn("minute__open__t004", provider.feature_columns)
            self.assertEqual(len(provider.feature_columns), 2 * 6 + 5 * 6)
            self.assertTrue(np.isfinite(window[provider.feature_columns].to_numpy()).all())

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
        _skip_unless_sklearn_ready(self)
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
        _skip_unless_sklearn_ready(self)
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
        _skip_unless_sklearn_ready(self)
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
        _skip_unless_xgboost_sklearn_ready(self)
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
        except ImportError:
            self.skipTest("optuna is not installed")
        _skip_unless_xgboost_sklearn_ready(self)
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
            backtest_root = Path(tmp) / "backtest"
            for factor in ("pos", "neg", "nan", "zero"):
                (backtest_root / factor).mkdir(parents=True)
            pd.DataFrame({"rank_ic": [0.1, 0.2, np.nan]}).to_parquet(
                backtest_root / "pos" / "ic.parquet", index=False
            )
            pd.DataFrame({"rank_ic": [-0.1, -0.3]}).to_parquet(
                backtest_root / "neg" / "ic.parquet", index=False
            )
            pd.DataFrame({"rank_ic": [np.nan, np.nan]}).to_parquet(
                backtest_root / "nan" / "ic.parquet", index=False
            )
            pd.DataFrame({"rank_ic": [0.1, -0.1]}).to_parquet(
                backtest_root / "zero" / "ic.parquet", index=False
            )
            context = ModelContext(
                run_id="r",
                alpha_id="a",
                feature_columns=["pos", "neg", "missing", "nan", "zero"],
                label_column="y",
                artifact_dir=Path("tmp"),
                model_params={"backtest_root": str(backtest_root), "ic_metric": "rank_ic"},
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
            backtest_root = Path(tmp) / "backtest"
            (backtest_root / "a").mkdir(parents=True)
            pd.DataFrame({"rank_ic": [np.nan]}).to_parquet(backtest_root / "a" / "ic.parquet", index=False)
            context = ModelContext(
                run_id="r",
                alpha_id="a",
                feature_columns=["a", "b"],
                label_column="y",
                artifact_dir=Path("tmp"),
                model_params={"backtest_root": str(backtest_root), "ic_metric": "rank_ic"},
                model_search={},
            )
            with self.assertRaisesRegex(ValueError, "no valid IC signs"):
                ICSignEqualWeightAlphaModel().fit(pd.DataFrame(), pd.DataFrame(), context)

    def test_ic_sign_equal_weight_rejects_legacy_ic_root(self) -> None:
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=["a"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={"ic_root": "data/backtest/stock/daily/ic"},
            model_search={},
        )
        with self.assertRaisesRegex(ValueError, "ic_root is removed"):
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

    def test_logsig_orthogonal_mlp_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        rows = []
        for trade_date, offset in [(1, 0.0), (2, 0.2)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": symbol,
                        "logsig_0001": offset + float(idx),
                        "logsig_0002": offset + float(3 - idx),
                        "logsig_0003": offset + float(idx % 2),
                        "y": float(idx),
                    }
                )
        train = pd.DataFrame(rows)
        context = ModelContext(
            run_id="logsig_alpha_v",
            alpha_id="logsig_alpha_v",
            feature_columns=["logsig_0001", "logsig_0002", "logsig_0003"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "hidden_layers": [6],
                "base_factors": 3,
                "orthogonal_lambda": 0.05,
                "epochs": 2,
                "batch_size": 4,
                "patience": 0,
                "seed": 7,
                "device": "cpu",
            },
            model_search={},
        )
        model = LogsigOrthogonalMLPAlphaModel()
        model.fit(train, pd.DataFrame(), context)
        self.assertEqual(model.model_info["base_factors"], 3)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = LogsigOrthogonalMLPAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-6))

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
            self.assertEqual(model.params["loss"], "mse")
            self.assertEqual(model.model_info["loss"], "mse")
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

    def test_sequence_pearson_ic_loss_matches_bar_gru_semantics_when_installed(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        self.assertTrue(
            torch.allclose(
                _negative_ic_loss(torch, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])),
                torch.tensor(-1.0),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                _negative_ic_loss(torch, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([30.0, 20.0, 10.0])),
                torch.tensor(1.0),
                atol=1e-6,
            )
        )
        self.assertIsNone(_negative_ic_loss(torch, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 2.0])))

    def test_sequence_models_pearson_ic_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        rows = []
        for trade_date, base in [(1, 0.0), (2, 1.0)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": symbol,
                        "x1": base + idx * 0.1,
                        "x2": 1.0 + idx * 0.2,
                        "x3": 2.0 - idx * 0.1,
                        "x4": 0.5 + idx * 0.3,
                        "y": float(idx),
                    }
                )
        train = pd.DataFrame(rows)
        base_context = dict(
            run_id="r",
            alpha_id="a",
            feature_columns=["x1", "x2", "x3", "x4"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "sequence_length": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 1,
                "batch_size": 2,
                "patience": 0,
                "seed": 7,
                "device": "cpu",
                "loss": "pearson_ic",
            },
            model_search={},
        )
        for model_cls in [RNNAlphaModel, LSTMAlphaModel, GRUAlphaModel]:
            model = model_cls()
            context = ModelContext(**base_context)
            model.fit(train, pd.DataFrame(), context)
            self.assertEqual(model.params["loss"], "pearson_ic")
            self.assertEqual(model.model_info["loss"], "pearson_ic")
            self.assertGreater(len(model.loss_history), 0)
            self.assertTrue(np.isfinite(model.loss_history[-1]["train_loss"]))
            pred = model.predict(train, context)
            self.assertEqual(len(pred), len(train))
            self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_bar_gru_model_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        feature_columns = [f"f{feature}__t{step:03d}" for step in range(4) for feature in range(2)]
        rows = []
        for trade_date, base in [(1, 0.0), (2, 1.0)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                row = {"trade_date": trade_date, "ts_code": symbol, "y": float(idx)}
                for col_idx, column in enumerate(feature_columns):
                    row[column] = base + idx * 0.1 + col_idx * 0.01
                rows.append(row)
        train = pd.DataFrame(rows)
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=feature_columns,
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "sequence_length": 4,
                "input_size": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 10,
                "patience": 0,
                "seed": 7,
                "device": "cpu",
            },
            model_search={},
        )
        model = BarGRUAlphaModel()
        model.fit(train, pd.DataFrame(), context)
        self.assertGreater(len(model.loss_history), 0)
        self.assertEqual(model.model_info["sequence_length"], 4)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = BarGRUAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-6))

    def test_bar_gru_model_bundle_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        feature_columns = [f"f{feature}__t{step:03d}" for step in range(4) for feature in range(2)]
        rows = []
        for trade_date, base in [(1, 0.0), (2, 1.0)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                row = {"trade_date": trade_date, "ts_code": symbol, "y": float(idx)}
                for col_idx, column in enumerate(feature_columns):
                    row[column] = base + idx * 0.1 + col_idx * 0.01
                rows.append(row)
        train = pd.DataFrame(rows)
        tensor = train[feature_columns].to_numpy(dtype="float32").reshape(len(train), 4, 2)
        bundle = DatasetBundle(
            train[["trade_date", "ts_code", "y"]].copy().reset_index(drop=True),
            feature_columns,
            "y",
            tensors={"bar": tensor},
            tensor_columns={"bar": feature_columns},
        )
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=feature_columns,
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "sequence_length": 4,
                "input_size": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 10,
                "patience": 0,
                "seed": 7,
                "device": "cpu",
            },
            model_search={},
        )
        model = BarGRUAlphaModel()
        empty = DatasetBundle(
            pd.DataFrame(columns=["trade_date", "ts_code", "y"]),
            feature_columns,
            "y",
            tensors={"bar": np.empty((0, 4, 2), dtype="float32")},
            tensor_columns={"bar": feature_columns},
        )
        model.fit_bundle(bundle, empty, context)
        pred = model.predict_bundle(bundle, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())

    def test_multi_bar_gru_model_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        daily_columns = [f"daily__f{feature}__t{step:03d}" for step in range(2) for feature in range(2)]
        minute_columns = [f"minute__f{feature}__t{step:03d}" for step in range(3) for feature in range(2)]
        feature_columns = [*daily_columns, *minute_columns]
        rows = []
        for trade_date, base in [(1, 0.0), (2, 1.0)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                row = {"trade_date": trade_date, "ts_code": symbol, "y": float(idx)}
                for col_idx, column in enumerate(feature_columns):
                    row[column] = base + idx * 0.1 + col_idx * 0.01
                rows.append(row)
        train = pd.DataFrame(rows)
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=feature_columns,
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "daily_sequence_length": 2,
                "minute_sequence_length": 3,
                "input_size": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "epochs": 2,
                "batch_size": 10,
                "patience": 0,
                "seed": 7,
                "device": "cpu",
            },
            model_search={},
        )
        model = MultiBarGRUAlphaModel()
        model.fit(train, pd.DataFrame(), context)
        self.assertGreater(len(model.loss_history), 0)
        self.assertEqual(model.model_info["daily_sequence_length"], 2)
        self.assertEqual(model.model_info["minute_sequence_length"], 3)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = MultiBarGRUAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-6))

    def test_residual_multi_bar_gru_model_smoke_when_installed(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed")
        daily_columns = [f"daily__f{feature}__t{step:03d}" for step in range(2) for feature in range(2)]
        minute_columns = [f"minute__f{feature}__t{step:03d}" for step in range(3) for feature in range(2)]
        feature_columns = [*daily_columns, *minute_columns]
        rows = []
        for trade_date, base in [(1, 0.0), (2, 1.0)]:
            for idx, symbol in enumerate(["a", "b", "c", "d"]):
                row = {"trade_date": trade_date, "ts_code": symbol, "y": float(idx)}
                for col_idx, column in enumerate(feature_columns):
                    row[column] = base + idx * 0.1 + col_idx * 0.01
                rows.append(row)
        train = pd.DataFrame(rows)
        context = ModelContext(
            run_id="r",
            alpha_id="a",
            feature_columns=feature_columns,
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={
                "daily_sequence_length": 2,
                "minute_sequence_length": 3,
                "input_size": 2,
                "hidden_size": 4,
                "num_layers": 1,
                "stage1_epochs": 2,
                "stage2_epochs": 2,
                "stage1_patience": 0,
                "stage2_patience": 0,
                "batch_size": 10,
                "seed": 7,
                "device": "cpu",
            },
            model_search={},
        )
        model = ResidualMultiBarGRUAlphaModel()
        model.fit(train, pd.DataFrame(), context)
        self.assertIn("stage1_daily", {row["stage"] for row in model.loss_history})
        self.assertIn("stage2_residual", {row["stage"] for row in model.loss_history})
        self.assertIn("stage1_best_loss", model.model_info)
        self.assertIn("stage2_best_loss", model.model_info)
        self.assertTrue(
            all(not parameter.requires_grad for name, parameter in model.model.named_parameters() if name.startswith("daily_"))
        )
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model.save(path)
            loaded = ResidualMultiBarGRUAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-6))

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

    def test_pca_ols_model_fit_predict_save_load(self) -> None:
        _skip_unless_sklearn_ready(self)
        context = ModelContext(
            run_id="mdl_000005",
            alpha_id="mdl_000005",
            feature_columns=["x1", "x2", "x3", "x4"],
            label_column="y",
            artifact_dir=Path("tmp"),
            model_params={"explained_variance": 0.95},
            model_search={},
            diagnostics={
                "enabled": True,
                "write_model_info": True,
                "write_window_summary": True,
            },
        )
        train = pd.DataFrame(
            {
                "trade_date": [1, 1, 1, 1, 2, 2, 2, 2],
                "ts_code": ["a", "b", "c", "d", "a", "b", "c", "d"],
                "x1": [1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1],
                "x2": [4.0, 3.0, 2.0, 1.0, 3.9, 2.9, 1.9, 0.9],
                "x3": [0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3],
                "x4": [0.1, 0.2, 0.3, 0.4, 0.0, 0.1, 0.2, 0.3],
                "y": [0.1, 0.2, 0.3, 0.4, 0.15, 0.25, 0.35, 0.45],
            }
        )
        model = PCAOLSAlphaModel()
        model.fit(train, pd.DataFrame(), context)
        pred = model.predict(train, context)
        self.assertEqual(len(pred), len(train))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        self.assertLessEqual(model.n_components_, model.n_original_features_)
        self.assertGreaterEqual(model.explained_variance_ratio_sum_, 0.95)
        with tempfile.TemporaryDirectory() as tmp:
            context.artifact_dir = Path(tmp) / "window"
            written = model.write_diagnostics(context)
            self.assertTrue(any(path.name == "model_info.json" for path in written))
            path = Path(tmp) / "model.pkl"
            model.save(path)
            loaded = PCAOLSAlphaModel.load(path)
            loaded_pred = loaded.predict(train, context)
            self.assertTrue(np.allclose(pred.to_numpy(), loaded_pred.to_numpy(), atol=1e-7))

    def test_monthly_ic_sign_config_parses(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "models" / "monthly_ic_sign_equal_weight.toml")
        self.assertEqual(config.alpha_id, "ml_alpha_ic_sign_ew")
        self.assertEqual(config.features.columns, "__all__")
        self.assertEqual(config.model.class_path, "yq_ml_alpha.models.ic_sign_model.ICSignEqualWeightAlphaModel")
        self.assertEqual(config.model.params["ic_metric"], "rank_ic")
        self.assertFalse(config.diagnostics.enabled)

    def test_new_model_configs_parse(self) -> None:
        model_dir = Path(__file__).resolve().parents[1] / "models"
        expected = {
            model_dir / "mdl_000001.toml": ("mdl_000001", "LinearRegressionAlphaModel"),
            model_dir / "mdl_000002.toml": ("mdl_000002", "LassoAlphaModel"),
            model_dir / "mdl_000003.toml": ("mdl_000003", "RidgeAlphaModel"),
            model_dir / "mdl_000004.toml": ("mdl_000004", "ElasticNetAlphaModel"),
            model_dir / "mdl_000005.toml": ("mdl_000005", "PCAOLSAlphaModel"),
            model_dir / "mdl_000006.toml": ("mdl_000006", "LSTMAlphaModel"),
            model_dir / "monthly_rnn_36.toml": ("ml_alpha_rnn", "RNNAlphaModel"),
            model_dir / "monthly_gru_36.toml": ("ml_alpha_gru", "GRUAlphaModel"),
            model_dir / "monthly_elstm_ranknet_36.toml": ("ml_alpha_elstm_ranknet", "eLSTMRankNetAlphaModel"),
            model_dir / "monthly_cnn_36.toml": ("ml_alpha_cnn", "CNNAlphaModel"),
            model_dir / "monthly_xgb_optuna_36.toml": ("ml_alpha_xgb_optuna", "XGBoostOptunaAlphaModel"),
            model_dir / "monthly_lgbm_optuna_36.toml": ("ml_alpha_lgbm_optuna", "LightGBMOptunaAlphaModel"),
        }
        for path, (alpha_id, class_name) in expected.items():
            filename = path.name
            config = load_config(path)
            self.assertEqual(config.alpha_id, alpha_id)
            self.assertTrue(config.model.class_path.endswith(class_name))
            self.assertEqual(config.features.columns, "__all__")
            if filename == "mdl_000006.toml":
                self.assertEqual(config.label.id, "future_vwap_return_5d")
                self.assertEqual(config.sample.train_frequency, "20")
                self.assertEqual(config.sample.predict_frequency, "daily")
                self.assertEqual(config.train_scheme.refit_frequency, "semiannual_end")
                self.assertEqual(config.train_scheme.train_lookback, "3y")
                self.assertEqual(config.train_scheme.validation_ratio, 0.2)
                self.assertEqual(config.train_scheme.validation_sample_count, 0)
                self.assertEqual(config.model.params["sequence_length"], 5)
                self.assertEqual(config.model.params["sequence_frequency"], "daily")
                self.assertEqual(config.model.params["hidden_size"], 64)
                self.assertEqual(config.model.params["num_layers"], 1)
                self.assertEqual(config.model.params["dropout"], 0.0)
                self.assertEqual(config.model.params["loss"], "pearson_ic")
            elif filename in {
                "mdl_000002.toml",
                "mdl_000003.toml",
                "mdl_000004.toml",
                "monthly_xgb_optuna_36.toml",
                "monthly_lgbm_optuna_36.toml",
                "monthly_rnn_36.toml",
                "monthly_gru_36.toml",
                "monthly_elstm_ranknet_36.toml",
                "monthly_cnn_36.toml",
            }:
                self.assertEqual(config.train_scheme.validation_sample_count, 1)
            else:
                self.assertEqual(config.train_scheme.validation_sample_count, 0)
                self.assertIsNone(config.train_scheme.validation_ratio)
                self.assertIsNone(config.train_scheme.train_lookback)
            if class_name in {"RNNAlphaModel", "GRUAlphaModel", "eLSTMRankNetAlphaModel"}:
                self.assertEqual(config.model.params["sequence_length"], 6)
            if class_name == "eLSTMRankNetAlphaModel":
                self.assertEqual(config.model.params["max_pairs_per_date"], 20000)
                self.assertEqual(config.model.params["sigma"], 1.0)
            if class_name == "PCAOLSAlphaModel":
                self.assertEqual(config.model.params["explained_variance"], 0.95)
                self.assertTrue(config.diagnostics.enabled)
                self.assertTrue(config.diagnostics.write_model_info)
                self.assertTrue(config.diagnostics.write_window_summary)

    def test_mdl_end_to_end_configs_migrated_to_factors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "models" / "mdl_000006.toml").exists())
        for name in ["mdl_000007.toml", "mdl_000008.toml"]:
            self.assertFalse((root / "models" / name).exists())
            self.assertFalse((root / "configs" / name).exists())
        for name in ["e2e_fct_000001.toml", "e2e_fct_000002.toml", "e2e_fct_000003.toml"]:
            self.assertFalse((root / "factors" / name).exists())
        for name in ["bar_gru_15m.toml", "multi_bar_gru_daily_15m.toml", "residual_multi_bar_gru.toml"]:
            self.assertTrue((root / "factors" / name).exists())
        registry = (root / "model_registry.toml").read_text(encoding="utf-8")
        self.assertIn("mdl_000006", registry)
        self.assertNotIn("mdl_000007", registry)
        self.assertNotIn("mdl_000008", registry)
        factor_registry = (root / "factor_registry.toml").read_text(encoding="utf-8")
        self.assertNotIn("e2e_fct_", factor_registry)
        self.assertIn("logsig_alpha_v", factor_registry)

    def test_tuned_configs_expose_search_space(self) -> None:
        model_dir = Path(__file__).resolve().parents[1] / "models"
        lasso = load_config(model_dir / "mdl_000002.toml")
        self.assertIn("alpha", lasso.model.search["space"])

        xgb = load_config(model_dir / "monthly_xgb_optuna_36.toml")
        self.assertIn("space", xgb.model.search)
        self.assertEqual(xgb.model.search["space"]["n_estimators"]["type"], "int")
        self.assertTrue(xgb.model.search["space"]["learning_rate"]["log"])

        lgbm = load_config(model_dir / "monthly_lgbm_optuna_36.toml")
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
