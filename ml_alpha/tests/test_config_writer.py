from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.output.alpha_writer import AlphaWriter


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


if __name__ == "__main__":
    unittest.main()
