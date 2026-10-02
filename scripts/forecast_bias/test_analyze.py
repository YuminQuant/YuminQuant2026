import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze import attach_industry, attach_size, load_actuals, score_forecasts, main


class ForecastBiasTests(unittest.TestCase):
    def test_error_loss_zero_and_disclosure_boundary(self):
        forecast = pd.DataFrame({"ts_code": ["A", "B", "C", "D"], "forecast_year": [2024] * 4,
                                 "report_date": pd.to_datetime(["2025-01-01"] * 4), "forecast": [-50., 1., 4., 1.]})
        actual = pd.DataFrame({"ts_code": ["A", "B", "C"], "forecast_year": [2024] * 3,
                              "actual_date": pd.to_datetime(["2025-03-01", "2025-03-01", "2025-01-01"]),
                              "actual": [-100., 0., 2.]})
        audit = {}
        result = score_forecasts(forecast, actual, 0, audit)
        self.assertEqual(result.ts_code.tolist(), ["A"])
        self.assertEqual(result.bias.tolist(), [.5])
        self.assertEqual(result.signed_error.tolist(), [.5])
        self.assertEqual(result.horizon_months.tolist(), [2])
        self.assertEqual(audit["excluded_missing_or_nonfinite_actual"], 1)
        self.assertEqual(audit["excluded_small_actual_denominator"], 1)
        self.assertEqual(audit["excluded_on_or_after_actual_disclosure"], 1)

    def test_first_disclosure_not_restated_and_units(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"ts_code": ["A", "A", "B", "B"], "end_date": [20241231] * 4,
                          "ann_date": [20250301, 20260301, 20250301, 20260301],
                          "f_ann_date": [20250302, 20260301, 20250301, 20260301],
                          "report_type": [1] * 4, "update_flag": [0, 1, 0, 1],
                          "n_income_attr_p": [100000., 999999., np.nan, 200000.]}).to_parquet(root / "2026.parquet")
            actual = load_actuals(root, {2024}, SimpleNamespace(metric="np", as_of=pd.Timestamp("2026-04-01")), {})
            self.assertEqual(actual.loc[actual.ts_code == "A", "actual"].iloc[0], 10.)
            self.assertTrue(pd.isna(actual.loc[actual.ts_code == "B", "actual"].iloc[0]))
            self.assertEqual(actual.loc[actual.ts_code == "A", "actual_date"].iloc[0], pd.Timestamp("2025-03-02"))

    def test_size_uses_full_market_and_no_future(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({"ts_code": ["A", "B", "C", "D"], "trade_date": [20250307] * 4,
                          "total_mv": [1., 2., 3., 4.]}).to_parquet(root / "20250307.parquet")
            df = pd.DataFrame({"ts_code": ["B", "B", "A"],
                               "report_date": pd.to_datetime(["2025-03-09", "2025-03-09", "2025-03-06"])})
            result = attach_size(df, root, 4, 7)
            self.assertEqual(result.size_group.iloc[:2].tolist(), [2., 2.])
            self.assertTrue(pd.isna(result.size_group.iloc[2]))

    def test_historical_industry_boundary_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "members.parquet"
            pd.DataFrame({"ts_code": ["A", "A", "B", "B"], "l1_code": ["OLD", "NEW", "X", "Y"],
                          "l1_name": ["old", "new", "x", "y"], "in_date": [20200101, 20250301, 20200101, 20200101],
                          "out_date": [20250301, None, None, None]}).to_parquet(path)
            df = pd.DataFrame({"ts_code": ["A", "A", "B"],
                               "report_date": pd.to_datetime(["2025-02-28", "2025-03-01", "2025-03-01"])})
            result = attach_industry(df, path)
            self.assertEqual(result.industry_code.iloc[:2].tolist(), ["OLD", "NEW"])
            self.assertTrue(pd.isna(result.industry_code.iloc[2]))

    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ["stock_data/analyst_report", "stock_data/income", "stock_data/daily/basic", "index_data/member_ci"]:
                (root / name).mkdir(parents=True)
            pd.DataFrame({"ts_code": ["A"], "report_date": [20250102], "quarter": ["2024Q4"],
                          "org_name": ["org"], "author_name": ["author"], "report_title": ["title"],
                          "create_time": ["2025-01-03 20:00:00"], "np": [12.]}).to_parquet(root / "stock_data/analyst_report/2025.parquet")
            pd.DataFrame({"ts_code": ["A"], "end_date": [20241231], "ann_date": [20250301], "f_ann_date": [None],
                          "report_type": [1], "update_flag": [0], "n_income_attr_p": [100000.]}).to_parquet(root / "stock_data/income/2025.parquet")
            pd.DataFrame({"ts_code": ["A"], "trade_date": [20250102], "total_mv": [100.]}).to_parquet(root / "stock_data/daily/basic/20250102.parquet")
            pd.DataFrame({"ts_code": ["A"], "l1_code": ["CI1"], "l1_name": ["Industry"],
                          "in_date": [20200101], "out_date": [None]}).to_parquet(root / "index_data/member_ci/ci_members.parquet")
            output = root / "results"
            main(["--start-date", "20250101", "--end-date", "20250131", "--as-of", "20250401",
                  "--data-root", str(root), "--config", str(root / "absent.toml"), "--output-dir", str(output),
                  "--min-count", "1", "--save-observations"])
            self.assertEqual(pd.read_csv(output / "bias_by_time.csv").median_bias.iloc[0], .2)
            self.assertEqual(len(list(output.glob("*.csv"))), 6)
            self.assertTrue((output / "bias_comparison.png").exists())
            self.assertTrue((output / "run_metadata.json").exists())
            self.assertTrue((output / "observations.parquet").exists())


if __name__ == "__main__":
    unittest.main()
