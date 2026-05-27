from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yq_analysis.metrics import annual_return, cumulative_return, max_drawdown, sharpe
from yq_analysis.report import make_backtest_report, make_ic_decay_report, make_return_report_by_year


def test_return_metrics_match_small_sample() -> None:
    returns = pd.Series([0.1, -0.05, 0.02])
    assert math.isclose(cumulative_return(returns), (1.1 * 0.95 * 1.02) - 1.0)
    assert math.isclose(annual_return(returns, periods_per_year=3), cumulative_return(returns))
    assert max_drawdown(returns) < 0.0
    assert np.isfinite(sharpe(returns, periods_per_year=3))


def test_metrics_ignore_nan_and_inf() -> None:
    returns = pd.Series([0.01, np.nan, np.inf, -0.02])
    assert math.isclose(cumulative_return(returns), (1.01 * 0.98) - 1.0)


def test_yearly_report_contains_cumulative_and_annual_return() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [20250102, 20250103, 20260102],
            "portfolio": ["group_1", "group_1", "group_1"],
            "return": [0.01, 0.02, -0.01],
        }
    )
    report = make_return_report_by_year(frame, periods_per_year=240)
    assert {"cumulative_return(%)", "annual_return(%)", "sharpe", "max_drawdown(%)"}.issubset(report.columns)
    assert report["year"].tolist() == [2025, 2026]


def test_backtest_report_accepts_current_schema() -> None:
    returns = pd.DataFrame(
        {
            "trade_date": [20250102, 20250102, 20250103, 20250103],
            "portfolio": ["group_1", "long_short", "group_1", "long_short"],
            "return": [0.01, 0.02, -0.01, 0.01],
            "excess_return": [0.005, np.nan, -0.002, np.nan],
            "turnover": [0.2, 0.5, np.nan, np.nan],
        }
    )
    ic = pd.DataFrame({"ic": [0.1, -0.2], "rank_ic": [0.05, 0.01]})
    factor_stats = pd.DataFrame({"factor_id": ["x", "x"], "coverage": [0.8, 0.9], "inf_rate": [0.0, 0.01]})
    report = make_backtest_report(returns, ic, factor_stats)
    assert set(report) == {
        "portfolio_total",
        "portfolio_by_year",
        "excess_total",
        "excess_by_year",
        "ic",
        "factor_stats",
    }
    assert not report["portfolio_total"].empty
    assert not report["excess_total"].empty
    assert "long_short" not in set(report["excess_total"]["portfolio"])
    assert "sortino" not in report["portfolio_total"].columns
    assert "std_return" not in report["portfolio_total"].columns
    assert "mean_return_bp_per_1pct_turnover" in report["portfolio_total"].columns
    assert "turnover_mean(%)" in report["portfolio_total"].columns
    assert not report["ic"].empty
    assert "coverage_mean" in report["factor_stats"].columns


def test_ic_decay_report_adds_approximate_multi_day_ic() -> None:
    ic = pd.DataFrame(
        {
            "horizon": list(range(1, 21)),
            "ic": [0.01] * 20,
        }
    )
    report = make_ic_decay_report(ic)
    decay = report[report["metric"] == "ic_mean"]
    approx_5d = report.loc[report["metric"] == "approx_5d_ic", "value"].iloc[0]
    approx_20d = report.loc[report["metric"] == "approx_20d_ic", "value"].iloc[0]
    assert len(decay) == 20
    assert math.isclose(approx_5d, 0.05 / math.sqrt(5))
    assert math.isclose(approx_20d, 0.20 / math.sqrt(20))
