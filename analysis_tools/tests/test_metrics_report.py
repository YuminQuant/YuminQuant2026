from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_all_missing_returns_keep_queryable_portfolios_without_fake_performance() -> None:
    returns = pd.DataFrame({
        "trade_date": [20260105, 20260105, 20260105],
        "portfolio": ["group_1", "group_10", "long_short"],
        "return": [np.nan, np.nan, np.inf],
        "excess_return": [np.nan, np.nan, np.nan],
    })
    with pytest.warns(RuntimeWarning, match="No finite portfolio returns"):
        report = make_backtest_report(returns)
    selected = report["excess_total"].query("portfolio == 'group_10' or portfolio == 'group_1'")
    assert len(selected) == 2
    assert selected["observations"].eq(0).all()
    assert selected["annual_return(%)"].isna().all()
    assert report["portfolio_total"]["annual_return(%)"].isna().all()


@pytest.mark.parametrize("returns", [None, pd.DataFrame(), pd.DataFrame({
    "trade_date": [20260105], "portfolio": ["group_1"], "return": [0.01],
})])
def test_unavailable_excess_report_preserves_query_schema(returns) -> None:
    report = make_backtest_report(returns)
    for name in ("excess_total", "excess_by_year"):
        assert report[name].query("portfolio == 'group_1'").empty
        assert "annual_return(%)" in report[name].columns
