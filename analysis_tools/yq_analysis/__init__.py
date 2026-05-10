"""Analysis helpers for backtest and strategy result series."""

from yq_analysis.io import load_backtest_result
from yq_analysis.report import (
    make_backtest_report,
    make_factor_stats_report,
    make_ic_report,
    make_return_report,
    make_return_report_by_year,
)

__all__ = [
    "load_backtest_result",
    "make_backtest_report",
    "make_factor_stats_report",
    "make_ic_report",
    "make_return_report",
    "make_return_report_by_year",
]
