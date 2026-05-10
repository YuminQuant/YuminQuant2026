from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from yq_analysis.metrics import (
    annual_return,
    annual_volatility,
    calmar,
    clean_series,
    cumulative_return,
    ic_abs_ir,
    ic_abs_mean,
    ic_mean,
    icir,
    max_drawdown,
    mean_return,
    observation_count,
    sharpe,
    std_return,
    win_rate,
)


def _portfolio_sort_key(name: Any) -> tuple[int, int | str]:
    text = str(name)
    match = re.fullmatch(r"group_(\d+)", text)
    if match:
        return (0, int(match.group(1)))
    if text == "long_short":
        return (1, text)
    return (2, text)


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ("trade_date", "factor_date", "date"):
        if column in frame.columns:
            return column
    return None


def _year_from_dates(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return (numeric // 10000).astype("Int64")


def _return_metrics(
    values: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
    turnover_values: pd.Series | None = None,
) -> dict[str, float | int]:
    mean = mean_return(values)
    turnover = clean_series(turnover_values).mean() if turnover_values is not None else np.nan
    if not np.isfinite(turnover) or abs(turnover) <= np.finfo(float).eps:
        bp_per_1pct_turnover = np.nan
    else:
        bp_per_1pct_turnover = mean * 10000.0 / (turnover * 100.0)
    return {
        "observations": observation_count(values),
        "cumulative_return(%)": _round_percent(cumulative_return(values)),
        "annual_return(%)": _round_percent(annual_return(values, periods_per_year=periods_per_year)),
        "annual_volatility(%)": _round_percent(
            annual_volatility(values, periods_per_year=periods_per_year)
        ),
        "sharpe": _round_float(
            sharpe(
                values,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
            ),
            3,
        ),
        "max_drawdown(%)": _round_percent(max_drawdown(values)),
        "calmar": _round_float(calmar(values, periods_per_year=periods_per_year), 3),
        "win_rate(%)": _round_percent(win_rate(values)),
        "mean_return_bp_per_1pct_turnover": _round_float(bp_per_1pct_turnover, 3),
        "turnover_mean(%)": _round_percent(turnover),
    }


def _round_float(value: float, digits: int) -> float:
    if not np.isfinite(value):
        return float("nan")
    return round(float(value), digits)


def _round_percent(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return round(float(value) * 100.0, 2)


def _portfolio_metrics(
    values: pd.Series,
    turnover_values: pd.Series | None,
    periods_per_year: int,
    risk_free_rate: float,
) -> dict[str, float | int]:
    return _return_metrics(values, periods_per_year, risk_free_rate, turnover_values)


def _series_metrics(
    values: pd.Series,
    periods_per_year: int,
    risk_free_rate: float,
) -> dict[str, float | int]:
    raw = _return_metrics(values, periods_per_year, risk_free_rate, None)
    raw.pop("mean_return_bp_per_1pct_turnover", None)
    raw.pop("turnover_mean(%)", None)
    return raw


def _has_valid_column(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and clean_series(frame[column]).shape[0] > 0


def make_return_report(
    returns: pd.DataFrame | pd.Series,
    return_col: str = "return",
    periods_per_year: int = 240,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Create long-run performance metrics by portfolio.

    If a plain Series is supplied, it is treated as one portfolio named `series`.
    """

    if isinstance(returns, pd.Series):
        rows = [{"portfolio": returns.name or "series", **_series_metrics(returns, periods_per_year, risk_free_rate)}]
        return pd.DataFrame(rows)
    if returns is None or returns.empty:
        return pd.DataFrame()

    if "portfolio" not in returns.columns:
        rows = [{"portfolio": "series", **_series_metrics(returns[return_col], periods_per_year, risk_free_rate)}]
        return pd.DataFrame(rows)

    rows = []
    for portfolio, group in returns.groupby("portfolio", sort=False):
        row = {
            "portfolio": portfolio,
            **_portfolio_metrics(
                group[return_col],
                group["turnover"] if "turnover" in group.columns else None,
                periods_per_year,
                risk_free_rate,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("portfolio", key=lambda s: s.map(_portfolio_sort_key)).reset_index(drop=True)


def make_return_report_by_year(
    returns: pd.DataFrame,
    return_col: str = "return",
    periods_per_year: int = 240,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    if returns is None or returns.empty:
        return pd.DataFrame()
    date_col = _date_column(returns)
    if date_col is None:
        raise ValueError("returns must contain trade_date, factor_date, or date for yearly report")

    frame = returns.copy()
    frame["year"] = _year_from_dates(frame[date_col])
    frame = frame.dropna(subset=["year"])
    group_cols = ["year"]
    if "portfolio" in frame.columns:
        group_cols.insert(0, "portfolio")

    rows = []
    for keys, group in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(group_cols, keys))
        if "portfolio" in frame.columns:
            metrics = _portfolio_metrics(
                group[return_col],
                group["turnover"] if "turnover" in group.columns else None,
                periods_per_year,
                risk_free_rate,
            )
        else:
            metrics = _series_metrics(group[return_col], periods_per_year, risk_free_rate)
        rows.append({**key_values, **metrics})
    output = pd.DataFrame(rows)
    if "portfolio" in output.columns:
        output = output.sort_values(
            ["portfolio", "year"],
            key=lambda s: s.map(_portfolio_sort_key) if s.name == "portfolio" else s,
        )
    else:
        output = output.sort_values("year")
    return output.reset_index(drop=True)


def make_ic_report(ic: pd.DataFrame | None) -> pd.DataFrame:
    if ic is None or ic.empty:
        return pd.DataFrame()

    rows = []
    for column, label in (("ic", "IC"), ("rank_ic", "RankIC")):
        if column not in ic.columns:
            continue
        values = ic[column]
        rows.append(
            {
                "metric": label,
                "observations": observation_count(values),
                "mean": _round_float(ic_mean(values), 3),
                "std": _round_float(std_return(values), 3),
                "ir": _round_float(icir(values), 3),
                "abs_mean": _round_float(ic_abs_mean(values), 3),
                "abs_ir": _round_float(ic_abs_ir(values), 3),
            }
        )
    return pd.DataFrame(rows)


def make_factor_stats_report(factor_stats: pd.DataFrame | None) -> pd.DataFrame:
    if factor_stats is None or factor_stats.empty:
        return pd.DataFrame()
    numeric = factor_stats.select_dtypes(include=[np.number])
    rows = {}
    for column in numeric.columns:
        if column in {"trade_date"}:
            continue
        rows[f"{column}_mean"] = clean_series(numeric[column]).mean()
    factor_id = (
        factor_stats["factor_id"].dropna().iloc[0]
        if "factor_id" in factor_stats.columns and factor_stats["factor_id"].notna().any()
        else None
    )
    if factor_id is not None:
        rows = {"factor_id": factor_id, **rows}
    return pd.DataFrame([rows])


def make_backtest_report(
    returns: pd.DataFrame | pd.Series | None,
    ic: pd.DataFrame | None = None,
    factor_stats: pd.DataFrame | None = None,
    return_col: str = "return",
    periods_per_year: int = 240,
    risk_free_rate: float = 0.0,
) -> dict[str, pd.DataFrame]:
    if returns is None:
        portfolio_total = pd.DataFrame()
        portfolio_by_year = pd.DataFrame()
        excess_total = pd.DataFrame()
        excess_by_year = pd.DataFrame()
    else:
        portfolio_total = make_return_report(returns, return_col, periods_per_year, risk_free_rate)
        portfolio_by_year = (
            make_return_report_by_year(returns, return_col, periods_per_year, risk_free_rate)
            if isinstance(returns, pd.DataFrame)
            else pd.DataFrame()
        )
        if isinstance(returns, pd.DataFrame) and _has_valid_column(returns, "excess_return"):
            excess_total = make_return_report(
                returns,
                "excess_return",
                periods_per_year,
                risk_free_rate,
            )
            excess_by_year = make_return_report_by_year(
                returns,
                "excess_return",
                periods_per_year,
                risk_free_rate,
            )
        else:
            excess_total = pd.DataFrame()
            excess_by_year = pd.DataFrame()
    return {
        "portfolio_total": portfolio_total,
        "portfolio_by_year": portfolio_by_year,
        "excess_total": excess_total,
        "excess_by_year": excess_by_year,
        "ic": make_ic_report(ic),
        "factor_stats": make_factor_stats_report(factor_stats),
    }
