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
    sortino,
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
) -> dict[str, float | int]:
    return {
        "observations": observation_count(values),
        "cumulative_return": cumulative_return(values),
        "annual_return": annual_return(values, periods_per_year=periods_per_year),
        "annual_volatility": annual_volatility(values, periods_per_year=periods_per_year),
        "sharpe": sharpe(
            values,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "sortino": sortino(
            values,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
        "max_drawdown": max_drawdown(values),
        "calmar": calmar(values, periods_per_year=periods_per_year),
        "win_rate": win_rate(values),
        "mean_return": mean_return(values),
        "std_return": std_return(values),
    }


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
        rows = [{"portfolio": returns.name or "series", **_return_metrics(returns, periods_per_year, risk_free_rate)}]
        return pd.DataFrame(rows)
    if returns is None or returns.empty:
        return pd.DataFrame()

    if "portfolio" not in returns.columns:
        rows = [{"portfolio": "series", **_return_metrics(returns[return_col], periods_per_year, risk_free_rate)}]
        return pd.DataFrame(rows)

    rows = []
    for portfolio, group in returns.groupby("portfolio", sort=False):
        row = {
            "portfolio": portfolio,
            **_return_metrics(group[return_col], periods_per_year, risk_free_rate),
        }
        if "turnover" in group.columns:
            row["turnover_mean"] = clean_series(group["turnover"]).mean()
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
        rows.append(
            {
                **key_values,
                **_return_metrics(group[return_col], periods_per_year, risk_free_rate),
            }
        )
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
                "mean": ic_mean(values),
                "std": std_return(values),
                "ir": icir(values),
                "abs_mean": ic_abs_mean(values),
                "abs_ir": ic_abs_ir(values),
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
    else:
        portfolio_total = make_return_report(returns, return_col, periods_per_year, risk_free_rate)
        portfolio_by_year = (
            make_return_report_by_year(returns, return_col, periods_per_year, risk_free_rate)
            if isinstance(returns, pd.DataFrame)
            else pd.DataFrame()
        )
    return {
        "portfolio_total": portfolio_total,
        "portfolio_by_year": portfolio_by_year,
        "ic": make_ic_report(ic),
        "factor_stats": make_factor_stats_report(factor_stats),
    }
