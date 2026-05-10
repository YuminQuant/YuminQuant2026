from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


ArrayLike = pd.Series | pd.DataFrame | np.ndarray | Sequence[float]


def clean_series(values: ArrayLike, column: str | None = None) -> pd.Series:
    """Convert input into a finite numeric Series.

    DataFrame inputs use `column` when provided, otherwise a `return` column,
    otherwise a single numeric column.
    """

    if isinstance(values, pd.DataFrame):
        if column is not None:
            if column not in values.columns:
                return pd.Series(dtype="float64")
            series = values[column]
        elif "return" in values.columns:
            series = values["return"]
        elif len(values.columns) == 1:
            series = values.iloc[:, 0]
        else:
            numeric = values.select_dtypes(include=[np.number])
            if len(numeric.columns) != 1:
                raise ValueError(
                    "DataFrame metric input must provide `column` when it has multiple numeric columns"
                )
            series = numeric.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)

    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def observation_count(values: ArrayLike, column: str | None = None) -> int:
    return int(clean_series(values, column).shape[0])


def cumulative_curve(values: ArrayLike, column: str | None = None) -> pd.Series:
    series = clean_series(values, column)
    if series.empty:
        return pd.Series(dtype="float64")
    return (1.0 + series).cumprod() - 1.0


def cumulative_return(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    if series.empty:
        return float("nan")
    return float((1.0 + series).prod() - 1.0)


def mean_return(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    return float(series.mean()) if not series.empty else float("nan")


def std_return(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    if len(series) < 2:
        return float("nan")
    return float(series.std(ddof=1))


def annual_return(
    values: ArrayLike,
    column: str | None = None,
    periods_per_year: int = 240,
) -> float:
    series = clean_series(values, column)
    if series.empty:
        return float("nan")
    total = cumulative_return(series)
    if total <= -1.0:
        return -1.0
    return float((1.0 + total) ** (periods_per_year / len(series)) - 1.0)


def annual_volatility(
    values: ArrayLike,
    column: str | None = None,
    periods_per_year: int = 240,
) -> float:
    std = std_return(values, column)
    if not np.isfinite(std):
        return float("nan")
    return float(std * np.sqrt(periods_per_year))


def sharpe(
    values: ArrayLike,
    column: str | None = None,
    periods_per_year: int = 240,
    risk_free_rate: float = 0.0,
) -> float:
    series = clean_series(values, column)
    if len(series) < 2:
        return float("nan")
    excess = series - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std <= np.finfo(float).eps:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino(
    values: ArrayLike,
    column: str | None = None,
    periods_per_year: int = 240,
    risk_free_rate: float = 0.0,
) -> float:
    series = clean_series(values, column)
    if len(series) < 2:
        return float("nan")
    excess = series - risk_free_rate / periods_per_year
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return float("nan")
    downside_std = downside.std(ddof=1)
    if downside_std <= np.finfo(float).eps:
        return float("nan")
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    if series.empty:
        return float("nan")
    nav = (1.0 + series).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def calmar(
    values: ArrayLike,
    column: str | None = None,
    periods_per_year: int = 240,
) -> float:
    ann = annual_return(values, column, periods_per_year)
    mdd = max_drawdown(values, column)
    if not np.isfinite(ann) or not np.isfinite(mdd) or abs(mdd) <= np.finfo(float).eps:
        return float("nan")
    return float(ann / abs(mdd))


def win_rate(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    if series.empty:
        return float("nan")
    return float((series > 0.0).mean())


def ic_mean(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    return float(series.mean()) if not series.empty else float("nan")


def icir(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    if len(series) < 2:
        return float("nan")
    std = series.std(ddof=1)
    if std <= np.finfo(float).eps:
        return float("nan")
    return float(series.mean() / std)


def ic_abs_mean(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column)
    return float(series.abs().mean()) if not series.empty else float("nan")


def ic_abs_ir(values: ArrayLike, column: str | None = None) -> float:
    series = clean_series(values, column).abs()
    if len(series) < 2:
        return float("nan")
    std = series.std(ddof=1)
    if std <= np.finfo(float).eps:
        return float("nan")
    return float(series.mean() / std)


def finite_mean(values: Any) -> float:
    series = clean_series(values)
    return float(series.mean()) if not series.empty else float("nan")
