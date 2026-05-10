from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from yq_analysis.metrics import cumulative_curve


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_return_summary requires matplotlib. Install yq-analysis[plot].") from exc
    return plt


def _portfolio_number(name: str) -> int | None:
    match = re.fullmatch(r"group_(\d+)", str(name))
    return int(match.group(1)) if match else None


def _date_index(frame: pd.DataFrame) -> pd.Index:
    for column in ("trade_date", "factor_date", "date"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
            return pd.to_datetime(values.astype(str), format="%Y%m%d", errors="coerce")
    return frame.index


def _series_by_portfolio(frame: pd.DataFrame, portfolio: str, value_col: str) -> pd.Series:
    subset = frame[frame["portfolio"] == portfolio].copy()
    if subset.empty or value_col not in subset.columns:
        return pd.Series(dtype="float64")
    subset.index = _date_index(subset)
    subset = subset.sort_index()
    return subset[value_col]


def _group_names(frame: pd.DataFrame, groups: int | None) -> list[str]:
    names = []
    for name in frame["portfolio"].dropna().unique():
        number = _portfolio_number(str(name))
        if number is not None and (groups is None or number <= groups):
            names.append(str(name))
    return sorted(names, key=lambda name: _portfolio_number(name) or 0)


def _set_zero_aligned_limits(ax_left, ax_right, left_values: Iterable[float], right_values: Iterable[float]) -> None:
    left = np.asarray([value for value in left_values if np.isfinite(value)], dtype=float)
    right = np.asarray([value for value in right_values if np.isfinite(value)], dtype=float)
    if left.size == 0 or right.size == 0:
        return

    def components(values: np.ndarray) -> tuple[float, float]:
        lower = min(float(values.min()), 0.0)
        upper = max(float(values.max()), 0.0)
        if abs(upper - lower) <= np.finfo(float).eps:
            upper = 0.01
            lower = -0.01
        return max(0.0, -lower), max(0.0, upper)

    left_neg, left_pos = components(left)
    right_neg, right_pos = components(right)
    fractions = []
    for neg, pos in ((left_neg, left_pos), (right_neg, right_pos)):
        total = neg + pos
        fractions.append(neg / total if total > 0 else 0.5)
    zero_frac = min(max(max(fractions), 0.05), 0.95)

    def limits(neg: float, pos: float) -> tuple[float, float]:
        total = max(
            neg / zero_frac if zero_frac > 0 else 0.0,
            pos / (1.0 - zero_frac) if zero_frac < 1.0 else 0.0,
            0.01,
        )
        return -zero_frac * total, (1.0 - zero_frac) * total

    ax_left.set_ylim(*limits(left_neg, left_pos))
    ax_right.set_ylim(*limits(right_neg, right_pos))


def plot_return_summary(
    returns: pd.DataFrame,
    groups: int | None = 10,
    return_col: str = "return",
    figsize: tuple[float, float] = (14.0, 9.0),
    title: str | None = None,
):
    """Plot group and long-short cumulative returns plus end-group turnover."""

    if returns is None or returns.empty:
        raise ValueError("returns is empty")
    if "portfolio" not in returns.columns:
        raise ValueError("returns must contain a portfolio column")

    plt = _require_matplotlib()
    fig, (ax_ret, ax_turnover) = plt.subplots(2, 1, figsize=figsize, constrained_layout=True)
    ax_ls = ax_ret.twinx()

    group_names = _group_names(returns, groups)
    cmap = plt.get_cmap("coolwarm")
    left_values: list[float] = []
    for idx, name in enumerate(group_names):
        series = _series_by_portfolio(returns, name, return_col)
        curve = cumulative_curve(series)
        if curve.empty:
            continue
        color = cmap(idx / max(len(group_names) - 1, 1))
        ax_ret.plot(curve.index, curve.values, color=color, linewidth=1.2, alpha=0.9, label=name)
        left_values.extend(curve.values.tolist())

    ls_series = _series_by_portfolio(returns, "long_short", return_col)
    ls_curve = cumulative_curve(ls_series)
    right_values: list[float] = []
    if not ls_curve.empty:
        ax_ls.plot(
            ls_curve.index,
            ls_curve.values,
            color="#1f1f1f",
            linewidth=2.2,
            label="long_short",
        )
        right_values.extend(ls_curve.values.tolist())

    if left_values and right_values:
        _set_zero_aligned_limits(ax_ret, ax_ls, left_values, right_values)
    ax_ret.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax_ret.set_ylabel("Group cumulative return")
    ax_ls.set_ylabel("Long-short cumulative return")
    ax_ret.set_title(title or "Backtest cumulative returns")

    handles_left, labels_left = ax_ret.get_legend_handles_labels()
    handles_right, labels_right = ax_ls.get_legend_handles_labels()
    ax_ret.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(max(len(labels_left + labels_right), 1), 6),
        frameon=False,
    )

    if group_names and "turnover" in returns.columns:
        first = group_names[0]
        last = group_names[-1]
        for name, color in ((first, "#3b6fb6"), (last, "#b63b3b")):
            series = _series_by_portfolio(returns, name, "turnover")
            if series.empty:
                continue
            ax_turnover.plot(series.index, series.values, linewidth=1.4, label=f"{name} turnover", color=color)
        ax_turnover.set_ylabel("Turnover")
        ax_turnover.set_title("End group turnover")
        ax_turnover.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    else:
        ax_turnover.text(0.5, 0.5, "No turnover data", transform=ax_turnover.transAxes, ha="center", va="center")
        ax_turnover.set_axis_off()

    return fig
