from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from yq_analysis.metrics import cumulative_curve
from yq_analysis.report import make_return_report


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
    series = pd.to_numeric(subset[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return series.dropna()


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
        lower = -zero_frac * total
        upper = (1.0 - zero_frac) * total
        padding = max((upper - lower) * 0.05, 0.002)
        return lower - padding, upper + padding

    ax_left.set_ylim(*limits(left_neg, left_pos))
    ax_right.set_ylim(*limits(right_neg, right_pos))


def _pad_single_axis(ax, values: Iterable[float]) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return
    lower = min(float(finite.min()), 0.0)
    upper = max(float(finite.max()), 0.0)
    if abs(upper - lower) <= np.finfo(float).eps:
        pad = max(abs(upper) * 0.05, 0.01)
    else:
        pad = (upper - lower) * 0.05
    ax.set_ylim(lower - pad, upper + pad)


def _plot_group_curves(ax, returns: pd.DataFrame, group_names: list[str], value_col: str, cmap) -> list[float]:
    values: list[float] = []
    for idx, name in enumerate(group_names):
        series = _series_by_portfolio(returns, name, value_col)
        curve = cumulative_curve(series)
        if curve.empty:
            continue
        color = cmap(idx / max(len(group_names) - 1, 1))
        ax.plot(curve.index, curve.values, color=color, linewidth=1.1, alpha=0.9, label=name)
        values.extend(curve.values.tolist())
    return values


def plot_return_summary(
    returns: pd.DataFrame,
    groups: int | None = None,
    return_col: str = "return",
    figsize: tuple[float, float] = (13.0, 10.0),
    title: str | None = None,
):
    """Plot group, excess, annual return, and turnover summaries."""

    if returns is None or returns.empty:
        raise ValueError("returns is empty")
    if "portfolio" not in returns.columns:
        raise ValueError("returns must contain a portfolio column")

    plt = _require_matplotlib()
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(4, 2, height_ratios=[2.2, 2.0, 2.0, 1.8])
    ax_ret = fig.add_subplot(grid[0, :])
    ax_excess = fig.add_subplot(grid[1, :])
    ax_ann = fig.add_subplot(grid[2, 0])
    ax_excess_ann = fig.add_subplot(grid[2, 1])
    ax_turnover = fig.add_subplot(grid[3, :])
    ax_ls = ax_ret.twinx()

    group_names = _group_names(returns, groups)
    cmap = plt.get_cmap("coolwarm")
    left_values = _plot_group_curves(ax_ret, returns, group_names, return_col, cmap)

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
    elif left_values:
        _pad_single_axis(ax_ret, left_values)
    elif right_values:
        _pad_single_axis(ax_ls, right_values)
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

    if "excess_return" in returns.columns:
        excess_values = _plot_group_curves(ax_excess, returns, group_names, "excess_return", cmap)
        _pad_single_axis(ax_excess, excess_values)
        ax_excess.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        ax_excess.set_title("Group excess cumulative returns")
        ax_excess.set_ylabel("Excess cumulative return")
        handles, labels = ax_excess.get_legend_handles_labels()
        if handles:
            ax_excess.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                ncol=min(max(len(labels), 1), 6),
                frameon=False,
            )
    else:
        ax_excess.text(0.5, 0.5, "No excess_return data", transform=ax_excess.transAxes, ha="center", va="center")
        ax_excess.set_axis_off()

    _plot_annual_bars(ax_ann, returns, group_names, return_col, "Annual return by group")
    _plot_annual_bars(ax_excess_ann, returns, group_names, "excess_return", "Annual excess return by group")

    _plot_turnover_bars(ax_turnover, returns, group_names)

    return fig


def _plot_annual_bars(ax, returns: pd.DataFrame, group_names: list[str], value_col: str, title: str) -> None:
    if value_col not in returns.columns:
        ax.text(0.5, 0.5, f"No {value_col} data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    report = make_return_report(returns[returns["portfolio"].isin(group_names)], return_col=value_col)
    if report.empty or "annual_return(%)" not in report.columns:
        ax.text(0.5, 0.5, f"No {value_col} data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    report = report[report["portfolio"].isin(group_names)]
    x = np.arange(len(report))
    colors = _require_matplotlib().get_cmap("coolwarm")(np.linspace(0, 1, max(len(report), 1)))
    ax.bar(x, report["annual_return(%)"], color=colors, width=0.72)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(report["portfolio"], rotation=45, ha="right")
    ax.set_ylabel("Annual return (%)")
    ax.set_title(title)
    _pad_single_axis(ax, report["annual_return(%)"].tolist())


def _plot_turnover_bars(ax, returns: pd.DataFrame, group_names: list[str]) -> None:
    if not group_names or "turnover" not in returns.columns:
        ax.text(0.5, 0.5, "No turnover data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    selected = [group_names[0], group_names[-1]] if len(group_names) > 1 else [group_names[0]]
    means = []
    labels = []
    for name in selected:
        series = _series_by_portfolio(returns, name, "turnover")
        if series.empty:
            continue
        means.append(series.mean() * 100.0)
        labels.append(name)
    if not means:
        ax.text(0.5, 0.5, "No turnover data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    colors = ["#3b6fb6", "#b63b3b"][: len(means)]
    ax.bar(labels, means, color=colors, width=0.45)
    ax.set_ylabel("Mean turnover (%)")
    ax.set_title("End group mean turnover")
    _pad_single_axis(ax, means)
