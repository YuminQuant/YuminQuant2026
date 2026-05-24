from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from yq_analysis.metrics import cumulative_curve
from yq_analysis.report import make_return_report

CNE6_BARRA_FACTORS = [
    "DIVIDEND_YIELD",
    "GROWTH",
    "LIQUIDITY",
    "MOMENTUM",
    "QUALITY",
    "SENTIMENT",
    "SIZE",
    "VALUE",
    "VOLATILITY",
]


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


def _pad_single_axis(ax, values: Iterable[float], pad_ratio: float = 0.05) -> None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return
    lower = min(float(finite.min()), 0.0)
    upper = max(float(finite.max()), 0.0)
    if abs(upper - lower) <= np.finfo(float).eps:
        pad = max(abs(upper) * pad_ratio, 0.01)
    else:
        pad = (upper - lower) * pad_ratio
    ax.set_ylim(lower - pad, upper + pad)


def _group_color_map(group_names: list[str], cmap) -> dict[str, object]:
    return {
        name: cmap(idx / max(len(group_names) - 1, 1))
        for idx, name in enumerate(group_names)
    }


def _barra_color_map(colors: list[object]) -> dict[str, object]:
    if not colors:
        raise ValueError("Barra color list cannot be empty")
    return {
        name: colors[idx % len(colors)]
        for idx, name in enumerate(CNE6_BARRA_FACTORS)
    }


def _plot_group_curves(
    ax,
    returns: pd.DataFrame,
    group_names: list[str],
    value_col: str,
    color_map: dict[str, object],
) -> list[float]:
    values: list[float] = []
    for name in group_names:
        series = _series_by_portfolio(returns, name, value_col)
        curve = cumulative_curve(series)
        if curve.empty:
            continue
        ax.plot(curve.index, curve.values, color=color_map[name], linewidth=1.1, alpha=0.9, label=name)
        values.extend(curve.values.tolist())
    return values


def _infer_factor_name(returns: pd.DataFrame, title: str | None) -> str:
    if "factor_id" in returns.columns:
        values = returns["factor_id"].dropna().unique()
        if len(values) == 1:
            return str(values[0])
    if title:
        return title
    return "backtest_summary"


def _safe_filename(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_\-.]+", "_", value).strip("._")
    return name or "backtest_summary"


def _save_figure(
    fig,
    returns: pd.DataFrame,
    title: str | None,
    save_dir: str | Path | None,
    factor_name: str | None,
    dpi: int | str | None,
) -> Path:
    output_dir = Path(save_dir) if save_dir is not None else Path(__file__).resolve().parents[1] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_filename(factor_name or _infer_factor_name(returns, title))
    path = output_dir / f"{stem}.jpg"
    save_kwargs = {"bbox_inches": "tight"}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(path, **save_kwargs)
    return path


def plot_return_summary(
    returns: pd.DataFrame,
    groups: int | None = None,
    return_col: str = "return",
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    save: bool = True,
    save_dir: str | Path | None = None,
    factor_name: str | None = None,
    dpi: int | str | None = None,
    barra_exposure: pd.DataFrame | None = None,
):
    """Plot group, excess, annual return, turnover, and optional Barra exposure summaries."""

    if returns is None or returns.empty:
        raise ValueError("returns is empty")
    if "portfolio" not in returns.columns:
        raise ValueError("returns must contain a portfolio column")

    plt = _require_matplotlib()
    has_barra = barra_exposure is not None and not barra_exposure.empty
    if figsize is None:
        figsize = (18.5, 8.6) if has_barra else (16.4, 7.8)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=0.048,
        right=0.992,
        top=0.945 if title else 0.978,
        bottom=0.135 if has_barra else 0.105,
        hspace=0.40,
        wspace=0.18,
    )
    if title:
        fig.suptitle(title, y=0.985, fontsize=13)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.05, 0.95], width_ratios=[1.0, 1.6])
    ax_ret = fig.add_subplot(grid[0, 0])
    ax_excess = fig.add_subplot(grid[1, 0], sharex=ax_ret)
    ax_turnover = fig.add_subplot(grid[2, 0], sharex=ax_ret)
    if has_barra:
        right_grid = grid[:, 1].subgridspec(
            4,
            2,
            height_ratios=[0.78, 0.20, 0.78, 0.44],
            hspace=0.20,
            wspace=0.28,
        )
        annual_row = 0
        barra_row = 2
    else:
        right_grid = grid[:, 1].subgridspec(
            3,
            2,
            height_ratios=[0.78, 0.20, 1.00],
            hspace=0.20,
            wspace=0.28,
        )
        annual_row = 0
        barra_row = None
    ax_ann = fig.add_subplot(right_grid[annual_row, 0])
    ax_excess_ann = fig.add_subplot(right_grid[annual_row, 1])
    ax_ls = ax_ret.twinx()

    group_names = _group_names(returns, groups)
    cmap = plt.get_cmap("coolwarm")
    color_map = _group_color_map(group_names, cmap)
    left_values = _plot_group_curves(ax_ret, returns, group_names, return_col, color_map)

    ls_series = _series_by_portfolio(returns, "long_short", return_col)
    ls_curve = cumulative_curve(ls_series)
    right_values: list[float] = []
    if not ls_curve.empty:
        ax_ls.plot(
            ls_curve.index,
            ls_curve.values,
            color="#1f1f1f",
            linewidth=1.4,
            label="long_short",
        )
        right_values.extend(ls_curve.values.tolist())

    if left_values:
        _pad_single_axis(ax_ret, left_values)
    if right_values:
        _pad_single_axis(ax_ls, right_values)
    ax_ret.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax_ret.set_title("Cumulative returns")
    ax_ret.tick_params(axis="x", labelbottom=False)

    handles_left, labels_left = ax_ret.get_legend_handles_labels()
    handles_right, labels_right = ax_ls.get_legend_handles_labels()
    legend_handles = handles_left + handles_right
    legend_labels = labels_left + labels_right

    if "excess_return" in returns.columns:
        excess_values = _plot_group_curves(ax_excess, returns, group_names, "excess_return", color_map)
        _pad_single_axis(ax_excess, excess_values)
        ax_excess.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
        ax_excess.set_title("Group excess cumulative returns")
    else:
        ax_excess.text(0.5, 0.5, "No excess_return data", transform=ax_excess.transAxes, ha="center", va="center")
        ax_excess.set_axis_off()

    _plot_annual_bars(ax_ann, returns, group_names, return_col, "Annual return by group", color_map)
    _plot_annual_bars(ax_excess_ann, returns, group_names, "excess_return", "Annual excess return by group", color_map)

    turnover_handles, turnover_labels = _plot_turnover_lines(ax_turnover, returns, group_names)
    legend_handles.extend(turnover_handles)
    legend_labels.extend(turnover_labels)
    if has_barra:
        ax_barra_ts = fig.add_subplot(right_grid[barra_row, 0])
        ax_barra_mean = fig.add_subplot(right_grid[barra_row, 1])
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        barra_colors = _barra_color_map(default_colors)
        barra_handles, barra_labels = _plot_barra_exposure_timeseries(
            ax_barra_ts,
            barra_exposure,
            barra_colors,
        )
        legend_handles.extend(barra_handles)
        legend_labels.extend(barra_labels)
        _plot_barra_ic_mean_bars(ax_barra_mean, barra_exposure, barra_colors)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=6 if has_barra else 4,
        frameon=False,
        fontsize=8 if has_barra else 9,
    )

    if save:
        _save_figure(fig, returns, title, save_dir, factor_name, dpi)
        plt.close(fig)

    return fig


def _plot_annual_bars(
    ax,
    returns: pd.DataFrame,
    group_names: list[str],
    value_col: str,
    title: str,
    color_map: dict[str, object],
) -> None:
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
    colors = [color_map[name] for name in report["portfolio"]]
    bars = ax.bar(x, report["annual_return(%)"], color=colors, width=0.72)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(report["portfolio"], rotation=45, ha="right")
    ax.set_title(title)
    _pad_single_axis(ax, report["annual_return(%)"].tolist(), pad_ratio=0.025)
    _annotate_bars(ax, bars, report["annual_return(%)"], suffix="%")


def _plot_turnover_lines(ax, returns: pd.DataFrame, group_names: list[str]) -> tuple[list[object], list[str]]:
    if not group_names or "turnover" not in returns.columns:
        ax.text(0.5, 0.5, "No turnover data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    selected = [group_names[0], group_names[-1]] if len(group_names) > 1 else [group_names[0]]
    values: list[float] = []
    handles: list[object] = []
    labels: list[str] = []
    for name, color in zip(selected, ["#3b6fb6", "#b63b3b"]):
        series = _series_by_portfolio(returns, name, "turnover")
        if series.empty:
            continue
        series = series * 100.0
        (line,) = ax.plot(
            series.index,
            series.values,
            linewidth=1.3,
            marker="o",
            markersize=2.0,
            label=f"{name} turnover",
            color=color,
        )
        handles.append(line)
        labels.append(f"{name} turnover")
        values.extend(series.values.tolist())
    if not values:
        ax.text(0.5, 0.5, "No turnover data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    ax.set_title("End group turnover", pad=12)
    _pad_single_axis(ax, values)
    return handles, labels


def _bar_name_order(names: Iterable[str]) -> list[str]:
    name_list = list(names)
    name_set = set(name_list)
    known = [name for name in CNE6_BARRA_FACTORS if name in name_set]
    extras = sorted(name for name in name_list if name not in CNE6_BARRA_FACTORS)
    return known + extras


def _barra_date_index(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("Int64")
    return pd.to_datetime(numeric.astype(str), format="%Y%m%d", errors="coerce")


def _plot_barra_exposure_timeseries(
    ax,
    barra_exposure: pd.DataFrame,
    color_map: dict[str, object],
) -> tuple[list[object], list[str]]:
    required = {"metric", "barra_factor", "trade_date", "value"}
    if barra_exposure is None or barra_exposure.empty or not required.issubset(barra_exposure.columns):
        ax.text(0.5, 0.5, "No Barra exposure data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    frame = barra_exposure[barra_exposure["metric"] == "long_group_exposure"].copy()
    if frame.empty:
        ax.text(0.5, 0.5, "No long group Barra exposure", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    frame["date"] = _barra_date_index(frame["trade_date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["date", "value"])
    if frame.empty:
        ax.text(0.5, 0.5, "No long group Barra exposure", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    values: list[float] = []
    handles: list[object] = []
    labels: list[str] = []
    factors = _bar_name_order(frame["barra_factor"].dropna().astype(str).unique())
    for factor in factors:
        series = (
            frame[frame["barra_factor"] == factor]
            .sort_values("date")
            .set_index("date")["value"]
        )
        if series.empty:
            continue
        series = series.cumsum()
        (line,) = ax.plot(
            series.index,
            series.values,
            color=color_map.get(factor),
            linewidth=1.1,
            marker="o",
            markersize=2.0,
            alpha=0.95,
            label=factor,
        )
        handles.append(line)
        labels.append(factor)
        values.extend(series.values.tolist())
    if not values:
        ax.text(0.5, 0.5, "No long group Barra exposure", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return [], []
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative long group Barra exposure")
    _pad_single_axis(ax, values)
    return handles, labels


def _plot_barra_ic_mean_bars(
    ax,
    barra_exposure: pd.DataFrame,
    color_map: dict[str, object],
) -> None:
    required = {"metric", "barra_factor", "value"}
    if barra_exposure is None or barra_exposure.empty or not required.issubset(barra_exposure.columns):
        ax.text(0.5, 0.5, "No Barra IC data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    mean_frame = barra_exposure[barra_exposure["metric"] == "barra_ic_mean"].copy()
    if mean_frame.empty:
        daily = barra_exposure[barra_exposure["metric"] == "barra_ic"].copy()
        if not daily.empty:
            daily["value"] = pd.to_numeric(daily["value"], errors="coerce")
            mean_frame = (
                daily.groupby("barra_factor", as_index=False)["value"]
                .mean()
                .assign(metric="barra_ic_mean")
            )
    if mean_frame.empty:
        ax.text(0.5, 0.5, "No Barra IC data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    mean_frame["barra_factor"] = mean_frame["barra_factor"].astype(str)
    mean_frame["value"] = pd.to_numeric(mean_frame["value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean_frame = mean_frame.dropna(subset=["value"])
    factors = _bar_name_order(mean_frame["barra_factor"].dropna().unique())
    mean_frame = mean_frame.set_index("barra_factor").reindex(factors).dropna(subset=["value"]).reset_index()
    if mean_frame.empty:
        ax.text(0.5, 0.5, "No Barra IC data", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    x = np.arange(len(mean_frame))
    colors = [color_map.get(name) for name in mean_frame["barra_factor"]]
    bars = ax.bar(x, mean_frame["value"], color=colors, width=0.72)
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(mean_frame["barra_factor"], rotation=45, ha="right")
    ax.set_title("Mean factor-Barra Pearson IC")
    _pad_single_axis(ax, mean_frame["value"].tolist(), pad_ratio=0.025)
    _annotate_bars(ax, bars, mean_frame["value"], suffix="", offset_ratio=0.01)


def _annotate_bars(ax, bars, values: Iterable[float], suffix: str = "", offset_ratio: float = 0.015) -> None:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    if not finite_values:
        return
    span = max(max(finite_values) - min(finite_values), np.finfo(float).eps)
    offset = span * offset_ratio
    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        height = float(value)
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            f"{height:.2f}{suffix}",
            ha="center",
            va=va,
            fontsize=8,
        )
