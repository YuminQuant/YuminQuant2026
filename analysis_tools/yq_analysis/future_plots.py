from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd


def _finish_axis(ax, title: str, ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.2", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.24)
    ax.spines[["top", "right"]].set_visible(False)


def _stack_returns(ax, x, returns: pd.DataFrame) -> None:
    long = returns["long"].to_numpy()
    short = returns["short"].to_numpy()
    ax.bar(x, long, width=0.77, color="tab:red", label="Long leg")
    same_sign = np.sign(long) == np.sign(short)
    ax.bar(x, short, bottom=np.where(same_sign, long, 0.0),
           width=0.77, color="tab:green", label="Short leg")


def _product_ticks(ax, x, index: pd.Index) -> None:
    codes = index.str.split(".", n=1, regex=False).str[0]
    ax.set_xticks(x, codes, rotation=45, ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_xlim(-0.8, len(x) - 0.2)
    ax.set_xlabel("Product")


def _plot_sector_means(ax, sector: pd.DataFrame, unit: str) -> None:
    x = np.arange(len(sector))
    _stack_returns(ax, x, sector)
    ax.scatter(x, sector["net"], marker="D", color="0.2", s=24, zorder=3, label="Net mean")
    _finish_axis(ax, "Equal-weight sector returns", f"Mean return ({unit})")
    ax.set_xticks(x, sector.index, fontsize=9, rotation=45, ha="right")


def _plot_sector_ic(ax, ic: pd.Series) -> None:
    x = np.arange(len(ic))
    ax.bar(x, ic, width=0.77, color=np.where(ic.ge(0), "tab:red", "tab:green"))
    _finish_axis(ax, "Pooled IC by sector", "Mean Pearson IC")
    ax.set_xticks(x, ic.index, fontsize=9, rotation=45, ha="right")
    for position in x[ic.isna()]:
        ax.annotate("N/A", (position, 0), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7)


def _plot_cumulative_sectors(ax, cumulative: pd.DataFrame, unit: str) -> None:
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (sector, values) in enumerate(cumulative.items()):
        ax.plot(values.index, values, color=colors[i % len(colors)],
                linestyle="-" if i < len(colors) else "--", linewidth=1.5, label=sector)
    _finish_axis(ax, "Cumulative sector returns", f"Cumulative return ({unit})")
    ax.set_xlabel("Trade date")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_future_return_summary(
    report: Mapping, *, title: str = "", unit: str = "bp", save: str | Path | None = None,
    dpi: int | str | None = None,
):
    """Plot sector IC and returns plus a full-width product returns panel."""
    import matplotlib.pyplot as plt

    products = report["product"]
    x = np.arange(len(products))
    fig = plt.figure(figsize=(19.0, 10.8), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    ic_ax = fig.add_subplot(grid[0, 0])
    mean_ax = fig.add_subplot(grid[0, 1])
    cumulative_ax = fig.add_subplot(grid[1, :])
    product_ax = fig.add_subplot(grid[2, :])
    _plot_sector_ic(ic_ax, report["sector_ic"]["ic"])
    _plot_sector_means(mean_ax, report["sector"], unit)
    _plot_cumulative_sectors(cumulative_ax, report["cumulative_sector"], unit)
    _stack_returns(product_ax, x, report["directed"])
    _finish_axis(product_ax, "IC-directed returns by product", f"Mean return ({unit})")
    _product_ticks(product_ax, x, products.index)
    bar_handles, bar_labels = mean_ax.get_legend_handles_labels()
    line_handles, line_labels = cumulative_ax.get_legend_handles_labels()
    fig.legend(bar_handles + line_handles, bar_labels + line_labels,
               loc="outside lower center", ncols=8, frameon=False, fontsize=9)
    if save is not None:
        try:
            path = Path(save)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {"bbox_inches": "tight"}
            if dpi is not None:
                save_kwargs["dpi"] = dpi
            fig.savefig(path, **save_kwargs)
        finally:
            plt.close(fig)
    return fig


def plot_future_sector_comparison(
    cumulative: Mapping[str, pd.DataFrame], *, title: str = "", unit: str = "bp",
):
    """Plot one panel per sector with a shared legend for quantile modes."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not cumulative:
        raise ValueError("At least one futures result is required")
    first = next(iter(cumulative.values()))
    if first.empty or any(
        not frame.index.equals(first.index) or not frame.columns.equals(first.columns)
        for frame in cumulative.values()
    ):
        raise ValueError("Futures comparisons require matching dates and sectors")
    columns = min(4, len(first.columns))
    rows = ceil(len(first.columns) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 3 * rows),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, sector in zip(axes.flat, first.columns):
        for name, frame in cumulative.items():
            ax.plot(frame.index, frame[sector], label=name, linewidth=1.5)
        _finish_axis(ax, str(sector))
    for ax in list(axes.flat)[len(first.columns):]:
        ax.set_axis_off()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[0, 0].xaxis.set_major_locator(locator)
    axes[0, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.supylabel(f"Cumulative return ({unit})")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncols=len(cumulative), frameon=False)
    return fig
