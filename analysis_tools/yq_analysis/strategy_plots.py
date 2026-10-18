"""Account NAV/drawdown, PnL and deployment comparisons (no benchmark charts)."""
from pathlib import Path

import numpy as np
import pandas as pd


def plot_strategy_comparison(reports: dict[str, dict[str, pd.DataFrame]], *,
                             save: bool | str | Path = False, dpi: int = 120):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if not reports:
        raise ValueError("At least one strategy report is required")
    if len(reports) > 4:
        raise ValueError("Use at most four strategies for readable monthly panels")
    first = next(iter(reports.values()))["daily"]
    if any(not r["daily"].index.equals(first.index) for r in reports.values()):
        raise ValueError("Strategy comparisons require identical observation dates")
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.35, 1, 1])
    nav_grid = grid[0, 0].subgridspec(2, 1, height_ratios=[2, 1], hspace=.04)
    nav_ax = fig.add_subplot(nav_grid[0])
    dd_ax = fig.add_subplot(nav_grid[1], sharex=nav_ax)
    pnl_ax = fig.add_subplot(grid[0, 1])
    usage_ax = fig.add_subplot(grid[1, 0])
    turnover_ax = fig.add_subplot(grid[1, 1])
    colors = ["#126782", "#e07a24", "#6a4c93", "#27864c"]
    for (label, report), color in zip(reports.items(), colors):
        d = report["daily"]
        nav_ax.plot(d.index, d.nav, label=label, color=color)
        dd_ax.plot(d.index, d.drawdown * 100, color=color, lw=.8)
        pnl_ax.plot(d.index, d.account_pnl, label=label, color=color)
        usage_ax.plot(d.index, d.gross_exposure * 100, label=label, color=color, alpha=.9)
        if d.turnover.notna().any():
            turnover_ax.plot(d.index, d.turnover.rolling(20, min_periods=1).mean() * 100,
                             label=label, color=color)
        if len(reports) == 1:
            pnl_ax.plot(d.index, d.net_realized_pnl_cum, label="Net realized", alpha=.7)
            pnl_ax.plot(d.index, d.unrealized_pnl, label="Unrealized", alpha=.7)
            pnl_ax.plot(d.index, d.cost_cum, label="Costs", ls="--", alpha=.7)
    nav_ax.set_title("Account NAV and drawdown (after costs)")
    nav_ax.axhline(1, color="grey", lw=.5)
    nav_ax.legend()
    nav_ax.tick_params(labelbottom=False)
    dd_ax.set_ylabel("Drawdown")
    dd_ax.yaxis.set_major_formatter(PercentFormatter())
    pnl_ax.set_title("Account PnL (currency, run-to-date)")
    pnl_ax.axhline(0, color="grey", lw=.5)
    pnl_ax.legend()
    usage_ax.set_title("Capital utilization = gross stock value / account equity")
    usage_ax.axhline(100, color="grey", ls="--", lw=.7, label="100% invested")
    usage_ax.yaxis.set_major_formatter(PercentFormatter())
    usage_ax.legend()
    turnover_ax.set_title("Gross turnover: 20-day mean (buy + sell / prior equity)")
    turnover_ax.yaxis.set_major_formatter(PercentFormatter())
    if turnover_ax.lines:
        turnover_ax.legend()
    else:
        turnover_ax.text(.5, .5, "Turnover unavailable", ha="center", transform=turnover_ax.transAxes)
    for ax in [nav_ax, dd_ax, pnl_ax, usage_ax, turnover_ax]:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    monthly_grid = grid[2, :].subgridspec(1, len(reports))
    limit = max(.01, max(float(np.nanmax(np.abs(r["monthly_returns"].to_numpy()))) for r in reports.values()))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    monthly_axes = []
    for i, (label, report) in enumerate(reports.items()):
        monthly = report["monthly_returns"]
        ax = fig.add_subplot(monthly_grid[i])
        monthly_axes.append(ax)
        im = ax.imshow(monthly.to_numpy(), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        ax.set_xticks(range(12), [str(m) for m in range(1, 13)])
        ax.set_yticks(range(len(monthly)), monthly.index.astype(str))
        ax.set_title(f"{label}: monthly return (%)")
        for row, values in enumerate(monthly.to_numpy()):
            for col, value in enumerate(values):
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=monthly_axes, orientation="horizontal", shrink=.6, aspect=50,
                 label="Monthly return (%) | grey = outside sample")
    fig.suptitle(f"Strategy comparison | {first.index[0]:%Y-%m-%d} - {first.index[-1]:%Y-%m-%d}", fontsize=18)
    if save:
        path = Path(save) if not isinstance(save, bool) else Path("analysis_tools/plots/strategy/comparison.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return fig


def plot_strategy_summary(report: dict[str, pd.DataFrame], *, rolling_window: int = 240,
                          periods_per_year: int = 240, save: bool | str | Path = False,
                          dpi: int = 120):
    """Single strategy variant; legacy rolling parameters retained for API compatibility."""
    name = str(report["portfolio_total"].iloc[0].portfolio)
    path = Path("analysis_tools/plots/strategy") / f"{name}.png" if save is True else save
    return plot_strategy_comparison({name: report}, save=path, dpi=dpi)
