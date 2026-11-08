from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from yq_analysis.metrics import annual_return, calmar, clean_series, max_drawdown, sharpe, win_rate


METRICS = ["ic", "cosic", "long_rtn", "short_rtn"]
LEGS = ["long", "short"]
SECTOR_PREFIX = "sector:"
TRIGGER_COUNT = "trigger_count"


def orient_future_returns(frame: pd.DataFrame, direction: pd.Series) -> pd.DataFrame:
    """Sign the long and short legs using the supplied product direction."""
    positive = direction.gt(0)
    valid = direction.gt(0) | direction.lt(0)
    upper, lower = frame["long_rtn"], frame["short_rtn"]
    return pd.DataFrame({
        "long": upper.where(positive, lower).where(valid),
        "short": -lower.where(positive, upper).where(valid),
    })


def _check_summary(summary: pd.DataFrame, sectors: pd.DataFrame) -> None:
    required = {"trade_date", "ts_code", *METRICS}
    if missing := required.difference(summary.columns):
        raise ValueError(f"Missing futures summary columns: {sorted(missing)}")
    if summary.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Duplicate trade_date and ts_code in futures summary")
    products = pd.Index(summary.loc[_product_rows(summary), "ts_code"].unique())
    if not products.difference(sectors.index).empty:
        raise ValueError(f"Products missing from sector map: {products.difference(sectors.index).tolist()}")


def _product_rows(summary: pd.DataFrame) -> pd.Series:
    codes = summary["ts_code"]
    return codes.ne("all") & ~codes.str.startswith(SECTOR_PREFIX)


def _mean_sector_ic(
    summary: pd.DataFrame, market: pd.DataFrame, sectors: pd.DataFrame,
) -> pd.DataFrame:
    rows = summary.loc[summary["ts_code"].str.startswith(SECTOR_PREFIX)]
    codes = rows["ts_code"].str.removeprefix(SECTOR_PREFIX)
    correlations = rows.groupby(codes)[["ic", "cosic"]].mean()
    correlations = correlations.reindex(sectors["abbr"].unique())
    if not rows.empty:
        correlations.loc["All"] = market[["ic", "cosic"]].mean()
    return correlations.reindex(["All", *sectors["abbr"].unique()]).rename_axis("sector")


def _sector_daily(
    net: pd.DataFrame, sectors: pd.DataFrame, dates: pd.Index,
) -> dict[str, pd.DataFrame]:
    groups = net.join(sectors["abbr"], on="ts_code")
    by_sector = groups.groupby(["trade_date", "abbr"], observed=True)[list(net.columns)].mean()
    abbr = pd.Index(sectors["abbr"].unique(), name="abbr")
    date_index = pd.to_datetime(dates.astype(str), format="%Y%m%d")
    result = {}
    for name in net.columns:
        wide = by_sector[name].unstack("abbr").reindex(index=dates, columns=abbr)
        wide.index = date_index
        result[name] = wide
    return result


def _mean_daily_triggers(
    products: pd.DataFrame, market: pd.DataFrame, sectors: pd.DataFrame,
) -> pd.Series:
    if TRIGGER_COUNT not in products.columns:
        return pd.Series(dtype=float, name="trade_num")
    groups = products[[TRIGGER_COUNT]].join(sectors["abbr"], on="ts_code")
    daily = groups.groupby(["trade_date", "abbr"], observed=True)[TRIGGER_COUNT].sum(min_count=1)
    means = daily.groupby("abbr", observed=True).mean().astype(float)
    means.loc["All"] = market[TRIGGER_COUNT].mean()
    return means.rename("trade_num").rename_axis("sector")


def make_future_backtest_report(
    summary: pd.DataFrame, sectors: pd.DataFrame, *, direction: pd.Series | None = None,
) -> dict:
    """Summarize daily product metrics and form equal-weight sector return proxies."""
    _check_summary(summary, sectors)
    market = summary.loc[summary["ts_code"].eq("all")].set_index("trade_date").sort_index()
    products = summary.loc[_product_rows(summary)].set_index(["trade_date", "ts_code"])
    means = products.groupby("ts_code", sort=True)[METRICS].mean()
    if direction is None:
        direction = np.sign(means["ic"]).rename("direction")
    directed = orient_future_returns(means, direction.reindex(means.index))
    valid_legs = directed.join(sectors["abbr"]).dropna(subset=LEGS)
    sector_returns = valid_legs.groupby("abbr", observed=True)[LEGS].mean()
    sector_returns = sector_returns.reindex(sectors["abbr"].unique())
    sector_returns["net"] = sector_returns[LEGS].sum(axis=1, min_count=2)
    daily_direction = products.index.get_level_values("ts_code").map(direction)
    daily_legs = orient_future_returns(products, pd.Series(daily_direction, index=products.index))
    daily_product = daily_legs.sum(axis=1, min_count=2).rename("net").sort_index()
    daily_sector = _sector_daily(daily_product.to_frame(), sectors, market.index)["net"]
    market_direction = pd.Series(np.sign(market["ic"].mean()), index=market.index)
    market_net = orient_future_returns(market, market_direction).sum(axis=1, min_count=2)
    return {
        "market": market,
        "market_net": market_net,
        "product": means,
        "direction": direction,
        "directed": directed,
        "sector": sector_returns,
        "sector_ic": _mean_sector_ic(summary, market, sectors),
        "trade_num": _mean_daily_triggers(products, market, sectors),
        "daily_product": daily_product,
        "product_ic": products[["ic", "cosic"]],
        "daily_sector": daily_sector,
        "cumulative_sector": daily_sector.cumsum(),
    }


def compare_future_sector_returns(
    daily_product: Mapping[str, pd.Series], sectors: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Compare cumulative sector returns on identical valid product-date pairs."""
    common = pd.concat(daily_product, axis=1, join="inner").sort_index()
    dates = common.index.get_level_values("trade_date").unique()
    common = common.dropna()
    if common.empty:
        raise ValueError("No common valid product-date pairs for futures comparison")
    daily = _sector_daily(common, sectors, dates)
    return {name: frame.cumsum() for name, frame in daily.items()}


def _sector_return_metrics(returns_bp: pd.Series, periods_per_year: int) -> dict:
    returns = clean_series(returns_bp) / 10_000
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    return {
        "annual_return(%)": 100 * annual_return(returns, periods_per_year=periods_per_year),
        "sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "max_drawdown(%)": 100 * max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year=periods_per_year),
        "win_rate(%)": 100 * win_rate(returns),
        "payoff_ratio": gains.mean() / -losses.mean(),
        "observations": len(returns),
    }


def make_future_sector_metrics(
    report: Mapping, sectors: pd.DataFrame, *, periods_per_year: int = 240,
) -> pd.DataFrame:
    """Summarize All and sector pooled IC alongside daily return metrics."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    returns = {"All": report["market_net"]}
    returns.update((sector, report["daily_sector"][sector]) for sector in sectors["abbr"].unique())
    metrics = pd.DataFrame.from_dict(
        {name: _sector_return_metrics(values, periods_per_year) for name, values in returns.items()},
        orient="index",
    ).rename_axis("sector")
    metrics["observations"] = metrics["observations"].astype(int)
    return report["sector_ic"].join(metrics).join(report["trade_num"])
