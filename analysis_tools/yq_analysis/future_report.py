from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from yq_analysis.metrics import annual_return, calmar, clean_series, max_drawdown, sharpe, win_rate


METRICS = ["ic", "cosic", "long_rtn", "short_rtn"]
LEGS = ["long", "short"]


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
    products = pd.Index(summary.loc[summary["ts_code"].ne("all"), "ts_code"].unique())
    if not products.difference(sectors.index).empty:
        raise ValueError(f"Products missing from sector map: {products.difference(sectors.index).tolist()}")


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


def make_future_backtest_report(
    summary: pd.DataFrame, sectors: pd.DataFrame, *, direction: pd.Series | None = None,
) -> dict:
    """Summarize daily product metrics and form equal-weight sector return proxies."""
    _check_summary(summary, sectors)
    market = summary.loc[summary["ts_code"].eq("all")].set_index("trade_date").sort_index()
    products = summary.loc[summary["ts_code"].ne("all")].set_index(["trade_date", "ts_code"])
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
    return {
        "market": market,
        "product": means,
        "direction": direction,
        "directed": directed,
        "sector": sector_returns,
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
    """Summarize sector IC and stock-style metrics from one report's daily bp returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    valid = clean_series(report["daily_product"]).index
    product_ic = report["product_ic"].reindex(valid)
    daily_ic = _sector_daily(product_ic, sectors, report["market"].index)
    return pd.DataFrame([
        {
            "sector": sector,
            "ic": daily_ic["ic"][sector].mean(),
            "cosic": daily_ic["cosic"][sector].mean(),
            **_sector_return_metrics(returns, periods_per_year),
        }
        for sector, returns in report["daily_sector"].items()
    ]).set_index("sector")
