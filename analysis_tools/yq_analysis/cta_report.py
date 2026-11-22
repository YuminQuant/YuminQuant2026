"""Performance and product attribution for futures strategy day-end outputs."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np
import pandas as pd

from yq_analysis.metrics import annual_return, annual_volatility, sharpe
from yq_analysis.strategy import SIDES, futures_daily_holding_totals_by_product, futures_products


def _load_run(path):
    config = tomllib.loads((path / "run_config.toml").read_text(encoding="utf-8"))
    cash = float(config["initial_cash"])
    if not np.isfinite(cash) or cash <= 0:
        raise ValueError(f"Invalid initial_cash in {path}")
    account = pd.read_parquet(path / "holdings.parquet").set_index("trade_date").sort_index()
    products = pd.read_parquet(path / "product_pnl.parquet").sort_values(["trade_date", "product"])
    if account.empty or not account.index.is_unique or products.duplicated(["trade_date", "product"]).any():
        raise ValueError(f"Empty account or duplicate daily rows in {path}")
    aggregate = products.groupby("trade_date")["net_pnl"].sum()
    if not aggregate.index.equals(account.index):
        raise ValueError(f"Account and product dates differ in {path}")
    net_pnl = account.account_pnl.diff()
    net_pnl.iloc[0] = account.account_pnl.iloc[0]
    np.testing.assert_allclose(aggregate, net_pnl, atol=0.0100001, rtol=0, err_msg=str(path))
    np.testing.assert_allclose(aggregate.cumsum(), account.account_pnl, atol=0.0050001, rtol=0, err_msg=str(path))
    equity = cash + account.account_pnl
    previous_equity = equity.shift(fill_value=cash)
    totals = products.groupby("trade_date").agg(
        gross_pnl=("gross_pnl", "sum"), fee=("fee", "sum"), slippage=("slippage", "sum"),
    )
    totals["net_pnl"] = net_pnl
    totals["daily_return"] = (net_pnl / previous_equity).where(previous_equity.gt(0) & equity.gt(0))
    totals["fill_count"] = account[SIDES].sum(axis=1)
    totals["turnover"] = products.groupby("trade_date").turnover_notional.sum() / previous_equity.where(previous_equity.gt(0))
    products["pnl_contribution"] = products.net_pnl / cash
    products["initial_cash"] = cash
    products["run_start"] = int(account.index[0])
    contribution_by_product = products.groupby("product").pnl_contribution
    equity_ratio = 1.0 + contribution_by_product.cumsum()
    previous_ratio = equity_ratio.groupby(products["product"]).shift(fill_value=1.0)
    products["daily_return"] = (products.pnl_contribution / previous_ratio).where(
        previous_ratio.gt(0) & equity_ratio.gt(0),
    )
    return totals, products


def _positive_day_stats(pnl):
    profits = pnl[pnl.gt(0)]
    losses = pnl[pnl.lt(0)]
    return {
        "win_rate": pnl.gt(0).mean(),
        "payoff_ratio": profits.mean() / -losses.mean() if not losses.empty else np.nan,
    }


def _return_metrics(returns, periods_per_year):
    valid = np.isfinite(returns).all()
    return {
        "annual_return": annual_return(returns, periods_per_year=periods_per_year) if valid else np.nan,
        "annual_volatility": annual_volatility(returns, periods_per_year=periods_per_year) if valid else np.nan,
        "sharpe": sharpe(returns, periods_per_year=periods_per_year) if valid else np.nan,
    }


def _portfolio_metrics(daily, periods_per_year):
    returns = daily.daily_return
    valid = np.isfinite(returns).all()
    nav = (1 + returns).cumprod()
    drawdown = (nav / nav.cummax().clip(lower=1) - 1).min() if valid else np.nan
    performance = _return_metrics(returns, periods_per_year)
    return {
        "sessions": len(daily),
        "net_pnl": daily.net_pnl.sum(),
        "gross_pnl": daily.gross_pnl.sum(),
        "fees": daily.fee.sum(),
        "slippage_with_spread": daily.slippage.sum(),
        "total_return": nav.iloc[-1] - 1 if valid else np.nan,
        **performance,
        "max_drawdown": drawdown,
        "calmar": performance["annual_return"] / -drawdown if drawdown < 0 else np.nan,
        **_positive_day_stats(daily.net_pnl),
        "fill_count": daily.fill_count.sum(),
        "avg_daily_turnover": daily.turnover.mean(),
    }


def _product_metrics(group, periods_per_year):
    pnl = group.net_pnl
    cumulative = pnl.cumsum()
    contribution = group.pnl_contribution
    return {
        "sessions": len(group),
        "net_pnl": pnl.sum(),
        "gross_pnl": group.gross_pnl.sum(),
        "fees": group.fee.sum(),
        "slippage_with_spread": group.slippage.sum(),
        **_return_metrics(group.daily_return, periods_per_year),
        "annual_pnl_contribution": contribution.mean() * periods_per_year,
        "contribution_sharpe": sharpe(contribution, periods_per_year=periods_per_year),
        "max_drawdown_pnl": (cumulative - cumulative.cummax().clip(lower=0)).min(),
        **_positive_day_stats(pnl),
        "fill_count": group.trade_count.sum(),
        "avg_daily_turnover": group.turnover_rate.mean(),
    }


def _load_traded_market_minutes(market_root, dates, fills):
    spans = fills.groupby("symbol").trade_date.agg(["min", "max"])
    bars = []
    for date in dates:
        symbols = spans.index[(spans["min"] <= date) & (date <= spans["max"])]
        if not symbols.empty:
            path = market_root / str(date // 10000) / f"{date}.parquet"
            bars.append(pd.read_parquet(path, columns=["ts_code", "trade_time"],
                                        filters=[("ts_code", "in", symbols.tolist())]))
    return pd.concat(bars, ignore_index=True).rename(columns={"ts_code": "symbol"})


def _load_holding_totals(path, market_root, dates, products):
    trade_path = path / "trades.parquet"
    expected = products.set_index(["trade_date", "product"]).trade_count
    if not trade_path.is_file():
        missing = expected[expected.gt(0)].index
        return None, {(date // 10000, product) for date, product in missing}
    fills = pd.read_parquet(trade_path, columns=["trade_date", "fill_id", "symbol", "side", "quantity", "fill_time"])
    traded_products = futures_products(pd.Index(fills.symbol))
    counts = fills.groupby([fills.trade_date, traded_products]).size().rename_axis(["trade_date", "product"])
    if not expected.subtract(counts, fill_value=0).eq(0).all():
        raise ValueError(f"Fill counts differ from product PnL in {path}")
    if fills.empty:
        return None, set()
    market = _load_traded_market_minutes(market_root, dates, fills)
    return futures_daily_holding_totals_by_product(fills, market), set()


def _collect_holding_totals(runs, output, market_root):
    totals, missing = [], set()
    for path, (account, products) in zip(runs, output):
        by_product, unavailable = _load_holding_totals(path, market_root, account.index, products)
        if by_product is not None:
            totals.append(by_product)
        missing.update(unavailable)
    return pd.concat(totals) if totals else None, missing


def _load_report_data(signal, root, market_root, year=None):
    root = Path(root)
    folder = root / signal
    if market_root is None:
        market_root = root.parent.parent / "future_data/minute"
    if not folder.is_dir():
        raise FileNotFoundError(f"Strategy directory not found: {folder}")
    runs = sorted(path for path in folder.iterdir() if path.is_dir() and len(path.name) == 4 and path.name.isdigit())
    if not runs:
        runs = [folder]
    elif year is not None:
        runs = [path for path in runs if path.name == str(year)]
        if not runs:
            raise FileNotFoundError(f"Strategy has no annual run for {year}: {folder}")
    output = [_load_run(path) for path in runs]
    daily = pd.concat([a for a, _ in output], verify_integrity=True).sort_index()
    if year is not None and not (daily.index // 10000 == year).all():
        raise ValueError(f"Selected run must be wholly within {year}: {folder}")
    products = pd.concat([b for _, b in output], ignore_index=True).sort_values(["trade_date", "product"])
    holding, missing = _collect_holding_totals(runs, output, Path(market_root))
    return daily, products, holding, missing


def _portfolio_table(daily, periods_per_year, *, include_total, year_ranges=False):
    metrics = {"All": _portfolio_metrics(daily, periods_per_year)} if include_total else {}
    metrics.update({
        int(year): _portfolio_metrics(group, periods_per_year)
        for year, group in daily.groupby(daily.index // 10000)
    })
    if year_ranges:
        years = sorted(set(daily.index // 10000))
        for i, first in enumerate(years):
            for last in years[i + 1:]:
                selected = daily.loc[(daily.index // 10000 >= first) & (daily.index // 10000 <= last)]
                metrics[f'{first}-{last}'] = _portfolio_metrics(selected, periods_per_year)
    return pd.DataFrame.from_dict(metrics, orient="index").rename_axis("period")


def _product_table(products, holding, missing, periods_per_year, *, year=None):
    table = pd.DataFrame.from_dict({
        product: _product_metrics(group, periods_per_year)
        for product, group in products.groupby("product", sort=True)
    }, orient="index").rename_axis("product")
    if holding is None:
        table["avg_holding_minutes"] = np.nan
        return table
    if year is not None:
        dates = holding.index.get_level_values("trade_date")
        holding = holding.loc[dates // 10000 == year]
    unavailable = {product for close_year, product in missing if year is None or close_year == year}
    daily_duration = holding.lot_minutes / holding.closed_lots.where(holding.closed_lots.gt(0))
    duration = daily_duration.groupby(level="product").mean().reindex(table.index)
    table["avg_holding_minutes"] = duration.mask(duration.index.isin(unavailable))
    return table


def _pnl_curves(daily, products, *, keep_leading_gaps=False):
    pnl = products.pivot(index="trade_date", columns="product", values="net_pnl").reindex(daily.index)
    curves = pnl.fillna(0.0).cumsum()
    if keep_leading_gaps:
        curves = curves.where(pnl.notna().cummax())
    return curves


def load_cta_report(signal, *, root="data/strategy/future", year=None, market_root=None, periods_per_year=240):
    """Read one strategy, optionally selecting an independent annual run."""
    daily, products, holding, missing = _load_report_data(signal, root, market_root, year)
    return (
        _portfolio_table(daily, periods_per_year, include_total=year is None),
        _product_table(products, holding, missing, periods_per_year, year=year),
        _pnl_curves(daily, products),
    )


def scoped_portfolio_tables(daily, products, scopes, periods_per_year=240):
    """Rebuild each selected portfolio from daily PnL, with annual cash resets."""
    result = {"All": _portfolio_table(daily, periods_per_year, include_total=True, year_ranges=True)}
    cash = products.groupby("trade_date").initial_cash.first().reindex(daily.index)
    runs = products.groupby("trade_date").run_start.first().reindex(daily.index)
    for scope, members in scopes.items():
        selected = products[products['product'].isin(members)]
        totals = selected.groupby('trade_date').agg(
            net_pnl=('net_pnl', 'sum'), gross_pnl=('gross_pnl', 'sum'),
            fee=('fee', 'sum'), slippage=('slippage', 'sum'),
            fill_count=('trade_count', 'sum'), turnover=('turnover_notional', 'sum'),
        ).reindex(daily.index, fill_value=0.0)
        cumulative = totals.net_pnl.groupby(runs).cumsum()
        previous = cash + cumulative - totals.net_pnl
        valid = previous.gt(0) & (cash + cumulative).gt(0)
        totals['daily_return'] = (totals.net_pnl / previous).where(valid)
        totals['turnover'] = totals.turnover / previous.where(previous.gt(0))
        result[scope] = _portfolio_table(totals, periods_per_year, include_total=True, year_ranges=True)
    return result


def build_cta_report_views(daily, products, holding, missing, *, periods_per_year=240, product_scopes=None):
    """Build annual and inclusive year-range views from daily backtest records."""
    tables = {"All": _product_table(products, holding, missing, periods_per_year)}
    for year, group in products.groupby(products.trade_date // 10000):
        tables[int(year)] = _product_table(group, holding, missing, periods_per_year, year=year)
    years = sorted(set(daily.index // 10000))
    for i, first in enumerate(years):
        for last in years[i + 1:]:
            selected = products[products.trade_date.between(first * 10000 + 101, last * 10000 + 1231)]
            spans = holding
            if spans is not None:
                dates = spans.index.get_level_values('trade_date')
                spans = spans.loc[(dates >= first * 10000 + 101) & (dates <= last * 10000 + 1231)]
            unavailable = {(year, product) for year, product in missing if first <= year <= last}
            tables[f'{first}-{last}'] = _product_table(selected, spans, unavailable, periods_per_year)
    curves = _pnl_curves(daily, products, keep_leading_gaps=True)
    views = (_portfolio_table(daily, periods_per_year, include_total=True), tables, curves)
    if product_scopes is not None:
        active = {code: daily.index.get_indexer(group.trade_date.unique()).tolist()
                  for code, group in products.groupby('product')}
        return (*views, scoped_portfolio_tables(daily, products, product_scopes, periods_per_year), active)
    return views


def load_cta_report_views(signal, *, root="data/strategy/future", market_root=None, periods_per_year=240, product_scopes=None):
    """Build report tables and curves with gaps before each product's first day."""
    daily, products, holding, missing = _load_report_data(signal, root, market_root)
    return build_cta_report_views(daily, products, holding, missing,
                                  periods_per_year=periods_per_year, product_scopes=product_scopes)


def plot_product_pnl(curves, *, columns=5, rows=4):
    """Yield one PnL figure per page, with a subplot for each product."""
    from math import ceil
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    curves = curves.copy()
    curves.index = pd.to_datetime(curves.index.astype(str), format="%Y%m%d")
    first, last = curves.index[0], curves.index[-1]
    anchor = pd.DataFrame(0.0, index=[first - pd.Timedelta(days=1)], columns=curves.columns)
    curves = pd.concat([anchor, curves])
    page_size = columns * rows
    for start in range(0, len(curves.columns), page_size):
        page = curves.columns[start:start + page_size]
        page_rows = ceil(len(page) / columns)
        fig, axes = plt.subplots(page_rows, columns, figsize=(15, 2.25 * page_rows),
                                 layout="constrained", squeeze=False, sharex=True)
        date_locator = mdates.AutoDateLocator(minticks=3, maxticks=4)
        date_axis = axes[0, 0].xaxis
        date_axis.set_major_locator(date_locator)
        date_axis.set_major_formatter(mdates.ConciseDateFormatter(date_locator, show_offset=False))
        axes[0, 0].set_xlim(curves.index[0], last)
        for index, (product, ax) in enumerate(zip(page, axes.flat)):
            values = curves[product]
            ax.plot(values.index, values, color="C0", linewidth=1.1)
            ax.axhline(0, color="0.55", linewidth=0.7)
            ax.set_title(f"{product}  {values.iloc[-1]:+,.0f}")
            ax.tick_params(labelsize=8)
            ax.tick_params(axis="x", labelbottom=index + columns >= len(page))
            ax.grid(axis="y", alpha=0.25)
        for ax in axes.flat[len(page):]:
            ax.set_visible(False)
        fig.suptitle(f"Product cumulative net PnL (CNY)  |  {first:%Y-%m-%d} to {last:%Y-%m-%d}  |  {page[0]}–{page[-1]}")
        yield fig
