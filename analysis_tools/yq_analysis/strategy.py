"""Account-based strategy reports. No reconstructed orders or round-trip PnL."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from yq_analysis.metrics import annual_return, annual_volatility, sharpe

SIDES = ["buy_count", "sell_count", "short_count", "cover_count"]
POSITION_FIELDS = {
    "symbols_json": "symbol", "signed_quantities_json": "quantity",
    "avg_costs_json": "avg_cost", "market_values_json": "market_value",
    "unrealized_pnls_json": "unrealized_pnl", "multipliers_json": "multiplier",
}
TRADE_FIELDS = {
    "trade_symbols_json": "symbol", "trade_sides_json": "side_code",
    "trade_quantities_json": "quantity", "trade_prices_json": "price",
    "trade_realized_pnls_json": "realized_pnl", "trade_net_pnls_json": "net_pnl",
}


def diagnose_stock_cash(report: dict, *, data_root: str | Path, signal_root: str | Path,
                        signal_id: str, top_n: int, rebalance_days: int = 5,
                        lot_size: int = 100, buy_cost_bps: float = 7,
                        threshold: float = .05) -> pd.DataFrame:
    """Explain high-cash holding periods for the full, unsliced stock strategy run.

    Reconstructs selected names using the preceding session's signal and the
    strict Top-N universe. Missing quotes do not promote lower-ranked stocks.
    This diagnostic follows the deliberately daily
    close-based pessimistic limit flags, rather than claiming a real order book.
    """
    data_root, signal_root = Path(data_root), Path(signal_root)
    daily = report["daily"]
    calendar = pd.read_parquet(data_root / "calendar/trade_cal_SSE.parquet")
    dates = sorted(calendar.loc[calendar.is_open == 1, "cal_date"].astype(int))
    records = []
    for i in range(0, len(daily), rebalance_days):
        period = daily.iloc[i:i + rebalance_days]
        if period.cash_ratio.max() <= threshold:
            continue
        row = period.iloc[0]
        date = int(row.trade_date)
        prior = max(d for d in dates if d < date)
        signal = pd.read_parquet(signal_root / str(prior // 10000) / f"{prior}.parquet",
                                 columns=["ts_code", signal_id])
        pv = pd.read_parquet(data_root / "chn_stock_data/daily/pv" / str(date // 10000) / f"{date}.parquet",
                             columns=["ts_code", "open", "high", "low", "close"])
        pv = pv.loc[np.isfinite(pv[["open", "high", "low", "close"]]).all(axis=1)
                    & pv[["open", "high", "low", "close"]].gt(0).all(axis=1)]
        selected = signal.loc[np.isfinite(signal[signal_id]) & ~signal.ts_code.str.upper().str.endswith(".BJ")]
        selected = selected.sort_values([signal_id, "ts_code"], ascending=[False, True]).head(top_n).merge(pv, on="ts_code", how="left")
        flags = pd.read_parquet(data_root / "chn_stock_data/daily/trade_filter" / str(date // 10000) / f"{date}.parquet",
                                columns=["ts_code", "is_limit_up"])
        selected = selected.merge(flags, on="ts_code", how="left")
        buyable = selected.loc[~selected.is_limit_up.fillna(False).astype(bool) & selected.open.notna()]
        minimum = buyable.open.min() * lot_size * (1 + buy_cost_bps / 10000)
        reason = ("all selected stocks limit-up" if selected.is_limit_up.fillna(False).all() else
                  "unfilled Top-N slots: limits / missing quotes" if len(buyable) < len(selected) else
                  "cash below cheapest executable lot" if row.cash < minimum + .05 else
                  "per-slot lot rounding / frozen prior holdings (no redistribution)")
        records.append({"rebalance_date": date, "signal_date": prior,
                        "selected": len(selected), "buyable": len(buyable),
                        "cash_after_rebalance": row.cash, "cheapest_lot_with_cost": minimum,
                        "max_cash_ratio(%)": period.cash_ratio.max() * 100, "reason": reason})
    return pd.DataFrame(records, columns=["rebalance_date", "signal_date", "selected", "buyable",
                                          "cash_after_rebalance", "cheapest_lot_with_cost",
                                          "max_cash_ratio(%)", "reason"])


def _expand(frame: pd.DataFrame, fields: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        arrays = {}
        for source, target in fields.items():
            value = json.loads(row[source])
            if not isinstance(value, list):
                raise ValueError(f"{source} must contain a JSON array")
            arrays[target] = value
        if len({len(value) for value in arrays.values()}) != 1:
            raise ValueError(f"Mismatched JSON array lengths on {row.trade_date}")
        for index in range(len(next(iter(arrays.values())))):
            rows.append({"trade_date": row.trade_date, "trade_time": row.trade_time,
                         **{name: values[index] for name, values in arrays.items()}})
    return pd.DataFrame(rows, columns=["trade_date", "trade_time", *fields.values()])


def _performance(values: pd.Series, periods_per_year: int) -> dict:
    nav = (1 + values).cumprod()
    dd = nav / nav.cummax().clip(lower=1) - 1
    ann = annual_return(values, periods_per_year=periods_per_year)
    mdd = dd.min()
    return {
        "observations": len(values), "cumulative_return(%)": (nav.iloc[-1] - 1) * 100,
        "annual_return(%)": ann * 100,
        "annual_volatility(%)": annual_volatility(values, periods_per_year=periods_per_year) * 100,
        "sharpe": sharpe(values, periods_per_year=periods_per_year),
        "max_drawdown(%)": mdd * 100,
        "calmar": ann / abs(mdd) if mdd < 0 else np.nan,
        "positive_days(%)": values.gt(0).mean() * 100,
    }


def _drawdown_episodes(daily: pd.DataFrame) -> pd.DataFrame:
    records, active = [], None
    peak, peak_date = 1.0, "period start"
    for date, nav in zip(daily.index, daily.nav):
        if nav >= peak:
            if active is not None:
                active["recovery"] = date
                records.append(active)
                active = None
            peak, peak_date = nav, date
        else:
            depth = (nav / peak - 1) * 100
            if active is None:
                active = {"peak": peak_date, "trough": date, "recovery": pd.NaT,
                          "drawdown(%)": depth, "underwater_days": 0}
            active["underwater_days"] += 1
            if depth < active["drawdown(%)"]:
                active.update({"trough": date, "drawdown(%)": depth})
    if active is not None:
        records.append(active)
    return pd.DataFrame(records, columns=["peak", "trough", "recovery", "drawdown(%)", "underwater_days"]).sort_values("drawdown(%)").reset_index(drop=True)


def make_strategy_report(
    holdings: pd.DataFrame, *, initial_cash: float, periods_per_year: int = 240,
    start_date: int | None = None, end_date: int | None = None,
    benchmark: pd.DataFrame | None = None, benchmark_name: str = "Benchmark",
    expected_dates=None,
) -> dict[str, pd.DataFrame]:
    """Daily metrics from account equity, with first-day return vs initial cash.

    Minute snapshots collapse to the last bar; fill counts sum across the day.
    Return calculation precedes date slicing so the first selected day's PnL is
    retained. Gross turnover counts both buy and sell notional / prior equity.
    Futures turnover is unavailable without per-fill contract specifications.
    """
    if not np.isfinite(initial_cash) or initial_cash <= 0 or periods_per_year <= 0:
        raise ValueError("initial_cash and periods_per_year must be positive")
    required = {"trade_date", "trade_time", "strategy_id", "asset_class", "account_pnl",
                "cash", "net_market_value", "gross_market_value", "unrealized_pnl",
                "realized_pnl_cum", "net_realized_pnl_cum", "margin_required",
                "position_count", *SIDES, *POSITION_FIELDS, *TRADE_FIELDS}
    missing = required - set(holdings)
    if missing:
        raise ValueError(f"Missing strategy columns: {sorted(missing)}")
    if holdings.empty:
        raise ValueError("Strategy output is empty")
    frame = holdings.copy()
    if frame.strategy_id.nunique(dropna=False) != 1 or frame.asset_class.nunique(dropna=False) != 1:
        raise ValueError("Analyze one strategy and asset class at a time")
    asset = frame.asset_class.iloc[0]
    if asset not in {"stock", "future"}:
        raise ValueError(f"Unsupported account type: {asset}")
    frame["trade_date"] = pd.to_numeric(frame.trade_date, errors="raise").astype(int)
    frame = frame.sort_values(["trade_date", "trade_time"], kind="stable")
    if frame.duplicated(["trade_date", "trade_time"]).any():
        raise ValueError("Duplicate account snapshots")
    numeric = list(required - set(POSITION_FIELDS) - set(TRADE_FIELDS)
                   - {"trade_date", "trade_time", "strategy_id", "asset_class"})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Non-finite account values")
    daily = frame.groupby("trade_date", sort=True).tail(1).set_index("trade_date")
    daily[SIDES] = frame.groupby("trade_date")[SIDES].sum()
    daily["equity"] = initial_cash + daily.account_pnl
    if daily.equity.le(0).any():
        raise ValueError("Non-positive equity: return ratios require separate insolvency analysis")
    daily["previous_equity"] = daily.equity.shift(1).fillna(initial_cash)
    daily["return"] = daily.equity / daily.previous_equity - 1
    daily["cost_cum"] = daily.realized_pnl_cum - daily.net_realized_pnl_cum
    daily["cost"] = daily.cost_cum.diff().fillna(daily.cost_cum.iloc[0])
    daily["fill_count"] = daily[SIDES].sum(axis=1)
    lo = start_date if start_date is not None else int(daily.index.min())
    hi = end_date if end_date is not None else int(daily.index.max())
    daily = daily.loc[(daily.index >= lo) & (daily.index <= hi)].copy()
    if daily.empty:
        raise ValueError("No strategy observations in selected dates")
    selected = frame.loc[frame.trade_date.between(lo, hi)]
    trades = _expand(selected, TRADE_FIELDS)
    if not trades.empty:
        trades["side"] = trades.side_code.map({1: "buy", -2: "sell", -1: "short", 2: "cover"})
    detail_counts = trades.groupby("trade_date").size().reindex(daily.index, fill_value=0)
    complete = detail_counts.eq(daily.fill_count)
    if not trades.empty:
        for column in ["quantity", "price"]:
            trades[column] = pd.to_numeric(trades[column], errors="coerce")
        valid_fills = (np.isfinite(trades.quantity) & trades.quantity.gt(0)
                       & np.isfinite(trades.price) & trades.price.gt(0))
        invalid_dates = trades.loc[~valid_fills, "trade_date"]
        complete &= ~daily.index.isin(invalid_dates)
    daily["trade_detail_complete"] = complete
    daily["turnover"] = np.nan
    if asset == "stock":
        trades["notional"] = pd.to_numeric(trades.quantity) * pd.to_numeric(trades.price)
        notional = trades.groupby("trade_date").notional.sum().reindex(daily.index, fill_value=0)
        daily["turnover"] = (notional / daily.previous_equity).where(complete)
    daily["cash_ratio"] = daily.cash / daily.equity
    daily["gross_exposure"] = daily.gross_market_value / daily.equity
    daily["net_exposure"] = daily.net_market_value / daily.equity
    daily["margin_ratio"] = daily.margin_required / daily.equity
    daily["nav"] = (1 + daily["return"]).cumprod()
    daily["drawdown"] = daily.nav / daily.nav.cummax().clip(lower=1) - 1
    daily["dividend_receivable"] = daily.get("dividend_receivable", 0.0)
    daily["dividend_income_cum"] = daily.get("dividend_income_cum", 0.0)
    reconstructed = daily.cash + daily.dividend_receivable + (daily.net_market_value if asset == "stock" else daily.unrealized_pnl)
    daily["equity_reconciliation_error"] = daily.equity - reconstructed
    daily["pnl_reconciliation_error"] = daily.account_pnl - daily.net_realized_pnl_cum - daily.unrealized_pnl
    audit = [
        {"check": "equity identity max error", "value": daily.equity_reconciliation_error.abs().max(), "status": "ok" if daily.equity_reconciliation_error.abs().max() <= .05 else "check initial_cash / output"},
        {"check": "PnL identity max error", "value": daily.pnl_reconciliation_error.abs().max(), "status": "ok" if daily.pnl_reconciliation_error.abs().max() <= .05 else "check"},
        {"check": "days without complete fill detail", "value": int((~complete).sum()), "status": "ok" if complete.all() else "turnover unavailable on these days"},
        {"check": "negative daily cost days", "value": int(daily.cost.lt(-.05).sum()), "status": "ok" if not daily.cost.lt(-.05).any() else "check"},
    ]
    if expected_dates is not None:
        expected = {int(d) for d in expected_dates if lo <= int(d) <= hi}
        missing_dates = expected - set(daily.index)
        audit.append({"check": "missing calendar dates", "value": len(missing_dates), "status": "ok" if not missing_dates else "annualized metrics may be biased"})
    strategy_name = str(frame.strategy_id.iloc[0])
    series = {strategy_name: daily["return"]}
    if benchmark is not None:
        if benchmark_name in {strategy_name, "Relative wealth"}:
            raise ValueError("Benchmark name must differ from strategy name and Relative wealth")
        if benchmark.trade_date.duplicated().any():
            raise ValueError("Duplicate benchmark dates")
        aligned = benchmark.set_index("trade_date")["return"].reindex(daily.index)
        valid = np.isfinite(aligned) & aligned.gt(-1)
        audit.append({"check": "missing/invalid benchmark dates", "value": int((~valid).sum()), "status": "ok" if valid.all() else "comparison disabled"})
        if valid.all():
            daily["benchmark_return"] = aligned
            daily["benchmark_nav"] = (1 + aligned).cumprod()
            daily["relative_return"] = (1 + daily["return"]) / (1 + aligned) - 1
            daily["relative_nav"] = daily.nav / daily.benchmark_nav
            series[benchmark_name] = aligned
            series["Relative wealth"] = daily.relative_return
        else:
            warnings.warn("Benchmark coverage is incomplete; comparison disabled", RuntimeWarning, stacklevel=2)
    summary = pd.DataFrame([{"portfolio": name, **_performance(values, periods_per_year)} for name, values in series.items()])
    yearly = []
    for name, values in series.items():
        for year, group in values.groupby(values.index // 10000):
            yearly.append({"portfolio": name, "year": year, **_performance(group, periods_per_year)})
    daily["trade_date"] = daily.index
    daily.index = pd.to_datetime(daily.index.astype(str), format="%Y%m%d")
    daily.index.name = "date"
    monthly = daily["return"].groupby([daily.index.year, daily.index.month]).agg(lambda s: ((1 + s).prod() - 1) * 100).unstack().reindex(columns=range(1, 13))
    monthly.index.name, monthly.columns.name = "year", "month"
    latest = _expand(selected.tail(1), POSITION_FIELDS)
    latest["weight(%)"] = pd.to_numeric(latest.market_value) / daily.equity.iloc[-1] * 100
    latest = latest.sort_values("weight(%)", ascending=False).reset_index(drop=True)
    activity = pd.DataFrame([{
        "initial_cash": initial_cash, "start_equity": daily.previous_equity.iloc[0],
        "end_equity": daily.equity.iloc[-1], "period_pnl": daily.equity.iloc[-1] - daily.previous_equity.iloc[0],
        "cost_total": daily.cost.sum(), "fill_count": int(daily.fill_count.sum()),
        "trading_days_with_fills": int(daily.fill_count.gt(0).sum()),
        "mean_positions": daily.position_count.mean(), "max_positions": daily.position_count.max(),
        "mean_cash(%)": daily.cash_ratio.mean() * 100,
        "mean_gross_turnover(%)": daily.turnover.mean() * 100,
    }])
    return {"daily": daily, "portfolio_total": summary, "portfolio_by_year": pd.DataFrame(yearly),
            "monthly_returns": monthly, "drawdowns": _drawdown_episodes(daily),
            "activity": activity, "latest_positions": latest, "trades": trades,
            "diagnostics": pd.DataFrame(audit)}
