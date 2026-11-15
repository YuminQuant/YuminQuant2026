"""Reconstruct product-level intraday positions and PnL from a futures CTA run."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from yq_analysis.cta_intraday_sources import load_intraday_inputs, prior_trading_days
from yq_analysis.strategy import _futures_trading_clock, futures_products


_OPENING = {"buy", "short"}
_DIRECTIONS = {"buy": 1, "sell": 1, "short": -1, "cover": -1}
_THRESHOLD_LOOKBACK = 5


@dataclass
class _Position:
    quantity: float
    average: float
    mark: float
    multiplier: float


def _read_runs(folder, product, start, end):
    years = sorted(path for path in folder.iterdir() if path.is_dir() and len(path.name) == 4 and path.name.isdigit())
    paths = [path for path in years if start // 10000 <= int(path.name) <= end // 10000] if years else [folder]
    for path in paths:
        pnl = pd.read_parquet(path / "product_pnl.parquet", columns=["trade_date", "product", "net_pnl", "trade_count"])
        pnl = pnl.loc[pnl["product"].eq(product)].set_index("trade_date").sort_index()
        pnl = pnl.loc[pnl.index <= end]
        if not pnl.index.is_unique:
            raise ValueError(f"Duplicate {product} daily PnL in {path}")
        if pnl.loc[pnl.index >= start].empty:
            continue
        holdings = pd.read_parquet(path / "holdings.parquet", columns=["trade_date", "symbols_json", "signed_quantities_json"])
        holdings = holdings.loc[holdings.trade_date.le(end)].sort_values("trade_date")
        if not holdings.trade_date.is_unique or not pnl.index.isin(holdings.trade_date).all():
            raise ValueError(f"Invalid account dates in {path}")
        fills = pd.read_parquet(path / "trades.parquet")
        if not fills.empty and (not fills.bar_frequency.eq("minute").all() or fills.fill_time.isna().any()):
            raise ValueError(f"Intraday reports require dated minute fills in {path}")
        codes = futures_products(pd.Index(fills.symbol))
        fills = fills.loc[(codes == product) & fills.trade_date.le(end)].sort_values("fill_id")
        counts = fills.groupby("trade_date").size()
        if not fills.fill_id.is_unique or not pnl.trade_count.subtract(counts, fill_value=0).eq(0).all():
            raise ValueError(f"Fill counts differ from {product} daily PnL in {path}")
        yield holdings, pnl, fills


def _snapshot(row, product):
    names = pd.Index(json.loads(row.symbols_json), dtype=object)
    quantities = json.loads(row.signed_quantities_json)
    products = futures_products(names)
    return {(code, 1 if quantity > 0 else -1): abs(float(quantity))
            for code, quantity, label in zip(names, quantities, products, strict=True) if label == product and quantity}


def _market_day(root, date, codes):
    columns = ["ts_code", "trade_date", "trade_time", "open", "high", "low", "close"]
    path = Path(root) / str(date // 10000) / f"{date}.parquet"
    bars = pd.read_parquet(path, columns=columns, filters=[("ts_code", "in", sorted(codes))])
    bars = bars.loc[bars.trade_date.eq(date)].rename(columns={"ts_code": "symbol"})
    if bars.duplicated(["symbol", "trade_time"]).any():
        raise ValueError(f"Duplicate quote key on {date}")
    return bars


def _collect_days(run, market_root, product, start, mapping, signals):
    holdings, pnl, fills = run
    by_date = {int(date): list(rows.itertuples(index=False)) for date, rows in fills.groupby("trade_date", sort=False)}
    previous = {}
    for row in holdings.itertuples(index=False):
        date = int(row.trade_date)
        ending = _snapshot(row, product)
        trades = by_date.get(date, [])
        display = date >= start and date in pnl.index
        main = mapping.get(date) if display else None
        wanted = {fill.symbol for fill in trades} | {code for code, _ in previous} | {code for code, _ in ending}
        if display:
            wanted.update([main] if main else [code for code, _ in signals.get(date, {})])
        if wanted or display:
            bars = _market_day(market_root, date, wanted) if wanted else pd.DataFrame(columns=["symbol", "trade_time"])
            yield date, display, main, trades, ending, bars
        previous = ending


def _completed_positions(fills, days):
    quotes = [day[-1][["symbol", "trade_time"]] for day in days if not day[-1].empty]
    matches = defaultdict(list)
    if fills.empty:
        return matches
    if not quotes:
        raise ValueError("Trade timestamps have no matching minute bars")
    clock = _futures_trading_clock(pd.concat(quotes, ignore_index=True))
    clock["fill_time"] = clock.fill_time.dt.strftime("%Y-%m-%d %H:%M:%S")
    stamp = clock.set_index(["symbol", "fill_time"]).trading_minute
    queues = defaultdict(deque)
    for fill in fills.itertuples(index=False):
        key = (fill.symbol, _DIRECTIONS[fill.side])
        if (fill.symbol, fill.fill_time) not in stamp:
            raise ValueError(f"Missing market minute for {fill.symbol} at {fill.fill_time}")
        minute = stamp.loc[fill.symbol, fill.fill_time]
        if fill.side in _OPENING:
            queues[key].append((fill, minute, fill.quantity))
            continue
        pending = fill.quantity
        while pending > 1e-9:
            if not queues[key]:
                raise ValueError(f"Unmatched close for {fill.symbol} at {fill.fill_time}")
            opened, origin, quantity = queues[key].popleft()
            used = min(pending, quantity)
            wall = int((pd.Timestamp(fill.fill_time) - pd.Timestamp(opened.fill_time)).total_seconds() / 60)
            sent = fill.signal_time.partition(" ")[2] if fill.signal_time else ""
            matches[fill.trade_date].append({
                "contract": fill.symbol, "direction": "long" if key[1] > 0 else "short",
                "quantity": float(used), "opened": opened.fill_time, "closed": fill.fill_time,
                "entry": float(opened.price), "exit": float(fill.price), "minutes": int(minute - origin),
                "wall_minutes": wall, "close_signal_time": sent,
            })
            pending -= used
            if quantity - used > 1e-9:
                queues[key].appendleft((opened, origin, quantity - used))
    return matches


def _apply_fill(book, fill):
    direction = _DIRECTIONS[fill.side]
    key = (fill.symbol, direction)
    prior = book.get(key)
    multiplier = fill.notional / (fill.price * fill.quantity)
    if fill.side in _OPENING:
        old_size = prior.quantity if prior is not None else 0.0
        amount = old_size + fill.quantity
        average = (old_size * prior.average + fill.quantity * fill.price) / amount if prior else fill.price
        book[key] = _Position(amount, average, fill.price, multiplier)
    else:
        if prior is None or fill.quantity > prior.quantity + 1e-9:
            raise ValueError(f"Unmatched close for {fill.symbol} at {fill.fill_time}")
        amount = prior.quantity - fill.quantity
        if amount > 1e-9:
            book[key] = _Position(amount, prior.average, fill.price, multiplier)
        else:
            book.pop(key)
    if not math.isclose(amount, abs(fill.position_qty_after), abs_tol=1e-9):
        raise ValueError(f"Replayed position differs for fill {fill.fill_id}")


def _unrealized(book):
    return sum(direction * pos.quantity * (pos.mark - pos.average) * pos.multiplier
               for (_, direction), pos in book.items())


def _replay_day(book, bars, fills, display, main, scores):
    if bars.empty:
        if fills:
            raise ValueError(f"Missing market bars for fill {fills[0].fill_id}")
        return [], [], [], 0.0
    # The futures minute replay only accepts positive finite OHLC bars. It has no minute settle price.
    ohlc = bars[["open", "high", "low", "close"]]
    valid = np.isfinite(ohlc).all(axis=1) & ohlc.gt(0).all(axis=1)
    quotes = bars.loc[valid].sort_values("trade_time")
    entries = defaultdict(list)
    for fill in fills:
        entries[fill.fill_time].append(fill)
    opening_unrealized = _unrealized(book)
    cash = net = 0.0
    times, pnls = [], []
    codes = sorted(quotes.symbol.unique(), key=lambda code: (code != main, code))
    charts = {code: dict(code=code, main=code == main, price=[], signal=[], long=[], short=[], fills=[]) for code in codes}
    for timestamp, rows in quotes.groupby("trade_time", sort=False):
        price = dict(zip(rows.symbol, rows.close))
        for fill in entries.pop(timestamp, []):
            if fill.symbol not in price:
                raise ValueError(f"Fill {fill.fill_id} has no usable quote")
            _apply_fill(book, fill)
            cash += fill.net_realized_pnl
            if display:
                charts[fill.symbol]["fills"].append(dict(index=len(times), time=timestamp, side=fill.side,
                    price=float(fill.price), lots=float(fill.quantity),
                    after=abs(float(fill.position_qty_after)), id=int(fill.fill_id)))
        for (symbol, _), position in book.items():
            if symbol in price:
                position.mark = price[symbol]
        net = cash + _unrealized(book) - opening_unrealized
        if not display:
            continue
        times.append(timestamp)
        pnls.append(round(float(net), 8))
        for code, chart in charts.items():
            chart["price"].append(float(price[code]) if code in price else None)
            score = scores.get((code, timestamp))
            chart["signal"].append(float(score) if score is not None and math.isfinite(score) else None)
            for side, name in [(1, "long"), (-1, "short")]:
                pos = book.get((code, side))
                chart[name].append(side * pos.quantity if pos else 0.0)
    if entries:
        raise ValueError(f"Missing minute bars for fill times {sorted(entries)[:3]}")
    return times, pnls, list(charts.values()) if display else [], net


def _rolling_thresholds(previous, mapping, signals):
    quantiles = {}
    for date in {day for days in previous.values() for day in days}:
        main = mapping.get(date)
        values = [score for (code, _), score in signals.get(date, {}).items()
                  if code == main and score is not None and math.isfinite(score)]
        if values:
            quantiles[date] = np.quantile(values, [.01, .99])
    result = {}
    for date, window in previous.items():
        if len(window) == _THRESHOLD_LOOKBACK and all(day in quantiles for day in window):
            low, high = np.mean([quantiles[day] for day in window], axis=0)
            if low < high:
                result[date] = {"p1": float(low), "p99": float(high)}
    return result


def _day_view(date, pnl, fills, matches, time, curve, charts, thresholds):
    quantity = sum(pair["quantity"] for pair in matches)
    mean = sum(pair["quantity"] * pair["minutes"] for pair in matches) / quantity if quantity else None
    return dict(date=datetime.strptime(str(date), "%Y%m%d").date().isoformat(),
                fills=len(fills), open_fills=sum(fill.side in _OPENING for fill in fills), close_lots=quantity,
                avg=mean, pnl=pnl, positions=matches, charts=charts, time=time, pnl_curve=curve,
                thresholds=thresholds)


def load_intraday_report(
    signal, product, start, end, *, root, market_root, signal_root, mapping_root, calendar_path=None, hdfs_host=None,
):
    """Return a day-selectable report for one product of a finished minute futures backtest."""
    folder = Path(root) / signal
    runs = list(_read_runs(folder, product, start, end))
    dates = sorted(int(date) for _, pnl, _ in runs for date in pnl.index if date >= start)
    if not dates or len(set(dates)) != len(dates):
        raise ValueError(f"No unique backtest sessions for {product} in {start}–{end}")
    if calendar_path is None:
        calendar_path = Path(market_root).parent.parent / "calendar/trade_cal_SHFE.parquet"
    previous = prior_trading_days(calendar_path, dates, _THRESHOLD_LOOKBACK)
    input_dates = {*dates, *(day for window in previous.values() for day in window)}
    mapping, signals = load_intraday_inputs(signal, product, input_dates, signal_root=signal_root,
                                            mapping_root=mapping_root, hdfs_host=hdfs_host)
    thresholds = _rolling_thresholds(previous, mapping, signals)
    views = []
    for run in runs:
        _, pnl, fills = run
        days = list(_collect_days(run, market_root, product, start, mapping, signals))
        matches = _completed_positions(fills, days)
        book = {}
        for date, display, main, trades, ending, bars in days:
            times, curve, charts, net = _replay_day(book, bars, trades, display, main, signals.get(date, {}))
            actual = {key: position.quantity for key, position in book.items()}
            if set(actual) != set(ending) or any(not math.isclose(qty, ending[key], abs_tol=1e-9)
                                                for key, qty in actual.items()):
                raise ValueError(f"Replayed {product} holdings differ from account day-end on {date}")
            expected = float(pnl.at[date, "net_pnl"]) if date in pnl.index else 0.0
            if not math.isclose(net, expected, abs_tol=.0050001, rel_tol=0):
                raise ValueError(f"{product} intraday PnL does not reconcile on {date}: {net:.6f} vs {expected:.6f}")
            if display:
                views.append(_day_view(date, expected, trades, matches[date], times, curve, charts,
                                       thresholds.get(date)))
    if len(views) != len(dates):
        raise ValueError(f"Missing replay sessions for {product}")
    averages = [day["avg"] for day in views if day["avg"] is not None]
    return dict(strategy=signal, product=product, days=views,
                mean=sum(averages) / len(averages) if averages else None)
