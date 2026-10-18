"""Controlled stock-strategy experiments; real baselines must match Rust outputs.

The diagnostic-only counterfactuals relax daily limit flags and/or whole-lot
constraints. They retain signal dates, prices, fees, distributions and holding
periods. They are not claims about executable trading returns.
"""
from collections import defaultdict
from pathlib import Path
import json
import hashlib
import math
import tomllib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOP_NS = [10, 20, 50, 100, 200, 300, 500]


def rate(value):
    return math.floor(value * 1e6 + .5) / 1e6 if pd.notna(value) and value > 0 else 0.0


class Portfolio:
    def __init__(self, n, limits, whole_lots, config):
        self.n, self.limits, self.whole_lots = n, limits, whole_lots
        self.initial = self.cash = config['initial_cash']
        execution = config['execution']
        self.buy_rate = (execution.get('buy_commission_bps', execution['commission_bps']) + execution['slippage_bps']) / 10000
        self.sell_rate = (execution.get('sell_commission_bps', execution['commission_bps']) + execution['stamp_tax_bps'] + execution['slippage_bps']) / 10000
        self.lot = execution['lot_size']
        self.q, self.mark = {}, {}
        self.pending, self.receivables, self.locked = [], [], []
        self.dividends = self.costs = 0.0
        self.rows, self.batch_rows = [], []
        self.symbol_pnl = defaultdict(float)
        self.max_error = self.max_quantity_error = 0.0

    def step(self, date, market, flags, selected, events, reference=None):
        previous_values = {s: q * self.mark[s] for s, q in self.q.items()}
        flows, income = defaultdict(float), defaultdict(float)
        for pay, amount in self.receivables:
            if pay <= date:
                self.cash += amount
        self.receivables = [(pay, amount) for pay, amount in self.receivables if pay > date]
        self.locked = [(d, s, q) for d, s, q in self.locked if d > date]
        for event, entitled, record_mark in self.pending:
            if event['ex_date'] > date:
                continue
            symbol = event['ts_code']
            stock_rate, cash_rate = rate(event['stk_div']), rate(event['cash_div_tax'])
            bonus = entitled * stock_rate
            if self.whole_lots:
                bonus = math.floor(bonus + 1e-6)
            if bonus > 0:
                self.q[symbol] = self.q.get(symbol, 0) + bonus
                listing = event['div_listdate']
                if pd.notna(listing) and listing > date:
                    self.locked.append((int(listing), symbol, bonus))
            if symbol in self.q:
                self.mark[symbol] = max(0, (record_mark - cash_rate) / (1 + stock_rate))
            amount = entitled * cash_rate
            income[symbol] += amount
            self.dividends += amount
            pay = event['pay_date']
            if pd.isna(pay) or pay < event['ex_date']:
                pay = event['ex_date']
            if pay <= date:
                self.cash += amount
            else:
                self.receivables.append((int(pay), amount))
        self.pending = [p for p in self.pending if p[0]['ex_date'] > date]
        if selected is not None:
            names = selected[:self.n]
            locked = defaultdict(float)
            for _, symbol, quantity in self.locked:
                locked[symbol] += quantity
            projected_cash, retained, orders = self.cash, {}, []
            blocked_sell = 0
            for symbol, quantity in sorted(self.q.items()):
                price = market.get(symbol, {}).get('open')
                sellable = price is not None and not (self.limits and flags.get(symbol, (False, False))[1])
                sell_qty = max(0, quantity - locked[symbol]) if sellable else 0
                if sell_qty > 0:
                    orders.append((symbol, -sell_qty))
                    projected_cash += sell_qty * price * (1 - self.sell_rate)
                elif quantity > 0:
                    blocked_sell += 1
                retained[symbol] = quantity - sell_qty
            retained_value = sum(q * market.get(s, {}).get('open', self.mark[s]) for s, q in retained.items())
            budget = max(0, projected_cash + retained_value) / len(names) if names else 0
            desired = []
            limit_slots = missing_slots = tiny_slots = 0
            for symbol in names:
                if symbol not in market:
                    missing_slots += 1
                    continue
                price = market[symbol]['open']
                blocked = self.limits and flags.get(symbol, (False, False))[0]
                limit_slots += int(blocked)
                allocation = 0 if blocked or retained.get(symbol, 0) > 1e-9 else budget
                desired.append((symbol, price, allocation))
            total = sum(b for _, _, b in desired)
            scale = min(1, max(projected_cash, 0) / total) if total > 0 else 1
            for symbol, price, allocation in desired:
                quantity = allocation * scale / (price * (1 + self.buy_rate))
                if self.whole_lots:
                    tiny_slots += int(allocation > 0 and quantity < self.lot)
                    quantity = math.floor(quantity / self.lot) * self.lot
                if quantity > 1e-9:
                    orders.append((symbol, quantity))
            for symbol, signed in orders:
                price = market[symbol]['open']
                if signed > 0:
                    affordable = max(0, self.cash / (price * (1 + self.buy_rate)))
                    if self.whole_lots:
                        affordable = math.floor(affordable / self.lot) * self.lot
                    signed = min(signed, affordable)
                notional = signed * price
                cost = abs(notional) * (self.buy_rate if signed > 0 else self.sell_rate)
                self.cash -= notional + cost
                flows[symbol] += notional + cost
                self.costs += cost
                self.q[symbol] = self.q.get(symbol, 0) + signed
                self.mark[symbol] = price
                if self.q[symbol] <= 1e-9:
                    del self.q[symbol]
            self.batch_rows.append(dict(trade_date=date, selected=len(names), limit_up_slots=limit_slots,
                missing_quote_slots=missing_slots, below_one_lot_slots=tiny_slots,
                blocked_old_positions=blocked_sell, slot_budget=budget, cash=self.cash))
        for symbol in self.q:
            if symbol in market:
                self.mark[symbol] = market[symbol]['close']
        values = {s: q * self.mark[s] for s, q in self.q.items()}
        for symbol in previous_values.keys() | values.keys() | flows.keys() | income.keys():
            self.symbol_pnl[symbol] += values.get(symbol, 0) - previous_values.get(symbol, 0) - flows[symbol] + income[symbol]
        receivable = sum(amount for _, amount in self.receivables)
        equity = self.cash + sum(values.values()) + receivable
        assert self.cash > -.01 and equity > 0
        self.rows.append(dict(trade_date=date, equity=equity, cash=self.cash,
            utilization=sum(values.values()) / equity, positions=len(self.q),
            dividend_income_cum=self.dividends, dividend_receivable=receivable, costs=self.costs))
        for event in events:
            symbol = event['ts_code']
            if self.q.get(symbol, 0) > 0:
                self.pending.append((event, self.q[symbol], self.mark[symbol]))
        if reference is not None:
            self.max_error = max(self.max_error, abs(equity - self.initial - reference.account_pnl),
                abs(self.cash - reference.cash), abs(receivable - reference.dividend_receivable),
                abs(self.dividends - reference.dividend_income_cum))
            saved = dict(zip(json.loads(reference.symbols_json), json.loads(reference.signed_quantities_json)))
            self.max_quantity_error = max(self.max_quantity_error,
                max((abs(saved.get(s, 0) - self.q.get(s, 0)) for s in saved.keys() | self.q.keys()), default=0))


def main():
    project = tomllib.loads((ROOT / 'config.toml').read_text(encoding='utf-8'))
    data = Path(project['paths']['base_data_dir'])
    if not data.is_absolute():
        data = ROOT / data
    out = data / 'strategy/attribution'
    out.mkdir(exist_ok=True, parents=True)
    config = tomllib.loads((ROOT / 'strategy_config/stock/strategy_001_top20_5d.toml').read_text())
    frames = {n: pd.read_parquet(data / f'strategy/stock/strategy_001_top{n}_5d/holdings.parquet') for n in TOP_NS}
    dates = frames[10].trade_date.astype(int).tolist()
    assert all(f.trade_date.tolist() == dates for f in frames.values())
    calendar = pd.read_parquet(data / 'calendar/trade_cal_SSE.parquet')
    all_dates = sorted(calendar.loc[calendar.is_open.eq(1), 'cal_date'].astype(int))
    prior_dates = dict(zip(all_dates[1:], all_dates[:-1]))
    events = pd.concat([pd.read_parquet(p) for p in sorted((data / 'chn_stock_data/dividend').glob('*.parquet'))])
    events = events.loc[events.div_proc.eq('实施') & events.record_date.ge(config['start_date'])
        & events.record_date.le(config['end_date']) & events.ex_date.gt(events.record_date)]
    events = events.drop_duplicates(['ts_code', 'record_date', 'ex_date'])
    events_by_date = {int(d): f.to_dict('records') for d, f in events.groupby('record_date')}
    portfolios = {(n, limits, lots): Portfolio(n, limits, lots, config)
                  for n in TOP_NS for limits in (False, True) for lots in (False, True)}
    for i, date in enumerate(dates):
        pv = pd.read_parquet(data / f'chn_stock_data/daily/pv/{date // 10000}/{date}.parquet',
            columns=['ts_code', 'open', 'high', 'low', 'close'])
        pv = pv.loc[np.isfinite(pv[['open', 'high', 'low', 'close']]).all(axis=1)
                    & pv[['open', 'high', 'low', 'close']].gt(0).all(axis=1)]
        market = pv.set_index('ts_code')[['open', 'close']].to_dict('index')
        selected, flags = None, {}
        if i % config['strategy']['rebalance_days'] == 0:
            prior = prior_dates[date]
            signal = pd.read_parquet(data / f'models/{prior // 10000}/{prior}.parquet',
                columns=['trade_date', 'ts_code', config['strategy']['signal_id']])
            score = config['strategy']['signal_id']
            signal = signal.loc[signal.trade_date.eq(prior) & np.isfinite(signal[score])
                & ~signal.ts_code.str.upper().str.endswith('.BJ')].drop_duplicates('ts_code', keep='last')
            selected = signal.sort_values([score, 'ts_code'], ascending=[False, True]).ts_code.tolist()
            f = pd.read_parquet(data / f'chn_stock_data/daily/trade_filter/{date // 10000}/{date}.parquet')
            flags = {r.ts_code: (bool(r.is_limit_up), bool(r.is_limit_down)) for r in f.itertuples()}
        for (n, limits, lots), portfolio in portfolios.items():
            portfolio.step(date, market, flags, selected, events_by_date.get(date, []),
                reference=frames[n].iloc[i] if limits and lots else None)
        if i % 200 == 0:
            print(f'Attribution {i}/{len(dates)} sessions', flush=True)
    summaries = []
    for (n, limits, lots), p in portfolios.items():
        daily, batches = pd.DataFrame(p.rows), pd.DataFrame(p.batch_rows)
        nav = daily.equity / p.initial
        name = f'top{n}_limits{int(limits)}_lots{int(lots)}'
        daily.to_parquet(out / f'{name}.parquet', index=False)
        if limits and lots:
            assert p.max_error < .011 and p.max_quantity_error < 1e-7, (n, p.max_error, p.max_quantity_error)
            batches.to_parquet(out / f'top{n}_execution_diagnostics.parquet', index=False)
            contributors = pd.DataFrame(p.symbol_pnl.items(), columns=['symbol', 'net_pnl']).sort_values('net_pnl')
            assert abs(contributors.net_pnl.sum() - (daily.equity.iloc[-1] - p.initial)) < .01
            contributors.to_parquet(out / f'top{n}_symbol_pnl.parquet', index=False)
        summaries.append(dict(top_n=n, limits=limits, whole_lots=lots,
            end_equity=daily.equity.iloc[-1], cumulative_return_pct=(nav.iloc[-1] - 1) * 100,
            annual_return_pct=(nav.iloc[-1] ** (240 / len(daily)) - 1) * 100,
            max_drawdown_pct=(nav / nav.cummax().clip(lower=1) - 1).min() * 100,
            mean_utilization_pct=daily.utilization.mean() * 100, mean_positions=daily.positions.mean(),
            dividend_income=p.dividends, costs=p.costs,
            mean_limit_up_slots_pct=(batches.limit_up_slots / batches.selected).mean() * 100,
            mean_below_lot_slots_pct=(batches.below_one_lot_slots / batches.selected).mean() * 100,
            max_rust_account_error=p.max_error, max_rust_quantity_error=p.max_quantity_error))
    summary = pd.DataFrame(summaries)
    summary.to_parquet(out / 'scenario_summary.parquet', index=False)
    # Exact two-factor Shapley decomposition, in annualized return percentage points.
    effects = []
    for n in TOP_NS:
        values = summary.loc[summary.top_n.eq(n)].set_index(['limits', 'whole_lots']).annual_return_pct
        ideal, limit_only, lot_only, actual = [values.loc[k] for k in [(False, False), (True, False), (False, True), (True, True)]]
        effects.append(dict(top_n=n, selection_reference=ideal,
            limit_effect=.5 * ((limit_only - ideal) + (actual - lot_only)),
            whole_lot_effect=.5 * ((lot_only - ideal) + (actual - limit_only)), actual=actual))
    pd.DataFrame(effects).to_parquet(out / 'annualized_attribution.parquet', index=False)
    fingerprints = {str((data / f'strategy/stock/strategy_001_top{n}_5d/holdings.parquet').relative_to(data)):
        hashlib.sha256((data / f'strategy/stock/strategy_001_top{n}_5d/holdings.parquet').read_bytes()).hexdigest()
        for n in TOP_NS}
    (out / 'input_fingerprints.json').write_text(json.dumps(fingerprints, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
