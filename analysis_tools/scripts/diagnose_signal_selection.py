"""Describe Top-50 signal slots, listing ages and theoretical five-session returns.

Forward returns use open * adjustment-factor ratios (gross total return proxy).
They describe selected stocks, including unfillable ones, not executable PnL.
"""
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    project = tomllib.loads((ROOT / 'config.toml').read_text(encoding='utf-8'))
    data = Path(project['paths']['base_data_dir'])
    if not data.is_absolute():
        data = ROOT / data
    config = tomllib.loads((ROOT / 'strategy_config/stock/strategy_001_top50_5d.toml').read_text())
    baseline = pd.read_parquet(data / 'strategy/stock/strategy_001_top50_5d/holdings.parquet')
    dates = baseline.trade_date.astype(int).tolist()
    cal = pd.read_parquet(data / 'calendar/trade_cal_SSE.parquet')
    all_dates = np.sort(cal.loc[cal.is_open.eq(1), 'cal_date'].astype(int))
    prior = dict(zip(all_dates[1:], all_dates[:-1]))
    basic = pd.read_parquet(data / 'chn_stock_data/info/stock_basic.parquet').set_index('ts_code')
    basic['list_date'] = pd.to_numeric(basic.list_date, errors='coerce')
    rows = []
    cache = {}

    def prices(date):
        if date not in cache:
            pv = pd.read_parquet(data / f'chn_stock_data/daily/pv/{date // 10000}/{date}.parquet', columns=['ts_code', 'open'])
            adj = pd.read_parquet(data / f'chn_stock_data/daily/adj_factor/{date // 10000}/{date}.parquet', columns=['ts_code', 'adj_factor'])
            cache[date] = pv.merge(adj, on='ts_code', how='left').set_index('ts_code')
        return cache[date]

    for i in range(0, len(dates), 5):
        date, signal_date = dates[i], int(prior[dates[i]])
        score = config['strategy']['signal_id']
        signal = pd.read_parquet(data / f'models/{signal_date // 10000}/{signal_date}.parquet', columns=['trade_date', 'ts_code', score])
        signal = signal.loc[signal.trade_date.eq(signal_date) & np.isfinite(signal[score])
                            & ~signal.ts_code.str.upper().str.endswith('.BJ')]
        selected = signal.drop_duplicates('ts_code', keep='last').sort_values([score, 'ts_code'], ascending=[False, True]).head(50)
        flags = pd.read_parquet(data / f'chn_stock_data/daily/trade_filter/{date // 10000}/{date}.parquet').set_index('ts_code')
        entry = prices(date)
        exit_date = dates[i + 5] if i + 5 < len(dates) else None
        exit_prices = prices(exit_date) if exit_date else None
        for rank, symbol in enumerate(selected.ts_code, 1):
            listed = basic.at[symbol, 'list_date'] if symbol in basic.index else np.nan
            age = int(np.searchsorted(all_dates, date) - np.searchsorted(all_dates, listed)) if pd.notna(listed) and listed > 0 else np.nan
            limit_up = bool(flags.at[symbol, 'is_limit_up']) if symbol in flags.index else False
            forward = np.nan
            if exit_prices is not None and symbol in entry.index and symbol in exit_prices.index:
                start, end = entry.loc[symbol], exit_prices.loc[symbol]
                if start.open > 0 and start.adj_factor > 0 and end.open > 0 and end.adj_factor > 0:
                    forward = end.open * end.adj_factor / (start.open * start.adj_factor) - 1
            rows.append(dict(entry_date=date, exit_date=exit_date, symbol=symbol, rank=rank,
                band='1-10' if rank <= 10 else '11-20' if rank <= 20 else '21-50',
                listing_age_sessions=age, limit_up=limit_up,
                hypothetical_total_return=forward))
    slots = pd.DataFrame(rows)
    out = data / 'strategy/attribution'
    slots.to_parquet(out / 'selected_slots.parquet', index=False)
    summaries = []
    for (band, limit_up), group in slots.groupby(['band', 'limit_up']):
        returns = group.hypothetical_total_return.dropna()
        summaries.append(dict(rank_band=band, limit_up=limit_up, slots=len(group),
            fresh_60_sessions_pct=group.listing_age_sessions.le(60).mean() * 100,
            completed_return_observations=len(returns),
            mean_5session_total_return_pct=returns.mean() * 100,
            median_5session_total_return_pct=returns.median() * 100,
            positive_return_pct=returns.gt(0).mean() * 100))
    summary = pd.DataFrame(summaries)
    summary.to_parquet(out / 'selected_slot_summary.parquet', index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
