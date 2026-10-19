"""Independently check strategy002 fills against local minute bars and T-day ranks."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_strategy002_research import data_root, load_report

def main():
    root = data_root()
    report, fills = load_report()
    cal = pd.read_parquet(root / "calendar/trade_cal_SSE.parquet")
    dates = sorted(cal.loc[cal.is_open == 1, "cal_date"].astype(int))
    stats = {"fills": len(fills), "rebalance_dates": fills.trade_date.nunique(),
        "max_fill_price_error": 0.0, "fills_blocked_by_daily_close_rule": 0,
        "max_minute_daily_open_difference": 0.0, "max_minute_daily_close_difference": 0.0}
    for date, group in fills.groupby("trade_date"):
        previous = max(d for d in dates if d < date)
        signal = pd.read_parquet(root / f"models/{previous//10000}/{previous}.parquet", columns=["ts_code", "mdl_000009"])
        signal = signal.loc[np.isfinite(signal.mdl_000009) & ~signal.ts_code.str.endswith(".BJ")]
        top = set(signal.sort_values(["mdl_000009", "ts_code"], ascending=[False,True]).head(50).ts_code)
        assert set(group.loc[group.side == "buy", "symbol"]) <= top
        symbols = list(group.symbol.unique())
        bars = pd.read_parquet(root / f"chn_stock_data/minute/{date//10000}/{date}.parquet",
            filters=[("ts_code", "in", symbols)], columns=["ts_code","trade_time","open","close","vol"])
        merged = group.merge(bars, left_on=["symbol","trade_time"], right_on=["ts_code","trade_time"], validate="many_to_one", how="left")
        assert merged.open.notna().all()
        error = (merged.price - merged.open).abs().max()
        stats["max_fill_price_error"] = max(stats["max_fill_price_error"], float(error))
        assert error < 1e-6
        assert (merged.vol > 0).all()
        limits = pd.read_parquet(root / f"chn_stock_data/daily/limit/{date//10000}/{date}.parquet")
        merged = merged.merge(limits[["ts_code","up_limit","down_limit"]], on="ts_code", validate="many_to_one")
        buys = merged.loc[merged.side == "buy"]
        sells = merged.loc[merged.side == "sell"]
        assert not (buys.up_limit.gt(0) & buys.price.ge(buys.up_limit - .0001)).any()
        assert not (sells.down_limit.gt(0) & sells.price.le(sells.down_limit + .0001)).any()
        daily = pd.read_parquet(root / f"chn_stock_data/daily/pv/{date//10000}/{date}.parquet",columns=["ts_code","open","close"])
        checks = bars.sort_values("trade_time").groupby("ts_code").agg(minute_open=("open","first"),minute_close=("close","last")).reset_index().merge(daily,on="ts_code")
        stats["max_minute_daily_open_difference"] = max(stats["max_minute_daily_open_difference"],float((checks.minute_open-checks.open).abs().max()))
        stats["max_minute_daily_close_difference"] = max(stats["max_minute_daily_close_difference"],float((checks.minute_close-checks.close).abs().max()))
        filters = pd.read_parquet(root / f"chn_stock_data/daily/trade_filter/{date//10000}/{date}.parquet")
        m = group.merge(filters, left_on="symbol",right_on="ts_code",validate="many_to_one")
        stats["fills_blocked_by_daily_close_rule"] += int(((m.side.eq("buy") & m.is_limit_up) | (m.side.eq("sell") & m.is_limit_down)).sum())
    output = root / "strategy/stock/strategy_002/fill_validation.json"
    output.write_text(json.dumps(stats,indent=2,default=int),encoding="utf-8")
    print(json.dumps(stats,indent=2,default=int),flush=True)

if __name__ == "__main__":
    main()
