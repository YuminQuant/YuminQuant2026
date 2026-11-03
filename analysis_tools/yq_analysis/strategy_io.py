"""Account and actual fills from local or registered deployment storage."""
from pathlib import Path
import json
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import pandas as pd
from .io import load_strategy_result
from .report import make_strategy_report


def load_strategy_run(root, strategy_id, *, storage=None, periods_per_year=240):
    folder = Path(root) / strategy_id
    def text(path):
        return storage.read_text(path, encoding="utf-8") if storage is not None else path.read_text(encoding="utf-8")
    config = tomllib.loads(text(folder / "run_config.toml"))
    holdings = load_strategy_result(root, strategy_id, storage=storage)
    if storage is not None:
        with storage.open_file(folder / "fills.csv", "r", encoding="utf-8") as stream:
            fills = pd.read_csv(stream)
    else:
        fills = pd.read_csv(folder / "fills.csv")
    mapping = {"trade_symbols_json":"symbol", "trade_sides_json":"side_code",
        "trade_quantities_json":"quantity", "trade_prices_json":"price",
        "trade_realized_pnls_json":"realized_pnl", "trade_net_pnls_json":"net_realized_pnl"}
    fills["side_code"] = fills.side.map({"buy":1,"sell":-2})
    if fills.side_code.isna().any():
        raise ValueError("Unsupported stock fill side")
    # Daily-only engine output deliberately omits intraday trade arrays.
    # Attach true fills only to the last snapshot; detailed output already has arrays.
    if not holdings.duplicated("trade_date").any():
        for target, source in mapping.items():
            values = fills.groupby("trade_date")[source].agg(list).map(json.dumps)
            holdings[target] = holdings.trade_date.map(values).fillna("[]")
    report = make_strategy_report(holdings, initial_cash=config["initial_cash"], periods_per_year=periods_per_year)
    daily = report["daily"]
    if not daily.trade_detail_complete.all():
        raise ValueError("Snapshot fill counts differ from actual fills")
    return report, fills, config
