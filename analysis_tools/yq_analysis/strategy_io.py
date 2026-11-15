"""Account and actual fills from local or registered deployment storage."""
from pathlib import Path
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
    exists = storage.exists if storage is not None else Path.exists
    parquet = folder / "trades.parquet"
    csv = folder / "fills.csv"
    fills = None
    if exists(parquet):
        fills = storage.read_parquet(parquet) if storage is not None else pd.read_parquet(parquet)
    elif exists(csv):
        if storage is not None:
            with storage.open_file(csv, "r", encoding="utf-8") as stream:
                fills = pd.read_csv(stream)
        else:
            fills = pd.read_csv(csv)
    report = make_strategy_report(
        holdings, initial_cash=config["initial_cash"], periods_per_year=periods_per_year, fills=fills,
    )
    if fills is not None and not report["daily"].trade_detail_complete.all():
        raise ValueError("Snapshot fill counts differ from actual fills")
    return report, report["trades"], config
