"""Run minute TWAP strategy002, validate fills, and refresh its notebook section."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_tools"))
from yq_analysis.io import load_strategy_result
from yq_analysis.report import make_strategy_report

SID = "strategy_002"

def data_root():
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    data = Path(cfg["paths"]["base_data_dir"])
    return data if data.is_absolute() else ROOT / data

def load_report():
    folder = data_root() / "strategy/stock" / SID
    cfg = tomllib.loads((folder / "run_config.toml").read_text(encoding="utf-8"))
    holdings = load_strategy_result(folder.parent, SID)
    fills = pd.read_csv(folder / "fills.csv")
    # Restore actual fill details into the daily snapshots for turnover diagnostics.
    mapping = {"trade_symbols_json": "symbol", "trade_sides_json": "side_code",
        "trade_quantities_json": "quantity", "trade_prices_json": "price",
        "trade_realized_pnls_json": "realized_pnl", "trade_net_pnls_json": "net_realized_pnl"}
    fills["side_code"] = fills.side.map({"buy": 1, "sell": -2})
    for target, source in mapping.items():
        values = fills.groupby("trade_date")[source].agg(list).map(json.dumps)
        holdings[target] = holdings.trade_date.map(values).fillna("[]")
    report = make_strategy_report(holdings, initial_cash=cfg["initial_cash"], periods_per_year=240)
    return report, fills

def refresh_notebook():
    import nbformat
    from nbclient import NotebookClient
    path = ROOT / "strategy.ipynb"
    existing = nbformat.read(path, as_version=4) if path.exists() else nbformat.v4.new_notebook()
    kept = []
    for cell in existing.cells:
        if cell.cell_type == "markdown" and cell.source.strip() == "# strategy002":
            break
        kept.append(cell)
    code = """from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'analysis_tools').is_dir())
sys.path.insert(0, str(PROJECT_ROOT / 'analysis_tools/scripts'))
from run_strategy002_research import load_report
from yq_analysis.plots import plot_strategy_comparison
strategy002_report, strategy002_fills = load_report()
display(strategy002_report['portfolio_total'].round(3))
display(strategy002_report['activity'].round(3))
display(strategy002_report['daily'][['equity','account_pnl','dividend_income_cum','gross_exposure']].tail(1).round(3))
fig = plot_strategy_comparison({'Top50 TWAP 30m': strategy002_report}, save=PROJECT_ROOT / 'analysis_tools/plots/strategy/strategy002.png')
display(fig)
plt.close(fig)
"""
    section = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("# strategy002"), nbformat.v4.new_code_cell(code)], metadata=existing.metadata)
    NotebookClient(section, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    existing.cells = kept + section.cells
    nbformat.write(existing, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    folder = data_root() / "strategy/stock" / SID
    folder.mkdir(parents=True, exist_ok=True)
    config = ROOT / "strategy_config/stock/strategy_002.toml"
    exe = ROOT / "factor_engine/target/release/yq-factor-engine.exe"
    if not args.report_only:
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(exe), "strategy-run", "--config", str(config)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(config, folder / "run_config.toml")
        (folder / "provenance.json").write_text(json.dumps({
            "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "holdings_sha256": hashlib.sha256((folder / "holdings.parquet").read_bytes()).hexdigest(),
        }, indent=2), encoding="utf-8")
    report, fills = load_report()
    daily = report["daily"]
    assert daily.equity_reconciliation_error.abs().max() < .05
    assert daily.pnl_reconciliation_error.abs().max() < .05
    assert (fills.cash_after >= 0).all()
    times = fills.trade_time.str[-8:]
    assert times.between("09:30:00", "09:59:00").all()
    assert set(fills.trade_date) <= set(daily.trade_date.iloc[::5])
    assert daily.trade_detail_complete.all()
    assert abs((fills.fee + fills.tax + fills.slippage).sum() - daily.cost.sum()) < .05
    for name in ["portfolio_total", "activity", "diagnostics", "daily"]:
        report[name].to_csv(folder / (name + ".csv"))
    summary = {**report["portfolio_total"].iloc[0].to_dict(), **report["activity"].iloc[0].to_dict(),
        "mean_utilization_pct": daily.gross_exposure.mean() * 100,
        "dividend_income": daily.dividend_income_cum.iloc[-1],
        "first_date": int(daily.trade_date.iloc[0]), "last_date": int(daily.trade_date.iloc[-1])}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    refresh_notebook()
    print("strategy.ipynb updated", flush=True)

if __name__ == "__main__":
    main()
