"""Save a completed stock strategy's performance and refresh its notebook section."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_tools"))
from yq_analysis.strategy_io import load_strategy_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/strategy/stock")
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--notebook", type=Path, default=ROOT / "strategy.ipynb")
    args = parser.parse_args()
    report, fills, config = load_strategy_run(args.root, args.strategy_id)
    folder = args.root / args.strategy_id
    daily = report["daily"]
    for name in ["portfolio_total","portfolio_by_year","activity","daily","diagnostics","monthly_returns"]:
        report[name].to_csv(folder / (name + ".csv"))
    summary = {**report["portfolio_total"].iloc[0].to_dict(), **report["activity"].iloc[0].to_dict(),
        "mean_utilization_pct":daily.gross_exposure.mean()*100,
        "dividend_income":daily.dividend_income_cum.iloc[-1],
        "first_date":int(daily.trade_date.iloc[0]),"last_date":int(daily.trade_date.iloc[-1])}
    (folder / "summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    print(json.dumps(summary,indent=2,default=str),flush=True)
    import nbformat
    from nbclient import NotebookClient
    title = "# " + args.strategy_id
    existing = nbformat.read(args.notebook,as_version=4) if args.notebook.exists() else nbformat.v4.new_notebook()
    cells = []; skipping = False
    for cell in existing.cells:
        if cell.cell_type == "markdown" and cell.source.strip().startswith("# "):
            skipping = cell.source.strip() == title
        if not skipping: cells.append(cell)
    code = f"""from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
%matplotlib inline
PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'analysis_tools').is_dir())
sys.path.insert(0, str(PROJECT_ROOT / 'analysis_tools'))
from yq_analysis.strategy_io import load_strategy_run
from yq_analysis.plots import plot_strategy_comparison
strategy_report, strategy_fills, strategy_config = load_strategy_run(Path({str(args.root)!r}), {args.strategy_id!r})
display(pd.DataFrame([dict(signal=strategy_config['strategy']['signal_id'],
    top_n=strategy_config['strategy']['top_n'], initial_cash=strategy_config['initial_cash'],
    rebalance_mode=strategy_config['strategy'].get('rebalance_mode', 'round_trip'),
    rebalance_days=strategy_config['strategy']['rebalance_days'],
    twap_minutes=strategy_config['strategy']['twap_minutes'],
    cash_buffer=strategy_config['strategy']['cash_buffer'])]))
display(strategy_report['portfolio_total'].round(3))
display(strategy_report['activity'].round(3))
display(strategy_report['daily'][['equity','account_pnl','dividend_income_cum','gross_exposure']].tail(1).round(3))
fig = plot_strategy_comparison({{'Top50 net TWAP': strategy_report}}, save=PROJECT_ROOT / 'analysis_tools/plots/strategy/{args.strategy_id}.png')
display(fig)
plt.close(fig)
"""
    fresh = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(title),nbformat.v4.new_code_cell(code)],metadata=existing.metadata)
    NotebookClient(fresh,timeout=180,kernel_name="python3",resources={"metadata": {"path":str(ROOT)}}).execute()
    existing.cells = cells + fresh.cells
    nbformat.write(existing,args.notebook)
    print(str(args.notebook),flush=True)

if __name__ == "__main__": main()
