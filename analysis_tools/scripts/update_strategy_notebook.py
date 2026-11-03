"""Refresh Top50 comparisons in the dedicated strategy settlement notebook."""
from pathlib import Path
import json

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "Top50": "strategy_001_top50_5d",
    "Ranks 11-50": "strategy_001_top50_exclude10_5d",
    "Ranks 21-50": "strategy_001_top50_exclude20_5d",
    "Top50 (no price limits)": "strategy_001_top50_no_limits_5d",
}

SETUP = '''from pathlib import Path
import sys
import tomllib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
%matplotlib inline

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "analysis_tools").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "analysis_tools"))
from yq_analysis.io import load_strategy_result
from yq_analysis.report import make_strategy_report
from yq_analysis.plots import plot_strategy_comparison

project_config = tomllib.loads((PROJECT_ROOT / "config.toml").read_text(encoding="utf-8"))
DATA_ROOT = Path(project_config["paths"]["base_data_dir"])
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
CASES = __CASES__
strategy_reports = {}
performance_rows, account_rows, parameter_rows = [], [], []
for label, strategy_id in CASES.items():
    folder = DATA_ROOT / "strategy/stock" / strategy_id
    cfg = tomllib.loads((folder / "run_config.toml").read_text(encoding="utf-8"))
    report = make_strategy_report(load_strategy_result(folder.parent, strategy_id),
        initial_cash=cfg["initial_cash"], periods_per_year=240)
    strategy_reports[label] = report
    performance = report["portfolio_total"].iloc[0].to_dict()
    performance["portfolio"] = label
    performance_rows.append(performance)
    daily = report["daily"]
    activity = report["activity"].iloc[0]
    account_rows.append({"portfolio": label, "end_equity": activity.end_equity,
        "account_pnl": daily.account_pnl.iloc[-1],
        "dividend_income_cum": daily.dividend_income_cum.iloc[-1],
        "dividend_receivable": daily.dividend_receivable.iloc[-1],
        "cost_total": activity.cost_total,
        "mean_utilization(%)": daily.gross_exposure.mean() * 100,
        "mean_positions": activity.mean_positions,
        "mean_gross_turnover(%)": activity["mean_gross_turnover(%)"]})
    parameter_rows.append({"portfolio": label, "signal": cfg["strategy"]["signal_id"],
        "rank_start": cfg["strategy"].get("exclude_top_n", 0) + 1,
        "rank_end": cfg["strategy"]["top_n"],
        "holding_sessions": cfg["strategy"]["rebalance_days"],
        "respect_price_limits": cfg["execution"].get("respect_price_limits", True),
        "initial_cash": cfg["initial_cash"]})
strategy_performance = pd.DataFrame(performance_rows).set_index("portfolio")
strategy_accounts = pd.DataFrame(account_rows).set_index("portfolio")
strategy_parameters = pd.DataFrame(parameter_rows).set_index("portfolio")
__PARAMETERS__
display(strategy_performance.round(3))
display(strategy_accounts.round(3))
'''


def cells(cases, figure_name, show_parameters):
    setup = SETUP.replace('__CASES__', repr(cases)).replace(
        '__PARAMETERS__', 'display(strategy_parameters)' if show_parameters else '')
    plot = f'''strategy_figure = plot_strategy_comparison(
    strategy_reports,
    save=PROJECT_ROOT / "analysis_tools/plots/strategy/{figure_name}.png",
)
display(strategy_figure)
plt.close(strategy_figure)
'''
    return [nbformat.v4.new_markdown_cell('# 策略检验'),
            nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell(plot)]


def execute(cells_to_run, metadata):
    nb = nbformat.v4.new_notebook(cells=cells_to_run, metadata=metadata)
    NotebookClient(nb, timeout=180, kernel_name='python3',
        resources={'metadata': {'path': str(ROOT)}}).execute()
    return nb


def main():
    path = ROOT / 'strategy.ipynb'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'cells': [], 'metadata': {}}
    other_start = next((i for i,c in enumerate(old['cells']) if i > 0 and c['cell_type'] == 'markdown'
                        and ''.join(c['source']).strip().startswith('# ')), len(old['cells']))
    research = execute(cells(CASES, 'top50_variants_5d', True), old['metadata'])
    research.cells[0].source = '# \u7b56\u7565\u7ed3\u7b97'
    updated = json.loads(nbformat.writes(research))
    updated['cells'] += old['cells'][other_start:]
    path.write_text(json.dumps(updated, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('strategy.ipynb refreshed; preserved other strategy sections')



if __name__ == '__main__':
    main()
