from pathlib import Path
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[2]
path = root / "test.ipynb"
original = json.loads(path.read_text(encoding="utf-8"))
start = next(i for i, c in enumerate(original["cells"]) if c["cell_type"] == "markdown" and "".join(c["source"]).strip().startswith("# 策略检验"))
prefix = original["cells"][:start]
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
cells = [
md('''# 策略检验：Top 10 / 20 / 50 / 100 / 200 / 300 / 500，5 个交易日调仓

七组均使用 `mdl_000009`，初始资金 10 万元，保留原佣金、印花税、滑点和 100 股整手限制。各组的历史配置快照随结果保存为 `run_config.toml`。

**修正后交易规则**：T 日收盘信号在 T+1 开盘买入，T+6 开盘卖出，持有 5 个交易日。首笔使用回测首日前一交易日信号；每批到期先卖出，再买新一批，重复入选也进行真实卖买并计费。跌停/缺行情不能卖的旧持仓仍保留，后续调仓再次尝试。

**严格 Top N、约等权**：按信号排名选 N 只，不因缺行情或涨停替换为后续排名。每只预算约 1/N，100 股整手向下取整，未成交份额与余额留现金，不重新分配到其他股票，不借款。现金比例较高可以是这项已确认规则的正常结果。

**分红送转**：成交与估值使用未复权真实价格。登记日收盘记录权益，除权日增加送转股并确认税前现金分红；现金到账前计入应收，送转股上市前锁定不可卖。权益 = 现金 + 持仓市值 + 应收股利。股息按本地 `cash_div_tax` 税前口径，不模拟个人持有期差异化红利税；零碎送转股向下取整。数据缺失的派息/上市日期会警告并使用除权日。

**旧结果已撤出正式结果目录**：此前余款分配曾造成 Top10 单只股票权重达 99.7%，且账户遗漏分红送转，旧净值不可直接作为修正后策略表现。旧策略对照：Top20 期末 102,817.75 元；仅取消 2% 缓冲为 102,269.52 元；旧调仓日配合此前余款再分配为 85,828.20 元。这些数值仅用于归因，均未补记公司行动。

不绘制 benchmark 和年度收益。净值与回撤位于左上同一面板，PnL 位于第一行第二列；保留资金利用率、换手率和月度收益。'''),
code('''from pathlib import Path
import sys
import tomllib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "analysis_tools").is_dir())
sys.path.insert(0, str(PROJECT_ROOT / "analysis_tools"))
from yq_analysis.io import load_strategy_result
from yq_analysis.report import make_strategy_report
from yq_analysis.plots import plot_strategy_comparison

TOP_NS = [10, 20, 50, 100, 200, 300, 500]
PERIODS_PER_YEAR = 240
START_DATE = None
END_DATE = None
project_config = tomllib.loads((PROJECT_ROOT / "config.toml").read_text(encoding="utf-8"))
DATA_ROOT = Path(project_config["paths"]["base_data_dir"])
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
strategy_root = DATA_ROOT / "strategy/stock"
calendar = pd.read_parquet(DATA_ROOT / "calendar/trade_cal_SSE.parquet")
strategy_reports, strategy_configs = {}, {}
performance_rows, deployment_rows, diagnostics = [], [], []
for n in TOP_NS:
    label, strategy_id = f"Top {n}", f"strategy_001_top{n}_5d"
    config = tomllib.loads((strategy_root / strategy_id / "run_config.toml").read_text(encoding="utf-8"))
    assert config["strategy"]["top_n"] == n
    assert config["strategy"]["rebalance_days"] == 5 and config["strategy"]["cash_buffer"] == 0
    holdings = load_strategy_result(strategy_root, strategy_id)
    report = make_strategy_report(
        holdings, initial_cash=config["initial_cash"], periods_per_year=PERIODS_PER_YEAR,
        start_date=START_DATE or config["start_date"], end_date=END_DATE or config["end_date"],
        expected_dates=calendar.loc[calendar.is_open == 1, "cal_date"],
    )
    strategy_reports[label], strategy_configs[label] = report, config
    performance = report["portfolio_total"].iloc[0].to_dict()
    performance["portfolio"] = label
    performance_rows.append(performance)
    d = report["daily"]
    activity = report["activity"].iloc[0]
    deployment_rows.append({
        "portfolio": label, "end_equity": activity.end_equity,
        "first_day_utilization(%)": d.gross_exposure.iloc[0] * 100,
        "mean_utilization(%)": d.gross_exposure.mean() * 100,
        "min_utilization(%)": d.gross_exposure.min() * 100,
        "mean_positions": d.position_count.mean(), "max_positions": d.position_count.max(),
        "cost_total": activity.cost_total, "fill_count": int(activity.fill_count),
        "mean_gross_turnover(%)": activity["mean_gross_turnover(%)"],
    })
    assert d.cash.ge(-.01).all(), "Unexpected negative cash"
    assert (d.gross_market_value - d.net_market_value).abs().max() < .05, "Unexpected short exposure"
    diagnostics.append(report["diagnostics"].assign(portfolio=label))

strategy_comparison = pd.DataFrame(performance_rows).set_index("portfolio")
strategy_deployment = pd.DataFrame(deployment_rows).set_index("portfolio")
display(strategy_comparison.round(3))
display(strategy_deployment.round(3))
'''),
md('''## 净值、回撤、PnL 与资金使用对比

收益已扣除费用，年化沿用因子检验的 240 个交易日。换手率为买卖成交金额之和 / 前一日净资产，图中展示 20 日均值；月度热力图使用统一色阶，单位为 %。净值与回撤共同使用同一日期轴，便于对应观察。'''),
code('''strategy_figure = plot_strategy_comparison(
    strategy_reports,
    save=PROJECT_ROOT / "analysis_tools/plots/strategy/top10_20_50_100_200_300_500_5d.png",
)
display(strategy_figure)
plt.close(strategy_figure)
'''),
md('''## Top10 与 Top50：选股、成交限制和资金利用率

下表四种情景使用同一信号、开盘价、5 日持有期、费率与分红送转。正式回测是“保留涨跌停限制 + 100 股整手”；其余为归因实验。允许小数股用于消除整手和预算造成的闲置资金；放开涨跌停是假设不可成交股票也能成交，**收益不可当作可实现业绩**。尤其高排名新股连续涨停，会使这一反事实出现极端高收益。

比较实际与“小数股但保留涨跌停”可以检验闲置资金是否造成 Top10 落后。条件选股表按排名段区分涨停/非涨停股票，用复权因子计算 T+1 至 T+6 开盘的理论总收益，仅用于分析信号的可交易性，不能直接替代账户净值。最后一批未满 5 日不参与该收益均值。

NAV =（现金 + 持仓市值 + 应收股利）/ 初始资金，PnL = 同一净资产 − 初始资金；分红已在两者中体现，不再重复加到图上。现金股利单列不包含送转股增值，送转股已经进入股数和市值。'''),
code('''attribution_root = DATA_ROOT / "strategy/attribution"
import hashlib
import json
for relative_path, expected_hash in json.loads((attribution_root / "input_fingerprints.json").read_text()).items():
    assert hashlib.sha256((DATA_ROOT / relative_path).read_bytes()).hexdigest() == expected_hash, "Attribution is stale: rerun attribute_stock_strategy.py"
scenario_summary = pd.read_parquet(attribution_root / "scenario_summary.parquet")
scenario_names = {
    (True, True): "实际：涨跌停 + 100股整手",
    (True, False): "对照：涨跌停 + 小数股",
    (False, True): "不可成交反事实：无涨跌停 + 整手",
    (False, False): "不可成交反事实：无涨跌停 + 小数股",
}
scenarios = scenario_summary.loc[scenario_summary.top_n.isin([10, 50])].copy()
scenarios["scenario"] = [scenario_names[(r.limits, r.whole_lots)] for r in scenarios.itertuples()]
display(scenarios[["top_n", "scenario", "cumulative_return_pct", "annual_return_pct", "mean_utilization_pct"]].round(3))

actual = scenario_summary.loc[scenario_summary.limits & scenario_summary.whole_lots].copy()
actual["price_pnl_before_costs"] = actual.end_equity - 100000 + actual.costs - actual.dividend_income
actual["total_pnl_before_costs"] = actual.end_equity - 100000 + actual.costs
display(actual[["top_n", "mean_positions", "mean_limit_up_slots_pct", "mean_below_lot_slots_pct",
    "price_pnl_before_costs", "dividend_income", "costs", "total_pnl_before_costs"]].round(3))
display(Markdown("费用加回表保持原成交路径，只是损益核算分解，并非重新运行的免手续费策略。限制比例按调仓日入选名额计数；不足一手比例仅统计未被涨停/旧持仓阻挡的名额。"))

slot_summary = pd.read_parquet(attribution_root / "selected_slot_summary.parquet")
display(slot_summary.round(3))
low = slot_summary.loc[slot_summary.rank_band.eq("1-10") & ~slot_summary.limit_up].iloc[0]
wide = slot_summary.loc[slot_summary.rank_band.eq("21-50") & ~slot_summary.limit_up].iloc[0]
blocked = slot_summary.loc[slot_summary.rank_band.eq("1-10") & slot_summary.limit_up].iloc[0]
display(Markdown(
    f"**信号可交易性结论**：Top10 被涨停挡住的 {int(blocked.slots)} 个名额中，"
    f"{blocked.fresh_60_sessions_pct:.1f}% 为上市不超过 60 个交易日的股票。"
    f"未被涨停挡住的排名 1–10 理论五日平均总收益为 {low.mean_5session_total_return_pct:.3f}%，"
    f"排名 21–50 为 {wide.mean_5session_total_return_pct:.3f}%。"
    "前者低于约 0.19% 的往返费率。Top10 的主要弱点是高分信号的可交易性差，"
    "实际可买部分收益较薄；平均资金利用率不是它落后 Top50 的主要解释。"
    "这些是按名额统计的理论毛收益，未模拟卖出受阻或实际预算，不能直接等同组合回测收益。"
))
effects = pd.read_parquet(attribution_root / "annualized_attribution.parquet").set_index("top_n")
gap = effects.loc[50] - effects.loc[10]
display(gap.rename("Top50 minus Top10 (annualized percentage points)").to_frame().round(3))
display(Markdown("归因表使用两因素 Shapley：平均先去除涨跌停与先去除整手约束两条路径的边际影响。selection_reference 是无限制、小数股参考组合的年化差距，limit_effect 与 whole_lot_effect 是约束影响差；三项之和等于实际年化差距。它受极端不可成交反事实影响，仅作为情景分解，不是独立因果估计。"))

contributors = pd.concat([
    pd.read_parquet(attribution_root / f"top{n}_symbol_pnl.parquet").head(10).assign(top_n=n)
    for n in (10, 50)
], ignore_index=True)
display(contributors.round(2))
'''),
md('''## 现金偏高的具体原因

逐个检查现金比例曾超过 5% 的持有期：按前一交易日信号还原严格 Top N，标记涨停/缺行情的未成交份额，并检查整手约束。现金不能用于给其他股票加大权重。现金诊断是原因分类，不代表逐笔模拟订单簿。'''),
code('''from yq_analysis.strategy import diagnose_stock_cash
cash_diagnostics = []
if START_DATE is None and END_DATE is None:
    for label, report in strategy_reports.items():
        cfg = strategy_configs[label]
        signal_root = Path(cfg["strategy"]["signal_root"])
        if not signal_root.is_absolute():
            signal_root = PROJECT_ROOT / signal_root
        costs = cfg["execution"]
        cash_check = diagnose_stock_cash(
            report, data_root=DATA_ROOT, signal_root=signal_root,
            signal_id=cfg["strategy"]["signal_id"], top_n=cfg["strategy"]["top_n"],
            rebalance_days=cfg["strategy"]["rebalance_days"], lot_size=costs["lot_size"],
            buy_cost_bps=costs.get("buy_commission_bps", costs.get("commission_bps", 0)) + costs["slippage_bps"],
        )
        cash_diagnostics.append(cash_check.assign(portfolio=label))
    cash_diagnostics_all = pd.concat(cash_diagnostics, ignore_index=True)
    display(cash_diagnostics_all.groupby(["portfolio", "reason"]).size().rename("holding_periods").to_frame())
    display(cash_diagnostics_all.sort_values("max_cash_ratio(%)", ascending=False).head(20).round(3))
else:
    display(Markdown("现金原因诊断需使用完整回测区间，请将 START_DATE / END_DATE 恢复为 None。"))
'''),
md('''## 回撤与数据对账

每组展示最深的 3 段回撤；`recovery=NaT` 表示仍未恢复。账户对账、成交明细完整性、交易日覆盖在下表检查；成交笔数是 fill 数量，不是完整往返交易次数。'''),
code('''display(pd.concat([
    r["drawdowns"].head(3).assign(portfolio=label)
    for label, r in strategy_reports.items()
], ignore_index=True).round(3))
display(pd.concat(diagnostics, ignore_index=True))
import json
scenario_summary = pd.read_parquet(DATA_ROOT / "strategy/attribution/scenario_summary.parquet")
ledger_checks = scenario_summary.loc[scenario_summary.limits & scenario_summary.whole_lots,
    ["top_n", "max_rust_account_error", "max_rust_quantity_error"]]
display(ledger_checks)
display(pd.DataFrame([
    {"portfolio": label, "dividend_income_cum": r["daily"].dividend_income_cum.iloc[-1],
     "ending_dividend_receivable": r["daily"].dividend_receivable.iloc[-1]}
    for label, r in strategy_reports.items()
]))
'''),
md('''## 单组明细

修改 `DETAIL_TOP_N` 可查看对应组的最新持仓、最近 10 日资金使用和最近成交。整手向下取整会让权重偏离严格等权；未能卖出的旧持仓可能使持仓数短期超过 Top N。'''),
code('''DETAIL_TOP_N = 20
detail_report = strategy_reports[f"Top {DETAIL_TOP_N}"]
display(detail_report["daily"][["equity", "return", "cash", "dividend_receivable", "dividend_income_cum", "gross_exposure", "position_count", "turnover", "fill_count"]].tail(10).round(4))
display(detail_report["latest_positions"].round(4))
display(detail_report["trades"].tail(20).round(4))
'''),
]
section = nbformat.v4.new_notebook(cells=cells, metadata=original["metadata"])
NotebookClient(section, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
original["cells"] = prefix + json.loads(nbformat.writes(section))["cells"]
assert original["cells"][:start] == prefix
path.write_text(json.dumps(original, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Saved {len(cells)} executed comparison cells; preserved {len(prefix)} other cells")
