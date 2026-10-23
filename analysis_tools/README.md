# Analysis Tools / 回测分析工具

`yq_analysis` 读取因子回测与策略模拟结果，生成收益、超额收益、IC、换手及账户诊断表，并绘制分组和策略对比图。它不执行因子计算，也不替代回测引擎。

`yq_analysis` reads factor backtests and strategy simulations, builds return/excess-return, IC, turnover and account diagnostics, and plots portfolio or strategy comparisons. It does not calculate factors or replace the backtest engine.

## 安装 / Installation

在仓库根目录执行；绘图需要可选的 Matplotlib 依赖。
Run from the repository root; plotting requires the optional Matplotlib dependency.

```bash
python -m pip install -e "./analysis_tools[plot]"
```

## 因子回测 / Factor backtests

先用 Rust backtest 生成结果。读取根目录和因子 ID 必须与该次回测匹配。
Generate results using Rust backtest first. Match the input root and factor ID to that run.

```python
from yq_analysis.io import load_backtest_result
from yq_analysis.report import make_backtest_report
from yq_analysis.plots import plot_return_summary

result = load_backtest_result("data/backtest/stock/daily", "ret20")
returns = result["returns"]
ic = result["ic"]
ic_main = (
    ic.query("horizon == 1")
    if ic is not None and "horizon" in ic.columns
    else ic
)
report = make_backtest_report(
    returns, ic_main, result["factor_stats"], periods_per_year=240
)
print(report["portfolio_total"])
print(report["excess_total"])
print(report["ic"])
fig = plot_return_summary(
    returns, groups=10, ic=ic,
    barra_exposure=result.get("barra_exposure"),
    index_group_returns=result.get("index_group_returns"),
    save=True,
)
```

主 IC 报告使用 horizon=1，完整 IC 表可用于衰减图，避免把不同 horizon 当作额外日期样本。`groups` 应与回测一致；缺少可选诊断表时不要假设每个图面板都有数据。

Use horizon=1 for the main IC report and the full IC table for decay plots, rather than treating horizons as extra dates. Match `groups` to the backtest. Optional diagnostic tables may be absent.

## 策略分析 / Strategy analysis

```python
from yq_analysis.io import load_strategy_result
from yq_analysis.report import make_strategy_report
from yq_analysis.plots import plot_strategy_summary

holdings = load_strategy_result("data/strategy/stock", "strategy_001")
report = make_strategy_report(
    holdings, initial_cash=100000, periods_per_year=240
)
print(report["portfolio_total"])
print(report["diagnostics"])
fig = plot_strategy_summary(report, save=True)
```

`initial_cash` 必须匹配策略配置。先用完整账户历史计算收益与对账，再截取展示区间。多策略比较使用 `plot_strategy_comparison`，各报告应具有相同观察日期。

Match `initial_cash` to the strategy configuration. Calculate returns and reconciliation on the full account history before slicing the display interval. Use `plot_strategy_comparison` for reports with matching observation dates.

## 输入与输出 / Inputs and outputs

默认因子输入位于 `data/backtest/stock/daily/{factor_id}/`，包括 `returns.parquet`、`ic.parquet`、`factor_stats.parquet` 及可选持仓、行业、Barra、指数内收益表。策略输入为对应目录的 `holdings.parquet`。

Default factor inputs live under `data/backtest/stock/daily/{factor_id}/`: returns, IC, factor statistics and optional holdings, industry, Barra and in-index returns. Strategy inputs use their `holdings.parquet`.

报表以 Python 字典/DataFrame 返回；保存图像通常位于 `analysis_tools/plots/`，不进入 Git。可用 `dpi` 等绘图参数控制输出。

Reports return Python dictionaries/DataFrames. Saved plots normally live under `analysis_tools/plots/` and remain untracked; plotting options such as `dpi` control output.

## 解释边界 / Interpretation

- 默认年化交易日数为 240，无风险收益率为 0；确认这与研究假设一致。
  Defaults use 240 annual periods and a zero risk-free rate; verify these assumptions.
- 覆盖率、收益和 IC 需要有效样本；空表不代表零收益。先检查日期、列名和上游因子是否非空。
  Coverage, returns and IC require valid samples. Empty tables do not imply zero returns; check dates, schema and upstream factor coverage.
- 策略收益已包含模拟成交规则及成本，不能简单加回成本来替代重新回测；账户模型不包含外部资金流。
  Strategy returns include execution rules and costs. Adding costs back is not a replacement for a rerun; external account cash flows are not modeled.
- 按全样本统计选择 long 侧属于事后分析，不能直接作为当时可用的交易信号。
  Choosing the long side using full-sample statistics is retrospective analysis, not a contemporaneously available signal.

回测参数见 [Rust 引擎](../factor_engine/README.md)，策略口径见[策略开发文档](../factor_engine/docs/STRATEGY_README.md)。
See the [Rust engine](../factor_engine/README.md) for backtest commands and the [strategy guide](../factor_engine/docs/STRATEGY_README.md) for execution semantics.
