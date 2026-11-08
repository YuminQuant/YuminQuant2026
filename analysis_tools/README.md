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

### 股票因子总报告 / Stock factor review

从项目根目录运行以下命令，增量更新现有 `data/reports/factor_review_20160101.html`。
首次执行登记已有本地文件，不用历史文件覆盖旧报告；之后新增或替换完整的五份
Parquet 结果即可更新。删除曾登记的因子目录后，下次运行会从报告移除该因子。
不读取因子元数据来决定增删，也不会删除回测文件。

Run from the project root. The existing HTML is the template and initial baseline.
Subsequent runs add or refresh complete local backtests and drop previously observed
factor directories that have been deleted. Use comparable backtest settings; the
script does not run the backtest or infer its settings.

```powershell
python scripts/chn_stock_factor_analysis.py
python scripts/chn_stock_factor_analysis.py --dry-run
python scripts/chn_stock_factor_analysis.py --refresh some_factor
python scripts/chn_stock_factor_analysis.py --remove some_factor
```

`--remove` 只修改报告，并记住排除项；用 `--refresh` 可恢复。首次运行前刚覆盖的
已有因子也可用 `--refresh` 强制结算。新因子可用 `--category "factor_id=分类"`
指定展示分类。相关性仅重算本地收益与报告版本一致的因子对；数据不齐时，
报告会说明覆盖范围。脚本保留原有曲线文件，新曲线写入独立子目录。

## 期货因子评价 / Futures factor evaluation

先用 Rust 期货分钟回测生成摘要。默认的过去五日阈值模式写入 `data/backtest/future/trailing-5`，显式选择单日模式时写入 `data/backtest/future`。读取、结算和绘图可以分别调用。

```python
from yq_analysis.future_io import load_future_backtest_result, load_future_sectors
from yq_analysis.future_report import make_future_backtest_report, compare_future_sector_returns
from yq_analysis.future_plots import plot_future_return_summary, plot_future_sector_comparison

sectors = load_future_sectors("factor_engine/docs/FUTURE_SECTORS.md")
daily = load_future_backtest_result("data/backtest/future", "yq_001")
trailing = load_future_backtest_result("data/backtest/future/trailing-5", "yq_001")
report = make_future_backtest_report(daily["summary"], sectors)
rolling = make_future_backtest_report(trailing["summary"], sectors, direction=report["direction"])
detail = plot_future_return_summary(report, title="yq_001 | Daily thresholds")
curves = compare_future_sector_returns(
    {"Daily": report["daily_product"], "Trailing 5": rolling["daily_product"]}, sectors,
)
comparison = plot_future_sector_comparison(curves, title="yq_001 | Threshold comparison")
```

对比采用同一品种 IC 方向，只使用两种模式都有有效双端收益的品种日。板块内对品种等权，日度值按交易日算术累加。应使用同一因子、同一标签和同一回测区间。

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
