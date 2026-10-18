# yq_analysis / Python 回测结果分析工具

`analysis_tools` 是一个轻量 Python 工具包，用来读取 Rust backtest 输出、生成绩效/IC 表格、绘制分组收益图。它不只服务 ML alpha，未来策略收益序列也可以复用其中的指标函数。

`analysis_tools` is a lightweight Python helper package for reading Rust backtest outputs, building performance/IC reports, and plotting group return summaries. It is not tied to ML alpha.

## 策略检验 / Strategy Diagnostics

`test.ipynb` 的“策略检验”部分仅保留 Top50 基准结果。策略研究统一在不受 Git 跟踪的 `strategy.ipynb` 中进行，当前比较 Top50、排名 11–50、排名 21–50、Top50 无涨跌停成交限制四组。不绘制 benchmark 或年度收益。左上面板将账户净值和回撤放在同一时间轴，第一行第二列显示 PnL，下方保留资金利用率、换手率、统一色阶的月度收益。

使用 `plot_strategy_comparison({"Top 10": report10, "Top 20": report20, "Top 50": report50})` 绘制对比；所有报告需要有相同的观察日期，最多支持八组，超过四组时月度面板分两行。`diagnose_stock_cash()` 用前一交易日信号和当天开盘/悲观涨停标志解释现金较高的持有期，区分 Top N 未成交份额与按股票预算进行整手取整后的余额。该诊断应传入完整、未截断的回测报告。

```python
from yq_analysis.io import load_strategy_result, load_index_returns
from yq_analysis.report import make_strategy_report
from yq_analysis.plots import plot_strategy_summary

holdings = load_strategy_result("data/strategy/stock", "strategy_001")
benchmark = load_index_returns("data/index_data/daily", "000905.SH", 20200101, 20260424)
strategy_report = make_strategy_report(
    holdings, initial_cash=100000, periods_per_year=240,
    benchmark=benchmark, benchmark_name="000905.SH",
)
display(strategy_report["portfolio_total"].round(3))
display(strategy_report["portfolio_by_year"].round(3))
fig = plot_strategy_summary(strategy_report, save=True)
```

输出包括 `daily`（日收益、净值、回撤、仓位、现金、换手、费用）、`portfolio_total`、`portfolio_by_year`、`monthly_returns`、`drawdowns`、`activity`、`latest_positions`、`trades`、`diagnostics`。

- 股票权益对账包含 `现金 + 持仓市值 + dividend_receivable`；应收股利不能用于买入，`dividend_income_cum` 展示累计税前现金股利。旧文件缺少这两列时按零读取。
- 账户净资产为 `initial_cash + account_pnl`，第一日收益包含相对初始资金的盈亏。筛选日期前先计算收益，保留筛选首日相对前一日净资产的收益。
- 分钟输出按日取最后估值，成交计数按日求和。Sharpe 默认无风险利率为零、年化 240 日；分年实际收益与年化外推分列展示。
- 回撤包含初始净值 1，因此首日亏损不会被遗漏。回撤表 `recovery=NaT` 表示尚未恢复，`underwater_days` 表示低于前高的交易日数。
- 相对净值为策略净值除以指数净值，`Relative wealth` 使用对应复合相对收益；指数按同日 close/pre_close 对齐，缺失时禁用比较，绝不补零。
- 股票双边换手为买卖成交金额之和 / 前一日净资产。成交价格在 Rust 输出中已取两位小数，因此换手是近似值。缺失或无效成交明细显示 NaN；期货缺逐笔合约规格时不推算换手。
- 累计成本由 `realized_pnl_cum - net_realized_pnl_cum` 还原，包含费用、税和滑点，收益已扣过这些成本。不能据此精确模拟免手续费策略，因为改变费用会影响后续仓位。
- 成交笔数是 fill 数量；不从这些记录推算完整往返交易胜率。资金追加/提取未建模，适用于当前引擎的无外部现金流账户。
- `initial_cash` 必须与该次回测一致；检查 `diagnostics` 中的账户、盈亏对账。可传入 `expected_dates` 检查缺失交易日。净资产非正、重复快照或损坏 JSON 会明确报错。

默认图片路径：`analysis_tools/plots/strategy/{strategy_id}.png`。

## 使用方式 / Usage

不安装时，把包目录加入 `sys.path`：

Without installing, add the package root to `sys.path`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("analysis_tools").resolve()))
```

读取回测结果 / Load a backtest result:

```python
from yq_analysis.io import load_backtest_result

result = load_backtest_result(r"data/backtest/stock/daily", "SIZE")
returns = result["returns"]
ic = result["ic"]
factor_stats = result["factor_stats"]
barra_exposure = result["barra_exposure"]
index_group_returns = result["index_group_returns"]
```

生成报表 / Build reports:

```python
from yq_analysis.report import make_backtest_report

# The backtest IC file may contain shifted IC decay rows for horizon 1..20.
# Use horizon == 1 for the main report table to avoid counting decay rows
# as extra observations.
ic_main = ic.query("horizon == 1") if ic is not None and "horizon" in ic.columns else ic

report = make_backtest_report(returns, ic_main, factor_stats, periods_per_year=240)
display(report["portfolio_total"])
display(report["portfolio_by_year"])
display(report["excess_total"])
display(report["excess_by_year"])
display(report["ic"])
```

绘图 / Plot:

```python
from yq_analysis.plots import plot_return_summary

fig = plot_return_summary(returns, groups=10, save=True)

fig = plot_return_summary(
    returns,
    groups=10,
    ic=ic,
    barra_exposure=barra_exposure,
    index_group_returns=index_group_returns,
    save=True,
)
```

传入 `barra_exposure`、`index_group_returns` 和 `ic` 时，默认图片使用新版横向布局，并自动加入 Barra 暴露、指数内超额和 IC 衰减诊断。
新版绘图规则 / Updated plot rules:

- 左侧三行依次为累计收益、累计超额收益、指数内 long 侧累计超额收益。
- 指数内 long 侧来自 `{factor_id}/index_group_returns.parquet`，固定展示 `000300.SH`、`000905.SH`、`000852.SH`；long 侧按全样本 RankIC 均值方向选择 `group_5` 或 `group_1`。
- 右侧为三行嵌套两列：第一行为年化收益、年化超额收益；第二行为 Barra 暴露和 Barra IC mean；第三行为换手率和 IC 衰减。
- IC 衰减图读取同一个 `ic` 表中的 `horizon=1..20`，每个 horizon 展示一根 Pearson `IC mean` 柱；RankIC decay 不再计算。最后两根柱追加近似 5d/20d IC：`sum(IC_1..IC_N) / sqrt(N)`，其中 `N=5/20`。
- Barra 柱状图只在绘图层缩写长名称，例如 `DIVIDEND_YIELD -> DY`、`LIQUIDITY -> LIQ`、`MOMENTUM -> MOM`、`VOLATILITY -> VOL`。
- 年化收益、年化超额收益、Barra IC mean、IC 衰减柱状图都会在柱子上标注两位小数。

Updated plot layout:

- Left column: cumulative return, cumulative excess return, and in-index long-side cumulative excess return.
- In-index curves come from `{factor_id}/index_group_returns.parquet` and cover `000300.SH`, `000905.SH`, and `000852.SH`. The long side is selected by full-sample mean RankIC: `group_5` when non-negative, `group_1` when negative.
- Right column uses three rows with two subplots per row: annual return and annual excess return; Barra exposure and Barra IC mean; turnover and IC decay.
- IC decay uses `horizon=1..20` rows from the same `ic` table and plots Pearson IC mean only; RankIC decay is no longer computed. The last two bars append approximate 5d/20d IC as `sum(IC_1..IC_N) / sqrt(N)` for `N=5/20`.
- Long Barra names are abbreviated only at plot time, for example `DIVIDEND_YIELD -> DY`, `LIQUIDITY -> LIQ`, `MOMENTUM -> MOM`, and `VOLATILITY -> VOL`.
- Bar charts annotate values with two decimals.

默认图片输出 / Default plot output:

```text
analysis_tools/plots/{factor_id}.jpg
```

`plot_return_summary(..., save=True)` 默认不覆盖 Matplotlib `savefig` 的 `dpi`；如需高清图片，可显式传入 `dpi=150`。
`plot_return_summary(..., save=True)` does not override Matplotlib's `savefig` DPI by default; pass `dpi=150` explicitly when a higher-resolution image is needed.

## 指标 / Metrics

收益指标默认 `periods_per_year=240`，无风险收益率为 `0`。

Return metrics default to `periods_per_year=240` and zero risk-free rate.

常见输出 / Common report columns:

```text
cumulative_return(%)
annual_return(%)
annual_volatility(%)
sharpe
max_drawdown(%)
calmar
win_rate(%)
mean_return_bp_per_1pct_turnover
turnover_mean(%)
```

IC 输出 / IC report:

```text
mean
std
ir
abs_mean
abs_ir
```

## 输入格式 / Input Shape

`load_backtest_result(root, factor_id)` 会读取：

`load_backtest_result(root, factor_id)` reads:

```text
{root}/{factor_id}/returns.parquet
{root}/{factor_id}/ic.parquet
{root}/{factor_id}/factor_stats.parquet
{root}/{factor_id}/barra_exposure.parquet
{root}/{factor_id}/index_group_returns.parquet
```

缺失文件返回 `None`，所以也可以只分析收益或只分析 IC。

Missing files return `None`, so returns-only or IC-only analysis is supported.

新增 backtest 诊断 / New backtest diagnostics:

- `{factor_id}/ic.parquet` 可能同时包含 `horizon=1..20`。这些 horizon 来自同一个 1d label 的横截面序列 shift，而不是不同收益期限 label。`plot_return_summary(..., ic=ic)` 会用这些 horizon 生成 Pearson IC 衰减图，并在最后追加近似 5d/20d IC 柱；`make_backtest_report(...)` 建议只传 `horizon == 1` 的主 IC，避免 observations 把 decay 行一起计入。
- `{factor_id}/index_group_returns.parquet` 是指数内五分组逐日收益，包含实际收益、指数 benchmark 收益和超额收益。绘图层只取按 RankIC 方向选出的 long 侧，并展示其累计超额收益。
- `{factor_id}/barra_exposure.parquet` 同时服务 Barra 暴露时间序列和 Barra IC mean 柱状图。

New backtest diagnostics:

- `{factor_id}/ic.parquet` may include `horizon=1..20`. These horizons are shifted cross-sections of the same 1d label, not different return-horizon labels. `plot_return_summary(..., ic=ic)` uses them for Pearson IC decay and appends approximate 5d/20d IC bars. For `make_backtest_report(...)`, pass only the main `horizon == 1` rows so the IC observation count is not inflated by decay rows.
- `{factor_id}/index_group_returns.parquet` stores daily in-index five-group returns, benchmark returns, and excess returns. The plotting layer selects the long side from mean RankIC direction and plots cumulative excess return.
- `{factor_id}/barra_exposure.parquet` feeds both the Barra exposure time series and the Barra IC mean bar chart.

Barra exposure rows use the `metric` column:

```text
barra_ic              daily cross-sectional Pearson IC between factor and CNE6 style exposure
barra_ic_mean         long-run mean of the daily Barra IC
long_group_exposure   selected long-side group exposure on rebalance dates
```

## 单指标函数 / Metric Functions

指标函数在 `yq_analysis.metrics` 中，一个指标一个函数，例如：

Metric functions live in `yq_analysis.metrics`, one function per metric:

```python
from yq_analysis.metrics import cumulative_return, annual_return, sharpe, max_drawdown

cumulative_return(returns["return"])
annual_return(returns["return"], periods_per_year=240)
sharpe(returns["return"], periods_per_year=240)
max_drawdown(returns["return"])
```

### Top50 研究工作流

1. `cargo build --release --offline --manifest-path factor_engine/Cargo.toml`
2. `python analysis_tools/scripts/run_top50_research.py`
3. `python analysis_tools/scripts/update_strategy_notebook.py`

第二步运行四组配置并保存配置快照、日志与文件指纹；第三步保留 `test.ipynb` 的因子部分，只刷新 Top50 结果，并在 `strategy.ipynb` 中执行四组比较。两个 notebook 都只展示结果，不加入实验结论或验证表，均由现有 `*.ipynb` 忽略规则排除。

剔除头部 10/20 只对应原 Top50 内的排名 11–50（40 只）及 21–50（30 只），按剩余股票约等权分配，不从排名 50 之后补位。无涨跌停组仅关闭价格涨跌停成交限制，整手、费用、可用资金、真实开盘价和分红送转记账保持一致。

NAV 与 PnL 来自同一账户权益，已含现金分红应收/到账和送转股，不能再次叠加现金股利。现金股利金额单列仅用于展示。旧多组归因脚本保留为源代码，旧回测、归因输出和临时实验产物已移至回收站，不再作为当前研究输入。
