# yq_analysis / Python 回测结果分析工具

`analysis_tools` 是一个轻量 Python 工具包，用来读取 Rust backtest 输出、生成绩效/IC 表格、绘制分组收益图。它不只服务 ML alpha，未来策略收益序列也可以复用其中的指标函数。

`analysis_tools` is a lightweight Python helper package for reading Rust backtest outputs, building performance/IC reports, and plotting group return summaries. It is not tied to ML alpha.

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

# The backtest IC file may contain IC decay rows for horizon 1/5/20.
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
- 指数内 long 侧来自 `index_group_returns/{factor_id}.parquet`，固定展示 `000300.SH`、`000905.SH`、`000852.SH`；long 侧按全样本 RankIC 均值方向选择 `group_5` 或 `group_1`。
- 右侧为三行嵌套两列：第一行为年化收益、年化超额收益；第二行为 Barra 暴露和 Barra IC mean；第三行为换手率和 IC 衰减。
- IC 衰减图读取同一个 `ic` 表中的 `horizon=1/5/20`，每个 horizon 展示 `IC mean` 和 `RankIC mean` 两根柱。
- Barra 柱状图只在绘图层缩写长名称，例如 `DIVIDEND_YIELD -> DY`、`LIQUIDITY -> LIQ`、`MOMENTUM -> MOM`、`VOLATILITY -> VOL`。
- 年化收益、年化超额收益、Barra IC mean、IC 衰减柱状图都会在柱子上标注两位小数。

Updated plot layout:

- Left column: cumulative return, cumulative excess return, and in-index long-side cumulative excess return.
- In-index curves come from `index_group_returns/{factor_id}.parquet` and cover `000300.SH`, `000905.SH`, and `000852.SH`. The long side is selected by full-sample mean RankIC: `group_5` when non-negative, `group_1` when negative.
- Right column uses three rows with two subplots per row: annual return and annual excess return; Barra exposure and Barra IC mean; turnover and IC decay.
- IC decay uses `horizon=1/5/20` rows from the same `ic` table and plots both IC mean and RankIC mean for each horizon.
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
{root}/returns/{factor_id}.parquet
{root}/ic/{factor_id}.parquet
{root}/factor_stats/{factor_id}.parquet
{root}/barra_exposure/{factor_id}.parquet
{root}/index_group_returns/{factor_id}.parquet
```

缺失文件返回 `None`，所以也可以只分析收益或只分析 IC。

Missing files return `None`, so returns-only or IC-only analysis is supported.

新增 backtest 诊断 / New backtest diagnostics:

- `ic/{factor_id}.parquet` 可能同时包含 `horizon=1/5/20`。`plot_return_summary(..., ic=ic)` 会用这些 horizon 生成 IC 衰减图；`make_backtest_report(...)` 建议只传 `horizon == 1` 的主 IC，避免 observations 把 1/5/20 三组 decay 行一起计入。
- `index_group_returns/{factor_id}.parquet` 是指数内五分组逐日收益，包含实际收益、指数 benchmark 收益和超额收益。绘图层只取按 RankIC 方向选出的 long 侧，并展示其累计超额收益。
- `barra_exposure/{factor_id}.parquet` 同时服务 Barra 暴露时间序列和 Barra IC mean 柱状图。

New backtest diagnostics:

- `ic/{factor_id}.parquet` may include `horizon=1/5/20`. `plot_return_summary(..., ic=ic)` uses these rows for IC decay. For `make_backtest_report(...)`, pass only the main `horizon == 1` rows so the IC observation count is not inflated by decay rows.
- `index_group_returns/{factor_id}.parquet` stores daily in-index five-group returns, benchmark returns, and excess returns. The plotting layer selects the long side from mean RankIC direction and plots cumulative excess return.
- `barra_exposure/{factor_id}.parquet` feeds both the Barra exposure time series and the Barra IC mean bar chart.

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
