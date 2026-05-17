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
```

生成报表 / Build reports:

```python
from yq_analysis.report import make_backtest_report

report = make_backtest_report(returns, ic, factor_stats, periods_per_year=240)
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
```

默认图片输出 / Default plot output:

```text
analysis_tools/plots/{factor_id}.jpg
```

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
```

缺失文件返回 `None`，所以也可以只分析收益或只分析 IC。

Missing files return `None`, so returns-only or IC-only analysis is supported.

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
