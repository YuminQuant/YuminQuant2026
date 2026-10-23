# Forecast Bias / 分析师预测偏离度

本工具读取本地分析师年度预测和已披露实际财报，研究预测误差与预测跨度、市值和行业的关系。不下载数据，不修改一致预期生产表，也不生成交易因子。

This tool compares local annual analyst forecasts with disclosed financial results, grouping errors by forecast horizon, company size and industry. It neither downloads data nor modifies consensus tables or produces trading factors.

## 安装与使用 / Setup and usage

需要 Python 3.11+、pandas、numpy、pyarrow、matplotlib。从项目根目录运行，数据路径默认读取 `config.toml`。

Requires Python 3.11+, pandas, numpy, pyarrow and matplotlib. Run from the repository root; data paths default to `config.toml`.

```bash
python scripts/forecast_bias/analyze.py --start-date 20220101 --end-date 20231231 --metric np --forecast-year 2023 --as-of 20260424
python scripts/forecast_bias/analyze.py --start-date 20220101 --end-date 20231231 --metric op_rt --industry sw --as-of 20260424 --save-observations
```

| 参数 / Option | 含义 / Meaning |
|---|---|
| `--start-date / --end-date` | 研报日期闭区间，不是预测财年 / Inclusive report dates, not forecast fiscal years |
| `--metric` | `np` 净利润、`op_rt` 收入、`eps` 每股收益 / Net profit, revenue or EPS |
| `--forecast-year` | 可选一个或多个财年；只使用年度 Q4 预测 / Optional fiscal years; annual Q4 forecasts only |
| `--as-of` | 实际财报可见截止日，默认当天且不得早于 end-date / Actual-disclosure cutoff; defaults to today and cannot precede end-date |
| `--industry` | `ci` 中信一级（默认）或 `sw` 申万一级 / CITIC L1 (default) or SW L1 |
| `--size-bins` | 市值分组数，默认 10 / Size groups, default 10 |
| `--min-actual-abs` | 实际值绝对值下限，默认 0 / Minimum absolute actual value, default 0 |
| `--config / --data-root` | 配置文件或数据根目录 / Configuration or data root |
| `--output-dir` | 必须尚不存在的结果目录 / A new, non-existing output directory |
| `--save-observations` | 保存逐条评价明细 / Save observation-level results |

其他选项见 `python scripts/forecast_bias/analyze.py --help`。
See `--help` for additional options.

## 计算口径 / Methodology

偏离度为 `abs(forecast - actual) / abs(actual)`，另统计有符号误差 `(forecast - actual) / abs(actual)`。零实际值排除，负值可参与；不截尾或用 epsilon 替换分母。接近零的实际值可能产生极大误差，应结合中位数、分位数和样本数解读。

Absolute error is `abs(forecast - actual) / abs(actual)`; signed error retains the numerator's sign. Zero actuals are excluded, negative actuals are allowed, and no clipping or epsilon denominator is applied. Inspect medians, quantiles and counts because near-zero actuals can create extreme errors.

预测 `np/op_rt/eps` 分别对应实际 `n_income_attr_p/revenue/basic_eps`。实际金额从元换算为万元，EPS 不换算，也不额外调整股本变化。实际值取目标年报最早披露的合并报表（类型 1/4），日期用 `f_ann_date`，缺失才用 `ann_date`；同日优先类型 1，再优先较高 update_flag。首次披露缺值不会用后续修订补齐。

Forecasts map to attributable net profit, revenue and basic EPS. Monetary actuals are converted from yuan to ten-thousand yuan; EPS is not adjusted for share-count changes. Actuals use the earliest consolidated annual disclosure (types 1/4), preferring type 1 and higher update flags on ties. Missing first-disclosure values are not replaced with later revisions.

只评价早于实际披露日的研报，同日保守排除。相同股票、报告日期、机构、作者、标题、财年的记录保留最早更新时间版本；不同日期预测仍全部等权参与，不等同于一致预期的每机构最新值聚合。

Reports must precede actual disclosure; same-day reports are excluded. Duplicate report identities retain the earliest update, while different report dates remain equally weighted. This differs from latest-per-institution consensus aggregation.

规模使用报告日或之前的市值截面，行业使用报告日历史分类；跨度为实际公告月份与报告月份之差。各分组使用各自有效样本，并提供按预测财年拆分的统计。

Size uses market data on or before the report date; industry uses historical membership. Horizon is the month difference between actual disclosure and report publication. Each grouping uses its available observations, with fiscal-year breakdowns.

## 输出与限制 / Outputs and limitations

默认输出至本目录 `output/<指标>_<日期>_<时间戳>/`，包括比较图、`bias_by_*.csv`、`run_metadata.json`，以及可选 `observations.parquet`。运行信息记录过滤情况；缺少成熟实际值的预测不参与评价。

Outputs go to a timestamped `output/` directory: comparison plots, grouped CSVs, run metadata and optional observations. Review exclusion counts; forecasts without mature actuals are omitted.

这是事后描述性研究，不是严格 PIT 交易信号。供应商 `create_time` 不等同于首次可见时间，本地文件无法恢复被覆盖的历史版本；实际披露月份也是事后信息。未实现 ST/新股及研报类别筛选，也不生成分析师权重。

This is retrospective descriptive research, not a strictly PIT trading signal. Vendor update timestamps do not establish first availability, overwritten history cannot be recovered, and actual-disclosure months are hindsight information. The tool does not filter ST/new listings or report categories, or generate analyst weights.
