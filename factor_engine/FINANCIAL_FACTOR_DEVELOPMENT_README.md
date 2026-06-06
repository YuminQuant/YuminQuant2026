# Financial Factor Development / 财务因子开发说明

This document defines the point-in-time financial statement workflow used by
`factor_engine`. It applies to normal stock factors and Barra exposure
generation.

本文档定义 `factor_engine` 的 PIT 财报取数与财务因子开发流程。该规则同时适用于普通股票因子和 Barra 暴露生成。

## Unified PIT Framework / 统一 PIT 框架

All financial statement consumers must use `PitFinancialData` with
`ReportTypePreference`. Do not add new factor-specific statement maps or use the
removed Barra `StatementData` style.

所有财报消费方都必须使用 `PitFinancialData` 和 `ReportTypePreference`。不要新增因子私有的财报 map，也不要恢复已经移除的 Barra `StatementData` 写法。

Recommended pattern / 推荐写法：

```rust
let income = PitFinancialData::from_table(
    data.daily(DatasetId::StockIncome)?,
    &["revenue", "n_income_attr_p"],
    ReportTypePreference::income_single_quarter(),
)?;

let balance = PitFinancialData::from_table(
    data.daily(DatasetId::StockBalanceSheet)?,
    &["total_cur_assets", "total_cur_liab"],
    ReportTypePreference::balance_sheet_consolidated(),
)?;
```

Useful helpers / 常用接口：

- `record_for_end_date(ts_code, trade_date, end_date)`: PIT-safe row for a report period. / 指定报告期的 PIT 安全记录。
- `latest_quarter_end_date(ts_code, trade_date)`: latest disclosed quarter. / 当前交易日可见的最新季度。
- `ttm_sum(ts_code, trade_date, column)`: latest available four-quarter sum. / 最新可得四季度 TTM 汇总。
- `ttm_sum_for_end_date(ts_code, trade_date, end_date, column)`: four-quarter sum anchored at a report period. / 以指定报告期为锚点的 TTM 汇总。
- `latest_annual_value(...)`, `latest_annual_end_date(...)`, `annual_value_for_end_date(...)`, `annual_values(...)`: annual helpers used by Barra and financial factors. / Barra 和财务因子共用的年度数据接口。

## Report Type / 报表类型

Financial statement rows contain `report_type`.

财务报表行包含 `report_type` 字段。

| Code / 代码 | Type / 类型 | Description / 说明 |
| --- | --- | --- |
| 1 | Consolidated / 合并报表 | Latest listed-company consolidated statement, default full-period statement. / 上市公司最新合并报表，默认累计报表。 |
| 2 | Single-quarter consolidated / 单季合并 | Single-quarter consolidated report. / 单一季度合并报表。 |
| 3 | Adjusted single-quarter consolidated / 调整单季合并 | Adjusted single-quarter consolidated report, preferred when available. / 调整后的单季合并报表，若存在则优先使用。 |
| 4 | Adjusted consolidated / 调整合并报表 | Current-year disclosure of prior-year comparable report data. / 本年度公布上年同期的财务报表数据。 |
| 5 | Pre-adjustment consolidated / 调整前合并报表 | Original consolidated report retained after data revision. / 数据变更后保留的调整前原始数据。 |
| 6 | Parent-company statement / 母公司报表 | Parent-company financial statement. / 母公司财务报表数据。 |
| 7 | Parent-company single-quarter / 母公司单季表 | Parent-company single-quarter statement. / 母公司单季度表。 |
| 8 | Adjusted parent single-quarter / 母公司调整单季表 | Adjusted parent-company single-quarter statement. / 母公司调整后的单季表。 |
| 9 | Adjusted parent statement / 母公司调整表 | Current-year disclosure of prior-year parent-company comparable data. / 本年度公布上年同期的母公司报表数据。 |
| 10 | Pre-adjustment parent statement / 母公司调整前报表 | Original parent-company statement retained before adjustment. / 母公司调整前原始数据。 |
| 11 | Pre-adjustment parent consolidated / 母公司调整前合并报表 | Original parent-company consolidated data retained before adjustment. / 母公司调整前合并报表原数据。 |
| 12 | Pre-adjustment parent statement / 母公司调整前报表 | Original parent-company data retained before adjustment. / 母公司报表变更前保留的原数据。 |

Current defaults / 当前默认优先级：

- Single-quarter income data / 利润表单季数据：`ReportTypePreference::income_single_quarter() = [3, 2]`
- Consolidated balance sheet and annual-style data / 合并资产负债表和年度口径数据：`ReportTypePreference::balance_sheet_consolidated() = [1, 4]`
- Generic consolidated data / 通用合并报表口径：`ReportTypePreference::consolidated() = [1, 4]`

Within the same `ts_code + end_date + report_type`, only rows with
`f_ann_date/ann_date <= trade_date` are visible. Versions are sorted by
`disclosure_date` descending, then `update_flag` descending.

同一 `ts_code + end_date + report_type` 下，只允许使用 `f_ann_date/ann_date <= trade_date` 的记录。版本按 `disclosure_date` 倒序、`update_flag` 倒序选择。

## Storage And Loading / 存储与读取

Statement parquet files are stored by `ann_date` year:

财报 parquet 按 `ann_date` 年份存储：

```text
data/stock_data/income/{year}.parquet
data/stock_data/balancesheet/{year}.parquet
data/stock_data/cashflow/{year}.parquet
```

Factor specs should request only needed value columns. The loader automatically
adds PIT key/version columns: `ts_code`, `ann_date`, `f_ann_date`, `end_date`,
`report_type`, and `update_flag`.

因子 spec 只应请求实际需要的数值列。loader 会自动补充 PIT 选版本所需的键列和版本列：`ts_code`、`ann_date`、`f_ann_date`、`end_date`、`report_type`、`update_flag`。

Example / 示例：

```rust
DataRequest::financial_quarters(
    DatasetId::StockIncome,
    &["revenue", "n_income_attr_p"],
    8,
)
```

The loader derives conservative announcement-year windows from the requested
quarter count and date batch, then reuses yearly tables through
`DisclosureTableCache`.

loader 会根据请求的季度数和 date batch 推导保守的公告年份窗口，并通过 `DisclosureTableCache` 复用年度表。

## Missing Values / 缺失值处理

Financial similarity factors use per-metric percentile ranks. For listed
non-BJ stocks, missing metric ranks are filled with `0` after percentile
ranking. Stocks that are not present in the daily PV panel are not filled and do
not output values. A stock with all ten financial dimensions missing still has a
zero vector and is excluded from cosine similarity.

财务相似度因子按单个指标做截面分位数标准化。对在市且非 BJ 的股票，单个指标缺失会在分位数标准化后填 `0`。不在日频 PV 面板中的股票不填充、不输出。若 10 个财务维度全部缺失，该股票为零向量，不参与余弦相似度。

This differs from many Barra CNE6 exposures, which may use industry/global
fills before standardization.

这与部分 Barra CNE6 暴露不同；Barra 中一些暴露会在标准化前做行业或全局填充。

## BJ Stock Rule / 北交所股票规则

Unless explicitly stated otherwise, financial cross-sectional peer/network
factors exclude `.BJ` stocks.

除非另有说明，财务类截面 peer/network 因子剔除 `.BJ` 股票。

- `.BJ` stocks do not participate in financial cross-section transforms. / `.BJ` 不参与财务指标截面处理。
- `.BJ` stocks do not participate in similarity matrices or peer networks. / `.BJ` 不参与相似度矩阵或 peer network。
- `.BJ` stocks do not enter regression or neutralization inputs. / `.BJ` 不进入回归或中性化输入。
- `.BJ` stocks should not produce final factor values. / `.BJ` 最终不输出有效因子值。

## Multi-output Financial Providers / 多输出财务 Provider

When several financial factors share expensive setup, use a multi-output
provider. The engine groups selected factors by `compute_provider_key()` and
calls `compute_many(requested_ids, ...)` once per provider. Single-output legacy
factors continue to use the default `compute()` path.

多个财务因子共享高成本前置计算时，应使用多输出 provider。engine 会按 `compute_provider_key()` 对已选择因子分组，并对每个 provider 调用一次 `compute_many(requested_ids, ...)`。普通单输出因子继续走默认 `compute()` 路径。

Development layout / 推荐结构：

```text
factor/common/financial_similarity.rs
factor/chn_stock/daily/f_momentum_80pec.rs
factor/chn_stock/daily/link_new.rs
```

The common provider should build PIT financial metrics, cross-sectional
transforms, and similarity/network state once, then compute only requested
branches.

common provider 应一次性构造 PIT 财务指标、截面处理结果和相似度/网络状态，然后只计算 `requested_ids` 请求的输出分支。

## Current Financial Similarity Factors / 当前财务相似度因子

The current provider outputs / 当前 provider 输出：

- `f_momentum_80pec`
- `link_new`

Required tags / 必需标签：

```text
XYZQ, financial, fundamental, pit, f_momentum, cs_network,
neutralize, barra, size, sector, daily
```

Both factors use PIT statements, percentile-rank standardization for ten
financial metrics, `.BJ` exclusion, and final Barra `SIZE` + SW sector
neutralization.

两个因子均使用 PIT 财报、10 个财务指标截面分位数标准化、剔除 `.BJ`，并最终做 Barra `SIZE` + 申万一级行业中性化。
