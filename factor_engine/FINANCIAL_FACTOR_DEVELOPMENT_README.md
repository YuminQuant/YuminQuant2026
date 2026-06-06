# Financial Factor Development / 财务因子开发说明

This document records the current financial-factor data access rules in
`factor_engine`, with a focus on point-in-time financial statements, report type
selection, column projection, missing-value behavior, and the recommended
development workflow for future financial factors.

本文档记录 `factor_engine` 当前财务因子的取数与开发规则，重点覆盖 PIT
财报口径、`report_type` 选择、列投影、缺失值处理，以及后续财务因子的推荐开发流程。

## Report Type / 报表类型

Financial statement rows contain `report_type`. The current meaning is:

财务报表行包含 `report_type` 字段，目前含义如下：

| Code / 代码 | Type / 类型 | Description / 说明 |
| --- | --- | --- |
| 1 | Consolidated / 合并报表 | Latest listed-company consolidated report, default full-period statement. / 上市公司最新合并报表，默认累计报表。 |
| 2 | Single-quarter consolidated / 单季合并 | Single-quarter consolidated report. / 单一季度合并报表。 |
| 3 | Adjusted single-quarter consolidated / 调整单季合并表 | Adjusted single-quarter consolidated report, preferred when available. / 调整后的单季合并报表，若存在优先使用。 |
| 4 | Adjusted consolidated / 调整合并报表 | Current-year disclosure of prior-year comparable report data, report period is prior year. / 本年度公布上年同期的财务报表数据，报告期为上年度。 |
| 5 | Pre-adjustment consolidated / 调整前合并报表 | Original consolidated report retained after data revision. / 数据发生变更后保留的调整前原始数据。 |
| 6 | Parent-company statement / 母公司报表 | Parent-company financial statement. / 母公司财务报表数据。 |
| 7 | Parent-company single-quarter / 母公司单季表 | Parent-company single-quarter statement. / 母公司单季度表。 |
| 8 | Adjusted parent single-quarter / 母公司调整单季表 | Adjusted parent-company single-quarter statement. / 母公司调整后的单季表。 |
| 9 | Adjusted parent statement / 母公司调整表 | Current-year disclosure of prior-year parent-company comparable report data. / 本年度公布上年同期的母公司财务报表数据。 |
| 10 | Pre-adjustment parent statement / 母公司调整前报表 | Original parent-company statement retained before adjustment. / 母公司调整之前保留的原始数据。 |
| 11 | Pre-adjustment parent consolidated / 母公司调整前合并报表 | Original parent-company consolidated data retained before adjustment. / 母公司调整之前的合并报表原数据。 |
| 12 | Pre-adjustment parent statement / 母公司调整前报表 | Original parent-company data retained before adjustment. / 母公司报表发生变更前保留的原数据。 |

## Current Preference Rules / 当前优先级规则

The PIT helper `PitFinancialData` accepts a `ReportTypePreference` and searches
types in order. It then chooses the newest version whose disclosure date is not
after the target trade date.

PIT helper `PitFinancialData` 会接收一个 `ReportTypePreference`，按优先级顺序查找
`report_type`。在命中类型后，再选择 `disclosure_date <= trade_date` 的最新版本。

Current defaults:

当前默认规则：

- Income single-quarter data / 利润表单季数据：
  `ReportTypePreference::income_single_quarter() = [3, 2]`
  - Prefer adjusted single-quarter consolidated report.
  - Fall back to normal single-quarter consolidated report.
  - 优先取调整单季合并表，缺失时回退到普通单季合并表。
- Balance sheet consolidated data / 资产负债表合并数据：
  `ReportTypePreference::balance_sheet_consolidated() = [1, 4]`
  - Prefer latest consolidated report.
  - Fall back to adjusted consolidated report.
  - 优先取最新合并报表，缺失时回退到调整合并报表。

Version selection inside the same `ts_code + end_date + report_type` group:

同一 `ts_code + end_date + report_type` 下的版本选择：

1. Keep only rows whose effective disclosure date is not after `trade_date`.
   The effective disclosure date uses `f_ann_date` first, falling back to
   `ann_date`.
2. Sort by `disclosure_date` descending, then `update_flag` descending.
3. Use the first available version.

1. 只使用 `f_ann_date/ann_date <= trade_date` 的记录；优先使用 `f_ann_date`，缺失时使用
   `ann_date`。
2. 按 `disclosure_date` 倒序、`update_flag` 倒序排序。
3. 取排序后的第一条可用版本。

## PIT Access Pattern / PIT 取数模式

Financial factors should use `PitFinancialData::from_table(...)` rather than
reading statement rows directly.

财务因子应通过 `PitFinancialData::from_table(...)` 读取财报，不应直接绕过 PIT 版本选择。

Recommended pattern:

推荐模式：

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

Useful helper methods:

常用 helper：

- `record_for_end_date(ts_code, trade_date, end_date)`
  - Fetch a PIT-safe record for a specific report period.
  - 按指定报告期取 PIT 安全版本。
- `latest_quarter_end_date(ts_code, trade_date)`
  - Find the latest disclosed quarter end date available at `trade_date`.
  - 找到 `trade_date` 当时已经可见的最新报告期。
- `ttm_sum_for_end_date(ts_code, trade_date, end_date, column)`
  - Sum the latest four quarters ending at `end_date`.
  - 以指定报告期为锚点，汇总最近四个季度。

## Column Projection / 列投影

Factor specs should request only the value columns needed by the factor. The
loader still reads required key/version columns for PIT selection, such as
`ts_code`, `ann_date`, `f_ann_date`, `end_date`, `report_type`, and
`update_flag`.

因子 spec 应只请求该因子需要的数值字段。loader 仍会读取 PIT 选版本必须的键列和版本列，例如
`ts_code`、`ann_date`、`f_ann_date`、`end_date`、`report_type`、`update_flag`。

Example:

示例：

```rust
DataRequest::financial_quarters(
    DatasetId::StockIncome,
    &["revenue", "n_income_attr_p"],
    8,
)
```

Do not request broad statement columns unless the factor genuinely needs them.

除非因子确实需要，否则不要宽表式读取大量财报字段。

## Missing Values / 缺失值处理

The first financial-similarity factors use a strict complete-vector rule:

第一版财务相似度因子采用严格完整向量口径：

- If any required financial item is missing, non-finite, or has an invalid
  denominator, the stock-date cannot build the full financial vector.
- That stock-date is excluded from cross-sectional percentile-rank
  standardization, F-Link similarity, peer calculations, regression input,
  neutralization input, and final output.
- No industry median fill, global median fill, or zero fill is applied in this
  factor family.

- 任一必需财务科目缺失、非有限、分母非法时，该股票该日无法构造完整财务向量。
- 该股票该日不参与截面分位数标准化、F-Link 相似度、peer 计算、回归输入、中性化输入和最终输出。
- 这一类因子暂不做行业中位数填充、全局中位数填充或置零。

This is different from Barra CNE6 generation, where many exposures may use
industry/global filling before standardization.

这与 Barra CNE6 生成层不同；Barra 暴露中很多字段会在标准化前做行业或全局填充。

## BJ Stock Rule / 北交所股票规则

Unless explicitly stated otherwise, financial cross-sectional peer/network
factors should exclude `.BJ` stocks:

除非另有明确说明，财务类截面 peer/network 因子应剔除 `.BJ` 股票：

- `.BJ` stocks do not participate in financial cross-section transforms.
- `.BJ` stocks do not participate in similarity matrices or peer networks.
- `.BJ` stocks do not enter regression or neutralization inputs.
- `.BJ` stocks should not produce final factor values.

- `.BJ` 股票不参与财务指标截面处理。
- `.BJ` 股票不参与相似度矩阵或 peer network。
- `.BJ` 股票不进入回归和中性化输入。
- `.BJ` 股票最终不输出有效因子值。

## Multi-output Financial Providers / 多输出财务 Provider

When several financial factors share the same expensive setup, use a
multi-output provider instead of duplicating computation.

如果多个财务因子共享同一套高成本前置计算，应使用多输出 provider，而不是重复计算。

Recommended layout:

推荐结构：

```text
factor/common/financial_similarity.rs
factor/chn_stock/daily/f_momentum_80pec.rs
factor/chn_stock/daily/link_new.rs
```

The common provider should:

common provider 负责：

- Define formal factor ids.
- Define `spec(kind)` helpers.
- Build PIT financial metrics once.
- Run cross-sectional transforms once.
- Build similarity/network state once when possible.
- Compute only branches requested by `requested_ids`.

- 定义正式因子 id。
- 定义 `spec(kind)` helper。
- 一次性构造 PIT 财务指标。
- 一次性做截面处理。
- 在可能的情况下共享相似度矩阵或网络状态。
- 根据 `requested_ids` 只计算本次请求的输出分支。

Each thin wrapper should:

每个薄 wrapper 负责：

- Return its own `spec()`.
- Return the same `compute_provider_key()` as sibling factors.
- Forward `compute_many(...)` to the shared provider.
- Keep `compute(...)` as a single-factor compatibility entry point.

- 返回自己的 `spec()`。
- 与 sibling 因子返回相同的 `compute_provider_key()`。
- 将 `compute_many(...)` 转发给共享 provider。
- 保留 `compute(...)` 作为单因子兼容入口。

## Current Financial Similarity Factors / 当前财务相似度因子

The first implemented financial-similarity provider outputs:

第一版财务相似度 provider 输出：

- `f_momentum_80pec`
- `link_new`

Tags:

标签：

```text
XYZQ, financial, fundamental, pit, f_momentum, cs_network,
neutralize, barra, size, sector, daily
```

Both factors:

两个因子均：

- Use PIT financial statements.
- Use cross-sectional percentile-rank standardization for the 10 financial
  metrics.
- Exclude `.BJ` stocks.
- Neutralize final raw signals by Barra `SIZE` and SW level-1 sector.

- 使用 PIT 财报。
- 对 10 个财务指标做截面分位数标准化。
- 剔除 `.BJ` 股票。
- 最终 raw 信号做 Barra `SIZE` 和申万一级行业中性化。

The current `ROE_TTM_YoY` formula is:

当前 `ROE_TTM_YoY` 公式为：

```text
ROE_TTM_YoY = (ROE_TTM_latest - ROE_TTM_yoy) / abs(ROE_TTM_yoy)
ROE_TTM = profit_ttm / latest_equity
ROE_TTM_yoy = profit_ttm_yoy / yoy_equity
```
