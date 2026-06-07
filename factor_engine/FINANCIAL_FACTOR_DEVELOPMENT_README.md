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

## Stock-Level Event Cache / 股票级事件驱动缓存

Financial statement values are sparse. New financial factors should split the
pipeline into slow stock-level statement snapshots and daily fast
cross-sectional work.

财报数据是低频稀疏数据。新的财务因子应把流程拆成“股票级财报慢指标 snapshot”和“每日快变量/截面处理”两层。

New financial factors should prefer provider-state caching:
`InstrumentAlignedSnapshotCache<T>` lives in the factor/provider state, and each
trading date calls `cached_financial_stock_snapshots_for_date(...)` to update
only stocks whose marker changed. This keeps exactly one current stock-level
snapshot cache per provider and avoids holding multiple event cross-sections in
memory.

新的财务因子应优先使用 provider-state 缓存：把
`InstrumentAlignedSnapshotCache<T>` 放在因子或 provider state 中，每个交易日调用
`cached_financial_stock_snapshots_for_date(...)`，只更新 marker 变化的股票。这样每个
provider 只维护一个当前股票级 snapshot cache，不会在内存中堆积多个事件截面。

`cached_financial_stock_snapshots(...)` is still available as a small batch
utility, but it should not be the default path for long-lived financial factors
or Barra financial exposures.

`cached_financial_stock_snapshots(...)` 仍可作为小范围批量工具使用，但不再是长期运行的
财务因子或 Barra 财务暴露的默认开发方式。

Use it when all of the following are true:

满足以下条件时使用该工具：

- The value is a stock-level financial statement formula. / 该值是股票级财报公式。
- The formula only changes when visible `ann_date/f_ann_date` records or declared synthetic events change. / 公式只会在 PIT 可见财报记录或声明的 synthetic event 变化时变化。
- The result can be reused across target dates until the marker changes. / marker 不变时可跨目标日复用结果。

Do **not** cache daily fast variables:

不要缓存每日快变量：

- market value, price, return, turnover, present universe / 市值、价格、收益、换手率、在市状态；
- percentile rank, zscore, regression, neutralization / 分位数、标准化、回归、中性化；
- peer/network reductions such as F-Link weighted returns / F-Link、peer 加权收益等截面网络降维；
- rolling calendar values unless the calendar boundary is added to the marker. / 如果 rolling calendar 边界会改变结果，必须把边界加入 marker，否则不要缓存。

Recommended provider-state pattern / 推荐 provider-state 写法：

```rust
#[derive(Clone, Copy, Debug)]
struct SlowSnapshot {
    revenue_ttm: Option<f64>,
    equity_latest: Option<f64>,
}

#[derive(Default)]
struct MyProviderState {
    slow_cache: InstrumentAlignedSnapshotCache<SlowSnapshot>,
}

fn slow_marker(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_latest_ttm(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
    );
    builder.include_latest_annual(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
    );
    builder.build()
}

fn slow_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<SlowSnapshot> {
    Some(SlowSnapshot {
        revenue_ttm: income.ttm_sum(ts_code, trade_date, "revenue"),
        equity_latest: balance.latest_annual_value(
            ts_code,
            trade_date,
            "total_hldr_eqy_exc_min_int",
        ),
    })
}

let mut values = vec![None; panel.shape_len()];
let instrument_count = panel.instruments().len();
for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
    if !panel.is_target_date(trade_date) {
        continue;
    }
    let snapshots = cached_financial_stock_snapshots_for_date(
        panel,
        trade_date,
        &mut state.slow_cache,
        |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
        |trade_date, ts_code, _| slow_marker(ts_code, trade_date, &income, &balance),
        |trade_date, ts_code, _| slow_snapshot(ts_code, trade_date, &income, &balance),
    );
    for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
        let offset = date_idx * instrument_count + instrument_idx;
        let Some(snapshot) = snapshot else {
            continue;
        };
        let Some(mv) = total_mv.values()[offset] else {
            continue;
        };
        values[offset] = snapshot.revenue_ttm.and_then(|revenue| safe_div(revenue, mv));
    }
}
let raw = panel.column_from_values(values)?;
```

Callback roles / 回调职责：

- `skip_fn(trade_date, ts_code, offset)`: return `true` for `.BJ`, non-present rows, or excluded universe members; this clears the stock cache. / 对 `.BJ`、不在市行或被股票池剔除的股票返回 `true`，同时清空该股票缓存。
- `marker_fn(...)`: declare all PIT records and synthetic events that can change the snapshot. / 声明所有会影响 snapshot 的 PIT 记录链和 synthetic event。
- `compute_fn(...)`: calculate only the stock-level slow snapshot; it runs only when marker changes. / 只计算股票级慢指标；只有 marker 变化时才执行。

Marker rules / marker 规则：

- Use record fingerprints, not values: `dataset + end_date + disclosure_date + report_type + update_flag`. / marker 使用记录指纹，而不是财务数值本身。
- Include every statement chain that can affect the snapshot: latest quarter, YoY quarter, previous quarter, TTM chain, annual chain, etc. / 任何会影响 snapshot 的记录链都必须加入 marker。
- If a formula depends on non-statement events, add a synthetic marker. `DP_LTM` is the main example because the 12-month implemented dividend window can change independently of statements. / 如果公式依赖非财报事件，需要加入 synthetic marker；典型例子是 `DP_LTM` 的 12 个月已实施分红窗口。
- When a stock is not present in `DailyPanel.present` or is excluded by the factor universe, clear or skip the cache entry rather than carrying the previous snapshot forward. / 股票不在市或被因子股票池剔除时，应清空或跳过缓存，不能沿用旧 snapshot。
- If marker is missing, do not reuse an old snapshot. / marker 缺失时不能复用旧 snapshot。

Multi-output providers should store shared slow snapshot caches in provider
state, then write all requested outputs from the current snapshots. For example,
`f_momentum_80pec` and `link_new` share the 10-dimensional financial vector;
Barra `growth`, `quality`, `value`, and `dividend_yield` cache their
statement/dividend-driven subcomponents but still run Barra standardization and
neutralization on the normal daily panel.

多输出 provider 应把共享慢指标 snapshot cache 放在 provider state 中，再基于当前 snapshot
输出请求的因子。例如 `f_momentum_80pec` 和 `link_new` 共用 10 维财务向量；Barra
`growth`、`quality`、`value`、`dividend_yield` 会缓存财报或分红驱动的子指标，但 Barra
标准化和中性化仍按正常日频 panel 执行。

## Factor Update Policy / 因子截面更新策略

Stock-level statement cache only controls slow financial lookups. The final
factor cross-section has its own explicit update policy. Do not infer this from
the `fundamental` tag.

股票级财报缓存只控制慢财务查表。正式因子截面还有独立的显式更新策略，不能仅凭 `fundamental` tag 自动推断。

Use `FactorUpdatePolicy` in the wrapper/provider:

在 wrapper 或 provider 中声明 `FactorUpdatePolicy`：

- `Daily`: default. Use for valuation factors, price/return factors, and any
  factor whose effective signal changes every trading day.
- `FinancialEventSnapshot`: pure statement factors. On financial event dates,
  recompute the whole cross-section, including rank/zscore, OLS, peer/network
  reductions, and neutralization. On non-event dates, write daily rows by
  replaying the latest final factor cross-section.
- `FinancialEventStateDailyFast`: mixed slow/fast factors. Recompute slow
  statement-driven state only on events, but recompute fast branches every day.
  `f_momentum_80pec` is the template: financial vectors and top peers are
  event-driven, while peer Ret20, Ret20 residualization, and SIZE+sector
  neutralization are daily.

- `Daily`：默认策略。用于估值因子、价格收益因子，以及任何有效信号每天都会变化的因子。
- `FinancialEventSnapshot`：纯财报慢因子。财务事件日重算完整截面，包括 rank/zscore、OLS、peer/network 降维和中性化；非事件日逐日写出，但回放最近一次最终因子截面。
- `FinancialEventStateDailyFast`：快慢混合因子。只在事件日更新财报慢状态，快变量分支每日计算。`f_momentum_80pec` 是模板：财务向量和 top peers 事件驱动，peer Ret20、Ret20 残差化、SIZE+sector 中性化每日更新。

Financial event schedules are built from the provider-declared event sources.
Statement events use `f_ann_date.or(ann_date)`. Dividend LTM factors should also
include dividend announcement dates, `ex_date`, and the 12-month window expiry
date. `trade_date` remains the trading-calendar date; a weekend disclosure is
picked up on the next target trading day because the engine checks
`(last_processed_trade_date, current_trade_date]`.

财务事件 schedule 由 provider 声明的事件源构建。财报事件使用 `f_ann_date.or(ann_date)`；分红 LTM 因子还应纳入分红公告日、`ex_date` 以及 12 个月窗口滚出日期。`trade_date` 仍然由交易日历控制；周末公告会在下一个目标交易日生效，因为引擎检查 `(last_processed_trade_date, current_trade_date]` 区间。

Implementation checklist / 实现清单：

1. Pure slow factor: override `update_policy()`, `initial_compute_state()`, and
   `compute_many_stateful()`, then call `compute_financial_event_snapshot_many`.
2. Mixed factor: keep a provider-specific state struct; update slow state only
   when `FinancialEventSchedule::has_event_after_until(...)` is true; compute
   fast daily branches for every target date.
3. Multi-output provider: every wrapper sharing `compute_provider_key()` must
   use the same state type, and `factor-batch-size` will not split that provider.
4. Non-event replay stores final factor values, not raw financial metrics. This
   means pure slow factors also reuse the event-date neutralized cross-section.

1. 纯慢因子：覆盖 `update_policy()`、`initial_compute_state()` 和 `compute_many_stateful()`，再调用 `compute_financial_event_snapshot_many`。
2. 快慢混合因子：维护 provider 自己的 state；只有 `FinancialEventSchedule::has_event_after_until(...)` 为真时更新慢状态；每个目标交易日都计算快变量分支。
3. 多输出 provider：共享 `compute_provider_key()` 的 wrapper 必须使用同一种 state 类型，`factor-batch-size` 不会把同一个 provider 拆开。
4. 非事件日回放的是最终因子值，而不是原始财务指标。因此纯慢因子也会复用事件日已经中性化后的截面。

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
