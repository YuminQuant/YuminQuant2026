# Financial Factor Development / 财务因子开发说明

This document defines the current point-in-time financial, main-business,
dividend, and analyst event workflow used by `factor_engine`.

本文档定义 `factor_engine` 当前的 PIT 财报、主营业务、分红和分析师事件型因子开发规则。

## Core Rule / 核心规则

Financial/event-style stock factors must not use `StockDailyPv` as an output
panel anchor. Each data family owns its own grid:

- PV factors use the PV panel.
- Minute and intraday factors use their own raw/minute panels.
- Financial, main-business, dividend, and analyst event factors use the stock
  universe panel built from `data/stock_data/info/stock_basic.parquet`.

财务/事件型股票因子不得用 `StockDailyPv` 作为输出网格锚点。各类数据自己建立网格：

- PV 因子使用 PV panel。
- 分钟和日内因子使用自己的 raw/minute panel。
- 财报、主营业务、分红、分析师事件型因子使用 `stock_basic.parquet` 构建的 stock universe panel。

Never add `DataRequest::new(DatasetId::StockDailyPv, &["close"])` only to get a
daily panel. If a formula genuinely needs prices, returns, turnover, market
value, or other fast data, request the exact dataset and columns and align them
onto the stock universe panel by `(trade_date, ts_code)`.

不要为了拿日频 panel 而声明 `StockDailyPv.close`。如果公式确实需要价格、收益、换手、市值等快变量，只请求真实需要的字段，再按 `(trade_date, ts_code)` 映射到 stock universe panel。

## Output Panel / 输出网格

Use:

```rust
let panel = data.stock_universe_panel()?;
```

The panel is built from `stock_basic.parquet` with at least these fields:
`ts_code`, `list_status`, `list_date`, `delist_date`, `exchange`, and `market`.
The current implementation keeps A-share style codes ending in `.SH`, `.SZ`, or
`.BJ`, sorts them by `ts_code`, and marks a row present when:

```text
list_date <= trade_date && (delist_date is null || trade_date <= delist_date)
```

该 panel 从 `stock_basic.parquet` 构建，至少读取 `ts_code`、`list_status`、`list_date`、`delist_date`、`exchange`、`market`。当前实现保留 `.SH`、`.SZ`、`.BJ` A 股代码，并按 `ts_code` 排序；当满足以下条件时该股票在该交易日 present：

```text
list_date <= trade_date && (delist_date 为空 || trade_date <= delist_date)
```

Records whose `ts_code` is not in `stock_basic` do not create output rows.
Legal stocks with missing financial statements, missing SIZE, missing industry,
or missing PV data remain in the output grid and produce `None` unless the
factor explicitly fills missing values.

不在 `stock_basic` 中的财报、主营业务、分红或分析师记录不会产生输出行。合法股票即使缺财报、缺 SIZE、缺行业、缺行情，也保留在输出网格中，因子值为 `None`，除非因子自身有明确填缺失逻辑。

## PIT Readers / PIT 读取

Financial statement consumers should use `DataPool::financial_reader(...)`.
Do not build factor-specific statement maps.

财报消费者应使用 `DataPool::financial_reader(...)`，不要新增因子私有的财报 map。

Example:

```rust
let income = data.financial_reader(
    DatasetId::StockIncome,
    ReportTypePreference::income_single_quarter(),
)?;
let balance = data.financial_reader(
    DatasetId::StockBalanceSheet,
    ReportTypePreference::balance_sheet_consolidated(),
)?;
let cashflow = data.financial_reader(
    DatasetId::StockCashFlow,
    ReportTypePreference::income_single_quarter(),
)?;
```

Useful reader helpers:

- `record_for_end_date(ts_code, trade_date, end_date)`: PIT-safe row for a report period.
- `latest_quarter_end_date(ts_code, trade_date)`: latest visible quarter end.
- `ttm_sum(ts_code, trade_date, column)`: latest available four-quarter sum.
- `ttm_sum_for_end_date(ts_code, trade_date, end_date, column)`: TTM sum anchored at a report period.
- `latest_annual_value(...)`, `latest_annual_end_date(...)`, `annual_value_for_end_date(...)`, `annual_values(...)`: annual helpers.

常用接口：

- `record_for_end_date(ts_code, trade_date, end_date)`：指定报告期的 PIT 安全记录。
- `latest_quarter_end_date(ts_code, trade_date)`：当前交易日可见的最新季度。
- `ttm_sum(ts_code, trade_date, column)`：最新可得四季度 TTM 汇总。
- `ttm_sum_for_end_date(ts_code, trade_date, end_date, column)`：以指定报告期为锚点的 TTM 汇总。
- `latest_annual_value(...)`、`latest_annual_end_date(...)`、`annual_value_for_end_date(...)`、`annual_values(...)`：年度口径 helper。

## Report Type / 报表类型

Financial statement rows contain `report_type`.

| Code | Type | Usage |
| --- | --- | --- |
| 1 | Consolidated | Listed-company consolidated statement, default full-period statement. |
| 2 | Single-quarter consolidated | Single-quarter consolidated statement. |
| 3 | Adjusted single-quarter consolidated | Preferred single-quarter income row when available. |
| 4 | Adjusted consolidated | Current-year disclosure of prior-year comparable data. |
| 5 | Pre-adjustment consolidated | Original consolidated row retained after revision. |
| 6-12 | Parent/pre-adjustment variants | Use only when a factor explicitly needs parent-company data. |

Current defaults:

- Income single quarter: `ReportTypePreference::income_single_quarter() = [3, 2]`
- Consolidated balance sheet: `ReportTypePreference::balance_sheet_consolidated() = [1, 4]`
- Generic consolidated: `ReportTypePreference::consolidated() = [1, 4]`

Within the same `ts_code + end_date + report_type`, only rows with
`f_ann_date.or(ann_date) <= trade_date` are visible. Versions are sorted by
`disclosure_date` descending, then `update_flag` descending.

同一 `ts_code + end_date + report_type` 下，只能使用 `f_ann_date.or(ann_date) <= trade_date` 的记录。版本按 `disclosure_date` 倒序、`update_flag` 倒序选择。

## Dependencies / 依赖声明

Factor specs should request only needed value columns. The loader automatically
adds PIT key/version columns such as `ts_code`, `ann_date`, `f_ann_date`,
`end_date`, `report_type`, and `update_flag`.

因子 spec 只应请求实际需要的数值列。loader 会自动补齐 PIT 键列和版本列，例如 `ts_code`、`ann_date`、`f_ann_date`、`end_date`、`report_type`、`update_flag`。

Example:

```rust
DataRequest::financial_quarters(
    DatasetId::StockIncome,
    &["revenue", "n_income_attr_p"],
    8,
)
```

Any request for financial, dividend, main-business, or analyst datasets causes
`DataPool` to build the stock universe panel from `stock_basic`. You do not need
to request `StockBasic` manually unless the formula itself uses stock-basic
fields such as `list_date`.

只要 request 中包含财报、分红、主营业务或分析师数据，`DataPool` 就会自动从 `stock_basic` 构建 stock universe panel。除非公式本身需要 `list_date` 等 stock-basic 字段，否则不需要手动请求 `StockBasic`。

## Event Replay / 事件回放

Pure slow financial factors should use `FactorUpdatePolicy::FinancialEventSnapshot`
and `compute_financial_event_snapshot_streaming_on_panel(...)`.

纯财报慢因子应使用 `FactorUpdatePolicy::FinancialEventSnapshot` 和 `compute_financial_event_snapshot_streaming_on_panel(...)`。

There is no PV-anchor compatibility wrapper. The old API that internally read
`data.daily_panel(DatasetId::StockDailyPv)` has been removed. Callers must pass
the panel explicitly.

目前没有 PV-anchor 兼容入口。旧的内部读取 `data.daily_panel(DatasetId::StockDailyPv)` 的 API 已删除。调用方必须显式传入 panel。

Template:

```rust
let panel = data.stock_universe_panel()?;
let schedule = FinancialEventSchedule::from_pit_readers(&[
    income.clone(),
    balance.clone(),
]);

let raw_series = compute_financial_event_snapshot_streaming_on_panel(
    requested_ids,
    context,
    data,
    panel,
    &mut state.final_cache,
    &schedule,
    &requested_specs,
    |ids, event_context, event_data| {
        compute_raw_on_event(ids, event_context, event_data)
    },
)?;
```

The replay cache stores final factor values, not raw financial metrics. On
event dates, compute the full event-date cross-section, including zscore,
regression, peer/network reductions, and neutralization. On non-event dates,
replay the latest final factor cross-section onto the same explicit panel.

回放缓存存的是最终因子值，不是原始财务指标。事件日应计算完整截面，包括 zscore、回归、peer/network 降维和中性化；非事件日把最近一次最终因子截面回放到同一个显式 panel 上。

## Slow Snapshot Cache / 慢指标缓存

For stock-level slow formulas, prefer provider-state caching:
`InstrumentAlignedSnapshotCache<T>` plus `cached_financial_stock_snapshots_for_date(...)`.
This cache is aligned by `ts_code`, not by raw vector position, so changing
batch order or instrument order does not shift financial values across stocks.

股票级慢指标公式优先使用 provider-state 缓存：`InstrumentAlignedSnapshotCache<T>` + `cached_financial_stock_snapshots_for_date(...)`。该缓存按 `ts_code` 对齐，而不是按原始数组位置对齐，因此 batch 或股票顺序变化不会导致财务指标错位。

Cache only stock-level slow snapshots. Do not cache daily fast variables:

- price, return, turnover, market value, present universe;
- percentile rank, zscore, regression, neutralization;
- peer/network reductions such as F-Link weighted returns;
- rolling calendar values unless the calendar boundary is part of the marker.

只缓存股票级慢指标。不要缓存每日快变量：

- 价格、收益、换手、市值、在市状态；
- 分位数、zscore、回归、中性化；
- F-Link 加权收益等 peer/network 降维结果；
- 未写入 marker 的 rolling calendar 结果。

Template:

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
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
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
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
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

let snapshots = cached_financial_stock_snapshots_for_date(
    panel,
    trade_date,
    &mut state.slow_cache,
    |_, _ts_code, offset| !panel.is_present_offset(offset),
    |trade_date, ts_code, _| slow_marker(ts_code, trade_date, &income, &balance),
    |trade_date, ts_code, _| slow_snapshot(ts_code, trade_date, &income, &balance),
);
```

Callback roles:

- `skip_fn(trade_date, ts_code, offset)`: return `true` for non-present rows or factor-excluded stocks; this clears that stock cache entry.
- `marker_fn(...)`: declare every PIT record and synthetic event that can change the snapshot.
- `compute_fn(...)`: calculate only the stock-level slow snapshot; it runs only when the marker changes.

回调职责：

- `skip_fn(trade_date, ts_code, offset)`：非 present 行或因子排除股票返回 `true`，同时清空该股票缓存。
- `marker_fn(...)`：声明所有会影响 snapshot 的 PIT 记录链和 synthetic event。
- `compute_fn(...)`：只计算股票级慢指标；只有 marker 变化时才执行。

Marker rules:

- Use record fingerprints, not financial values: `dataset + end_date + disclosure_date + report_type + update_flag`.
- Include every statement chain that can affect the snapshot: latest quarter, YoY quarter, previous quarter, TTM chain, annual chain, etc.
- If the formula depends on non-statement events, add a synthetic marker. Dividend LTM windows are the main example.
- If a marker is missing, do not reuse an old snapshot.

marker 规则：

- marker 使用记录指纹，而不是财务数值本身：`dataset + end_date + disclosure_date + report_type + update_flag`。
- 所有影响 snapshot 的记录链都必须纳入 marker：最新季度、同比季度、上一季度、TTM 链、年度链等。
- 如果公式依赖非财报事件，加入 synthetic marker。分红 LTM 窗口是典型例子。
- marker 缺失时不能复用旧 snapshot。

## Fast Data Alignment / 快变量对齐

When a financial factor genuinely needs a fast dataset, keep the stock universe
panel as the main grid and map the fast column onto it:

```rust
let panel = data.stock_universe_panel()?;
let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
let adj_factor =
    panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
```

Missing fast data for legal stocks remains `None`. Cross-sectional regressions
and neutralization should use only rows where raw factor, SIZE, and industry are
all valid; rows outside that intersection stay in the output grid with `None`.

合法股票缺快变量时保留为 `None`。截面回归和中性化只使用 raw 因子、SIZE、行业都有效的交集样本；交集外合法股票仍保留输出行，值为 `None`。

## Update Policies / 更新策略

Declare `FactorUpdatePolicy` explicitly in every factor wrapper/provider. Never
infer update behavior from tags.

每个因子 wrapper/provider 都必须显式声明 `FactorUpdatePolicy`，不要从 tag 自动推断。

- `Daily`: valuation, price, return, turnover, market-cap, and any factor whose signal changes every trading day.
- `FinancialEventSnapshot`: pure slow statement/event factors. Recompute the final cross-section only when financial events occur; replay final values otherwise.
- `FinancialEventStateDailyFast`: mixed slow/fast factors. Update slow state only on events, then recompute fast branches every target date.

- `Daily`：估值、价格、收益、换手、市值，以及任何有效信号每日变化的因子。
- `FinancialEventSnapshot`：纯慢财报/事件因子。财务事件日重算最终截面，非事件日回放最终值。
- `FinancialEventStateDailyFast`：快慢混合因子。事件日更新慢状态，每个目标交易日重新计算快变量分支。

Financial event schedules are built from provider-declared event sources.
Statement events use `f_ann_date.or(ann_date)`. Non-trading-day disclosures are
picked up on the next target trading day because the engine checks
`(last_processed_trade_date, current_trade_date]`.

财务事件 schedule 由 provider 声明的事件源构建。财报事件使用 `f_ann_date.or(ann_date)`。非交易日公告会在下一个目标交易日生效，因为引擎检查 `(last_processed_trade_date, current_trade_date]`。

## Missing Values / 缺失值

Default behavior is conservative:

- Missing core financial fields or invalid denominators produce `None`.
- Missing optional add-back fields may be filled only when the factor spec says so.
- Missing SIZE or industry means the stock cannot participate in neutralization and should remain `None` after that step.
- Factor-specific fills must be local and documented in the factor implementation.

默认处理保持保守：

- 核心财务字段缺失或分母非法输出 `None`。
- 可选加回项只有在因子口径明确说明时才能填 0 或其他值。
- 缺 SIZE 或行业时，该股票不参与中性化，中性化后保留 `None`。
- 因子级填缺失逻辑必须局部实现并在因子代码中说明。

Financial similarity factors are a known special case: for present non-BJ
stocks, missing per-metric percentile ranks may be filled with `0` after
ranking, while all-zero vectors are excluded from cosine similarity.

财务相似度因子是已知特例：对 present 且非 BJ 股票，单指标分位数 rank 缺失可在 rank 后填 `0`；全零向量不参与余弦相似度。

## BJ Stock Rule / 北交所股票规则

The stock universe panel includes `.BJ`; each factor decides whether `.BJ`
participates.

stock universe panel 本身包含 `.BJ`；是否参与由具体因子决定。

Unless explicitly stated otherwise, financial peer/network factors exclude `.BJ`
stocks from cross-sectional transforms, similarity matrices, regressions, and
final valid outputs.

除非因子另有说明，财务 peer/network 因子会将 `.BJ` 从截面处理、相似度矩阵、回归和最终有效输出中剔除。

## Multi-output Providers / 多输出 Provider

When several financial factors share expensive setup, use a multi-output
provider. The engine groups selected factors by `compute_provider_key()` and
calls `compute_many(requested_ids, ...)` once per provider. Wrappers sharing a
provider key must use the same state type and remain requested-aware.

多个财务因子共享高成本前置计算时，应使用多输出 provider。engine 会按 `compute_provider_key()` 分组，并对每个 provider 调用一次 `compute_many(requested_ids, ...)`。共享 provider key 的 wrapper 必须使用同一种 state 类型，并且保持 requested-aware。

Example layout:

```text
factor/common/financial_similarity.rs
factor/chn_stock/daily/f_momentum_80pec.rs
factor/chn_stock/daily/link_new.rs
```

## Implementation Checklist / 实现清单

1. Declare exact `DataRequest`s. Do not add `StockDailyPv.close` as an anchor.
2. Use `let panel = data.stock_universe_panel()?;` for financial/event stock factors.
3. Use `data.financial_reader(...)`, `data.main_business_reader()`, `data.dividend_reader()`, or `data.daily(DatasetId::StockAnalystReport)` as appropriate.
4. Put stock-level slow snapshots in provider state with `InstrumentAlignedSnapshotCache<T>`.
5. Include every PIT record chain and synthetic event in the marker.
6. Use `compute_financial_event_snapshot_streaming_on_panel(...)` for pure event replay.
7. Align fast datasets onto the stock universe panel with `panel.column_from_table(...)`.
8. Neutralize/regress on valid intersections only; preserve legal out-of-intersection stocks as `None`.
9. Add tests for formulas, marker invalidation, batch/order alignment, missing values, and no PV-anchor dependency.

1. 精确声明 `DataRequest`，不得把 `StockDailyPv.close` 当锚点。
2. 财务/事件型股票因子使用 `let panel = data.stock_universe_panel()?;`。
3. 按需使用 `data.financial_reader(...)`、`data.main_business_reader()`、`data.dividend_reader()` 或 `data.daily(DatasetId::StockAnalystReport)`。
4. 股票级慢 snapshot 放进 provider state，使用 `InstrumentAlignedSnapshotCache<T>`。
5. marker 覆盖所有 PIT 记录链和 synthetic event。
6. 纯事件回放使用 `compute_financial_event_snapshot_streaming_on_panel(...)`。
7. 快变量用 `panel.column_from_table(...)` 映射到 stock universe panel。
8. 回归/中性化只用有效交集，交集外合法股票保留 `None`。
9. 测试覆盖公式、marker 失效、batch/股票顺序对齐、缺失值，以及无 PV anchor 依赖。

Recommended validation commands:

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml financial
cargo test --manifest-path factor_engine\Cargo.toml comprehensive_profitability special_roa sfli2 abcfo
```
