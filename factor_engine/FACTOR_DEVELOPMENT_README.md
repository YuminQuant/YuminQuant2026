# Factor Development README / 因子开发教程

本教程面向新增 Rust 因子的研究员和工程实现者。核心原则是：一个正式因子一个 `.rs` 文件，依赖声明精确，公式尽量留在因子文件，公共层只放可复用的数据视图和数学工具。

This guide is for adding Rust factors to YuminQuant. The main rule is: one formal factor per `.rs` file, precise data dependencies, factor formulas in factor files, and reusable data/math helpers in `common` or `operators`.

## 1. 普通日频因子 / Ordinary Daily Factor

新增股票日频因子文件：

Create a stock daily factor file:

```text
factor_engine/src/factor/chn_stock/daily/my_factor.rs
```

最小结构 / Minimal shape:

```rust
use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::pool::DataPool;
use crate::factor::Factor;
use crate::operators::time_series::ts_mean::ts_mean;
use crate::Result;

pub struct MyFactor;

pub fn create() -> Box<dyn Factor> {
    Box::new(MyFactor)
}

impl Factor for MyFactor {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "my_factor".to_string(),
            name: "My Factor".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: 1,
            tags: ["research", "daily"].into_iter().map(str::to_string).collect(),
            description: "20-day mean close demo factor.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            lookback: Lookback { trading_days: 19 },
            aliases: Vec::new(),
            intraday_raw_dependencies: Vec::new(),
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let factor = close.ts(|values| ts_mean(values, 20, 1))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
```

`build.rs` 会自动扫描目录并注册，不需要手写 registry。新增或修改 `spec()` 后运行：

`build.rs` discovers factor files automatically. After adding or editing `spec()`, run:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- plan --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors my_factor
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors my_factor --profile
```

## 2. 数据依赖 / Data Dependencies

所有输入列必须写进 `FactorSpec.dependencies`。引擎会按当前 factor batch 合并依赖，只读取需要的 parquet 和列。

Every input column must be declared in `FactorSpec.dependencies`. The engine merges dependencies per factor batch and reads only needed files and columns.

常用数据集 / Common datasets:

```text
DatasetId::StockDailyPv           open, high, low, close, pre_close, vol, amount
DatasetId::StockDailyBasic        pe, pe_ttm, pb, total_mv, circ_mv, turnover_rate_f
DatasetId::StockAdjFactor         adj_factor
DatasetId::StockDailyLimit        up_limit, down_limit
DatasetId::StockSwClassification  l1_code, l2_code, l3_code
DatasetId::StockCiClassification  l1_code, l2_code, l3_code
DataRequest::index_daily(...)     index close/pre_close etc.
```

### 2.1 ?????? / Sparse Date Loading

??????`DataRequest::new(...)` ??? factor ????? `context.load_dates`???????????????????/??/???????????? `requirements_for_context(context)` ????????????????? `DataRequest::explicit_dates(...)`?
By default, `DataRequest::new(...)` reads the factor's own dense `context.load_dates`. For event-driven factors or factors that need month-end, week-end, or any custom sample dates, compute those dates inside `requirements_for_context(context)` and pass them directly to `DataRequest::explicit_dates(...)`.

???? / Recommended pattern:

```rust
impl Factor for MyFactor {
    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        let dates = my_custom_dates(context);
        vec![DataRequest::explicit_dates(
            DatasetId::StockDailyPv,
            &["close"],
            dates,
        )]
    }
}
```

ROIC-WACC ?????? provider ???? `target_dates + recent week-end dates`?????????? `explicit_dates`??????????????????????? `target_and_month_ends` ????? API?????? `my_custom_dates(context)` ????????
For ROIC-WACC-style factors, the provider computes `target_dates + recent week-end dates` and passes that vector to `explicit_dates`. If a future factor needs month ends, quarter ends, or other event dates, do not add a new constructor; just change the local date generator.

???? / Notes:

- `explicit_dates` ???????????????????? `context.target_dates` ???????
- ?? dataset ??????? `DataPool` ???????? union?
- `FactorSpec.lookback.trading_days` ?????? factor ?????????????????????????????
- ?? `requirements_for_context()` ?????????? factor ??????????? batch ??????? lookback ???????????

- `explicit_dates` sorts and deduplicates dates. If the factor must output daily rows, include `context.target_dates` in the vector.
- Requests for the same dataset are merged by columns and date union in `DataPool`.
- `FactorSpec.lookback.trading_days` still builds the factor-local context window; it no longer means every dependency must be read over the whole window.
- The default `requirements_for_context()` resolves ordinary dependencies into explicit dates for that factor, so another factor's long lookback does not pollute this factor's read window.


When adding a new dataset, update in this order:

1. Add `DatasetId` or a parameterized `DataRequest`.
2. Add path rules in `DataCatalog`.
3. Add loader support in `MarketDataLoader`.
4. Add `DataPool` panel caching if it is a daily fact table.
5. Add path/read tests.

## 3. DailyPanel 表达式 / DailyPanel Expressions

`DailyPanel` 是 `date x instrument` 对齐后的主视图。它支持时序、截面和二元操作。

`DailyPanel` is the aligned `date x instrument` view used by daily factors.

```rust
let panel = data.daily_panel(DatasetId::StockDailyPv)?;
let close = panel.column("close")?;
let open = panel.column("open")?;

let ret_1d = close.zip_binary(&open, |c, o| {
    if o > 0.0 { Some(c / o - 1.0) } else { None }
})?;
let ranked = ret_1d.cs(|values| cs_pctrank(values, true))?;
```

当另一个 daily table 共享 `trade_date + ts_code`，但没有自己的 panel 时：

When another daily table shares `trade_date + ts_code` but has no cached panel:

```rust
let adj = panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
let adj_close = panel.column("close")?.zip_binary(&adj, |close, factor| Some(close * factor))?;
```

## 4. 分钟 raw + 日频后处理 / Minute Raw + Daily Postprocess

分钟派生日频因子建议采用“正式因子薄封装 + common helper”的两层结构：

Minute-derived daily factors should use a two-layer design: a thin formal factor wrapper plus a reusable common helper.

```text
formal factor .rs
  -> common helper
  -> raw_spec()
  -> minute_compute_many()
  -> raw cache
  -> compute() daily postprocess
```

Raw cache path:

```text
data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet
```

### 4.1 真实案例：flash_crash_prob_v / Real Example: flash_crash_prob_v

`flash_crash_prob_v` 的正式因子文件很薄，只声明身份和公式类型：

The formal factor file for `flash_crash_prob_v` is intentionally thin; it only declares identity and formula kind:

```rust
use crate::factor::common::xyzq_vshape_structure::XyzqVshapeFactorKind;

crate::define_xyzq_vshape_structure_factor!(
    StockDailyFlashCrashProbV,
    "flash_crash_prob_v",
    "flashCrashProbV",
    "flashCrashProbV",
    XyzqVshapeFactorKind::FlashCrashProbV
);
```

复杂逻辑放在 `factor/common/xyzq_vshape_structure.rs`。这种写法适合一组因子共享同一套分钟扫描和中间统计。

The heavy logic lives in `factor/common/xyzq_vshape_structure.rs`. This pattern is preferred when a family of factors shares one minute scan and intermediate statistics.

### 4.2 raw id 与 raw spec / Raw IDs And Raw Spec

先在公共 raw id 文件中定义要落盘的日频 raw：

First define the daily raw columns to be materialized:

```rust
pub const MINV_RAW_ID: &str = "daily_minv";
pub const NEGV_MEAN_RAW_ID: &str = "daily_negv_mean";
```

然后声明 raw 需要读取哪些分钟字段，以及是否需要跨日窗口：

Then declare which minute columns the raw provider needs, and whether it needs cross-day input:

```rust
const RAW_WINDOW_DAYS: usize = 1;

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], RAW_WINDOW_DAYS)
}
```

`window_days = 1` 表示 raw 只需要当天分钟数据。只有真正需要跨日拼接的 raw 才应该提高这个窗口或使用 stateful provider。

`window_days = 1` means the raw formula only needs the current day's minute file. Increase it, or use a stateful provider, only when the formula truly needs cross-day continuity.

### 4.3 正式因子声明 raw 依赖 / Formal Factors Declare Raw Dependencies

正式因子在 `FactorSpec.intraday_raw_dependencies` 中声明自己要消费哪些日频 raw。`flash_crash_prob_v` 需要 `daily_minv` 和 `daily_negv_mean`：

Formal factors declare required daily raw columns through `FactorSpec.intraday_raw_dependencies`. `flash_crash_prob_v` consumes `daily_minv` and `daily_negv_mean`:

```rust
let intraday_raw_dependencies = match def.kind {
    XyzqVshapeFactorKind::FlashCrashProbV => vec![
        IntradayDailyRawRequest::new(MINV_RAW_ID, SHARED_RAW_LOOKBACK),
        IntradayDailyRawRequest::new(NEGV_MEAN_RAW_ID, SHARED_RAW_LOOKBACK),
    ],
    _ => vec![],
};
```

这里的 `SHARED_RAW_LOOKBACK` 控制正式 `compute()` 需要读取多少天已物化 raw；它不等于 raw 计算时读取多少天分钟数据。

`SHARED_RAW_LOOKBACK` controls how many days of materialized raw the final `compute()` needs. It is not the same thing as how many minute files the raw formula reads.

### 4.4 同时输出多个 raw / Materializing Multiple Raw Columns

如果多个 raw 来自同一次分钟扫描，优先实现 `minute_compute_many()`，不要让每个 raw 各自重复读分钟 parquet。

If several raw columns share one minute scan, implement `minute_compute_many()` rather than repeating minute IO per raw.

在正式因子的 `impl Factor` 里要显式告诉引擎三件事：

In the formal factor's `impl Factor`, tell the engine three things explicitly:

```rust
impl Factor for StockDailyNegvMean {
    fn spec(&self) -> FactorSpec {
        xyzq_vshape_structure::factor_spec(DEF)
    }

    // 1. 这个 family 可以产出哪些 raw。
    // 1. Which raw columns this family can materialize.
    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_vshape_structure::raw_specs()
    }

    // 2. 同一 provider key 的 raw 会被合并调度。
    // 2. Raw columns with the same provider key are scheduled together.
    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_vshape_structure_provider".to_string()
    }

    // 3. 一次读取分钟数据，同时返回多个 raw series。
    // 3. Read minute data once and return multiple raw series.
    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_vshape_structure::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_vshape_structure::compute_factor(DEF, data)
    }
}
```

这里最关键的是 `intraday_raw_provider_key()`。如果两个 raw 的 provider key 相同，引擎会把它们放到同一个 materialize 批次里；如果 key 不同，即使公式共享逻辑，也可能被分开调度，导致重复扫描分钟数据。

The key method is `intraday_raw_provider_key()`. Raw columns with the same provider key are materialized in one provider batch. If keys differ, the engine may schedule them separately even if the formula could share one minute scan.

`minute_compute()` 适合只有一个 raw 的简单因子；`minute_compute_many()` 适合一组 raw 共享同一组中间结果的因子。比如 V-shape provider 一次计算 `negv_mean`、`negv_max`、`negvwgt_mean`、`negvwgt_max`、`daily_minv`。

Use `minute_compute()` for a single independent raw. Use `minute_compute_many()` when a family of raw columns shares intermediate results. For example, the V-shape provider computes `negv_mean`, `negv_max`, `negvwgt_mean`, `negvwgt_max`, and `daily_minv` in one pass.

common helper 里再实现真正的 multi-raw 输出函数：

The common helper then implements the actual multi-raw materialization:

```rust
pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| all_raw_ids().contains(raw_id))
        .collect::<BTreeSet<_>>();

    let mut values = all_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();

    for trade_date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        // 1. group by ts_code
        // 2. sort by trade_time
        // 3. compute all sibling raw from the same minute points
        // 4. push only requested raw columns
    }

    Ok(output_requested_series(values, requested))
}
```

实际代码里 `push_requested()` 负责“只写本次需要的 raw”，这样同一个 provider 可以服务多因子批次，又不会额外物化没选中的 sibling raw：

In production code, `push_requested()` writes only selected raw columns. One provider can serve a family of factors without materializing unused sibling raw:

```rust
fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}
```

### 4.5 Requested raw and deprecated factors / requested raw 与 deprecated 因子

中文规则：

- 调度层只会把 active 因子的 raw requirements 传给 provider；`deprecated` 因子不会通过 `--all-factors` 或 `--tags` 进入 selected specs。
- provider 收到的 `raw_ids` 就是本次真正需要的 raw 集合。多 raw provider 必须按这个集合计算，而不是把 family 中所有 sibling raw 都顺手算一遍。
- 允许共享必要前置状态，例如同一天分钟收益、5min bar、20 日状态矩阵；但具体 raw 指标分支必须由 `requested.contains(raw_id)` 或 `requested.contains_any([...])` 控制。
- deprecated 因子独有 raw 不应被计算、不应写入 `_cache/intraday_daily`，也不应参与最终 `compute()`；如果 active 因子复用同一个 raw id，则该 raw 仍然正常计算。
- 新的多 raw provider 优先使用 `RequestedRawIds`，避免各 provider 自己重复手写 requested set 过滤逻辑。

English rules:

- The scheduler passes only active factor raw requirements into a provider. Deprecated factors are excluded from broad selections such as `--all-factors` and `--tags`.
- The `raw_ids` argument is the exact raw set requested for this materialization. Multi-raw providers must compute from this set, not from every sibling raw in the family.
- Shared setup is allowed, for example minute returns, 5-minute bars, or a 20-day state matrix. Concrete metric branches must still be guarded by `requested.contains(raw_id)` or `requested.contains_any([...])`.
- Raw columns used only by deprecated factors should not be computed, written to `_cache/intraday_daily`, or consumed by final `compute()`. If an active factor reuses the same raw id, that raw remains active.
- Prefer `RequestedRawIds` for new multi-raw providers so requested filtering has one consistent shape.

Typical pattern:

```rust
let known = all_raw_ids();
let requested = RequestedRawIds::new(raw_ids, &known);
if requested.is_empty() {
    return Ok(Vec::new());
}

let need_tail = requested.contains_any(&[VAR95_RAW_ID, CVAR95_RAW_ID]);
if need_tail {
    // compute only the requested tail metrics
}
```

这类因子运行时可以一次请求整组因子：

You can run the whole sibling family in one batch:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors negv_mean,negv_max,negvwgt_mean,negvwgt_max,flash_crash_prob_v --profile --refresh-minute-cache
```

### 4.6 正式 compute 消费 raw / Daily Compute Consumes Raw

正式因子不再读取分钟文件，而是读取日频 raw panel 做滚动、截面处理和中性化：

The final factor compute should consume daily raw panels, not minute files:

```rust
fn compute_flash_crash_prob_v(data: &DataPool) -> Result<PanelColumn> {
    let panel = data.intraday_daily_raw_panel(MINV_RAW_ID)?;
    let minv = panel.column(MINV_RAW_ID)?;
    let negv_mean = panel.column(NEGV_MEAN_RAW_ID)?;

    let mean_prior_minv = minv.ts(|values| ts_mean(values, 21, 1))?;
    let lambda = mean_prior_minv.map_values(|value| clean(value).and_then(|v| {
        if v > f64::EPSILON { Some(1.0 / v) } else { None }
    }));

    let threshold = negv_mean.cs(|values| {
        let mut valid = values.iter().filter_map(|value| clean(*value)).collect::<Vec<_>>();
        let q75 = quantile_linear(&mut valid, 0.75);
        vec![q75; values.len()]
    })?;

    let flash_raw = lambda.zip_binary(&threshold, |lambda, threshold| {
        Some((-lambda * threshold).exp())
    })?;
    let smoothed = flash_raw.ts(|values| ts_mean(values, 20, 1))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    neutralize_size_sector(&ranked, panel, data)
}
```

实际实现需要对 `None`、非有限值、零分母做更严格的清洗；上面片段只展示数据流。

Production code should handle `None`, non-finite values, and zero denominators carefully; the snippet above focuses on data flow.

### 4.7 开发检查清单 / Minute Factor Checklist

- 只需要当天分钟数据的 raw 使用 `window_days = 1`。
- 同一批 raw 共享分钟扫描时，实现 `minute_compute_many()` 和统一 provider key。
- 跨日公式优先考虑 additive raw；不适合 additive 时再用 stateful provider。
- raw 公式、raw id 或 raw version 变化后，用 `--refresh-minute-cache` 重跑。
- raw id 如果因历史兼容导致名字和新语义不完全一致，必须在测试名或注释里写清楚。

- Use `window_days = 1` for same-day minute raw.
- Use `minute_compute_many()` and a shared provider key when sibling raw columns share one scan.
- Guard each concrete sibling raw metric with `RequestedRawIds`; deprecated-only raw should not be computed.
- Prefer additive daily raw for cross-day formulas; use stateful providers only when additive raw is not suitable.
- Rerun with `--refresh-minute-cache` after raw formula, raw id, or raw version changes.
- If historical raw names no longer match exact semantics, document the semantic change in tests or comments.

## 5. 跨日状态机 raw / Cross-Day Stateful Raw

当公式需要跨日连续序列，但不适合落大量分钟中间列时，可以使用 stateful provider。典型例子：

Use a stateful provider when the formula needs cross-day continuity but should not persist large minute-level intermediate columns. Typical examples:

- 最近 5 日 5min 凸显因子：state 保存最近 5 日合成后的 5min return/salience。
- 5min 流动性 additive raw：state 只保存前一交易日最后一根 5min Amihud，用来计算下一日第一根 `ΔAmihud`。

状态机原则：

Stateful provider rules:

- state 只保存下一天真正需要的最小信息。
- 每天读当天分钟文件，算完后释放当天原始分钟数据。
- 首个目标日前的 warmup 只用于初始化 state。
- raw version 或 raw id 改变时，避免和旧缓存混用。

## 6. 后处理与中性化 / Postprocess And Neutralization

因子后处理必须在公式里显式写出。常见选择：

Postprocess should be explicit in the factor formula. Common choices:

```text
ts_mean(raw, window, min_periods)
cs_zscore
cs_pctrank
SIZE + SW level-1 sector neutralization
20d return + SIZE + SW level-1 sector neutralization
```

注意：回测 CLI 的 `sector` 代表申万一级行业；开发中如需中性化，建议清楚写明用 `StockSwClassification.l1_code` 还是 `StockCiClassification.l1_code`。

Note: backtest CLI `sector` means Shenwan level-1 sector. In factor code, state clearly whether Shenwan or CITIC classification is used.

## 7. Deprecated 与删除列 / Deprecated Tags And Column Removal

不再推荐使用的因子不要删除 `.rs` 文件，给 metadata tags 增加 `deprecated`，这样 `--all-factors` 和 `--tags` 默认跳过。

Do not delete old factor source files just to retire them. Add the `deprecated` tag so broad selections skip them.

从正式因子库或外部 alpha root 删除历史列：

Remove historical columns from factor parquet or external alpha roots:

```powershell
python scripts\remove_factor_columns.py --start-date 20110101 --end-date 20260424 --columns WQAlpha007,WQAlpha021 --dry-run
python scripts\remove_factor_columns.py --start-date 20110101 --end-date 20260424 --columns WQAlpha007,WQAlpha021

python scripts\remove_factor_columns.py --factor-root data\models --start-date 20110101 --end-date 20260424 --columns ml_alpha_lstm --dry-run
python scripts\remove_factor_columns.py --factor-root data\models --start-date 20110101 --end-date 20260424 --columns ml_alpha_lstm
```

## 8. 常用验证 / Validation Checklist

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo check --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor --profile
```

常见问题 / Common issues:

- `missing required column ts_code`: 输入 parquet 结构不对或读取路径不对。
- stale metadata: 新增/改名因子后忘记跑 `metadata`。
- all-null output: 检查 lookback、输入日期、行业分类、PIT 数据是否可用。
- 旧 raw cache: raw 公式变更后需要 `--refresh-minute-cache`。
- Label 缺未来数据时可能跳过目标日，因子通常仍写出 null。
