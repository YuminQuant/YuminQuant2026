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

新增数据集时，按这个顺序改：

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

分钟派生日频因子推荐两层：

Minute-derived daily factors should use two layers:

1. `minute_compute()` 读取单日分钟数据，计算日频 raw。
2. `compute()` 读取 raw daily panel，做 rolling、rank、中性化等后处理。

Raw cache path:

```text
data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet
```

正式因子在 `FactorSpec.intraday_raw_dependencies` 中声明 raw 依赖。raw 公式变化后必须重跑：

Formal factors declare raw dependencies in `FactorSpec.intraday_raw_dependencies`. If raw formulas change, rerun with:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor --refresh-minute-cache --profile
```

分钟因子设计建议：

Minute factor guidelines:

- 只需要当天分钟数据的 raw 使用 `window_days = 1`。
- 跨日拼接需要状态机时，只在 state 中保存必要统计量或最近合成 bar，不保存全量分钟原始数据。
- 能落成可加 raw 的跨日公式，优先落日频 additive raw，再用 `ts_sum` 恢复窗口公式。
- provider 要支持 sibling raw 去重，同批多个因子共享一次分钟扫描。

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
