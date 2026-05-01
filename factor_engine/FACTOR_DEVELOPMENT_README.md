# Factor Development README

This document is for researchers adding new Rust factors to the YuminQuant
factor engine.

## Add A Stock Daily Factor

Create one file under:

```text
factor_engine/src/factor/chn_stock/daily/{factor_id}.rs
```

Use the file stem as the short factor id and output column. Each factor exposes
`create()`, implements `Factor`, declares its metadata in `spec()`, and writes
the expression in `compute()`.

Minimal shape:

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
            tags: ["price_volume", "daily"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            description: "Example factor".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            lookback: Lookback { trading_days: 19 },
            aliases: Vec::new(),
            intraday_raw_dependencies: Vec::new(),
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let factor = close.ts(|values| ts_mean(values, 20, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}
```

After adding the file, run:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- plan --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors my_factor
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors my_factor --profile
```

`build.rs` scans the factor directory and generates the registry; there is no
handwritten factor list to update.

## Data Dependencies

Every input must be declared in `FactorSpec.dependencies`. The engine uses these
requests to load only the necessary datasets and columns for each factor batch.

Common stock daily datasets:

- `DatasetId::StockDailyPv`: `open`, `high`, `low`, `close`, `pre_close`, `vol`,
  `amount`
- `DatasetId::StockDailyBasic`: `pe`, `pe_ttm`, `pb`, `total_mv`, `circ_mv`,
  `turnover_rate_f`
- `DatasetId::StockAdjFactor`: `adj_factor`
- `DatasetId::StockSwClassification`: `l1_code`, `l2_code`, `l3_code`
- parameterized index data via `DataRequest::index_daily("000985.CSI", &["close", "pre_close"])`

Daily fact tables are read from daily parquet files, for example:

```text
data/stock_data/daily/pv/2026/20260424.parquet
data/stock_data/daily/basic/2026/20260424.parquet
data/stock_data/daily/adj_factor/2026/20260424.parquet
```

If a new dataset is needed, add it in this order:

1. Add a `DatasetId` or a parameterized `DataRequest` variant.
2. Add its path rules in `DataCatalog`.
3. Add loader support in `MarketDataLoader`.
4. Add panel caching in `DataPool` only if the data is a daily fact table.
5. Add tests for path resolution and a small read.

Do not hide factor formulas in `common`. Put formulas in the factor file; put
only reusable data views, panel operations, and generic utilities in `common`.

## DailyPanel Expression Style

`DailyPanel` is the main daily factor view. It aligns data to a shared
`date x instrument` index and stores each column as a flat vector. `DataPool`
builds each panel once per batch, so all factors in the same batch reuse it.

Useful patterns:

```rust
let panel = data.daily_panel(DatasetId::StockDailyPv)?;
let close = panel.column("close")?;
let open = panel.column("open")?;

let return_20d = close.ts(|values| ts_pctchg(values, 20))?;
let corr = close.ts_binary(&open, |c, o| ts_corr(c, o, 20, 20))?;
let ranked = return_20d.cs(|values| cs_rank(values, true))?;
```

Use `column_from_table()` when another table is aligned by the same
`trade_date + ts_code` keys but does not have its own cached panel:

```rust
let adj = panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
let adj_close = panel.column("close")?.zip_binary(&adj, |close, factor| Some(close * factor))?;
```

For industry or Barra neutralization, keep it explicit in `compute()`:

```rust
let sector_map = data.stock_sw_classification_map()?;
let neutral = raw.cs_by_group(
    |date, codes| sector_map.groups_for(date, codes, ClassificationLevel::Sector),
    cs_neutralize,
)?;
```

## Minute-To-Daily Factors

Minute-derived daily factors should use two layers:

1. `minute_compute()` calculates one trading day's minute-to-daily raw statistic.
2. `compute()` reads the raw daily panel and applies final daily post-processing.

Raw cache path:

```text
data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet
```

A minute factor that needs no additional post-processing should still use this
same flow: `compute()` can simply return the raw column as the final factor. The
benefit is consistent caching, profiling, and writer behavior.

Use `--refresh-minute-cache` when the raw formula changes and old cache files
should be rebuilt.

## Execution Flow And Batching

For `run --asset stock --frequency daily --start-date 20260101 --end-date 20260130`:

1. The engine reads `factor_metadata.parquet` and selects factors by asset,
   frequency, `--factors`, or `--tags`.
2. It aligns user dates to the trading calendar.
3. It splits target dates into date batches. The default is one trading day per
   batch.
4. It splits selected factors into factor batches. The default is `64`.
5. For each date batch and factor batch, it computes `load_dates` from lookback.
   A 20-day factor for one target day loads roughly 21 trading-day files.
6. It reads only the required dataset columns for that factor batch.
7. It builds cached panels once per dataset in `DataPool`.
8. It computes factors in parallel inside the factor batch using rayon.
9. It writes the selected factor columns for each target date immediately.
10. It drops the batch data and moves to the next batch.

This favors low peak memory. Smaller `--factor-batch-size` reduces memory more
and increases local IO. Larger batches reuse loaded data better but keep more
intermediate columns alive.

`--profile` is the fastest way to inspect this tradeoff. It prints per-batch
load, compute, and write timings plus row and non-null counts.

## Output Semantics

Official factor output:

```text
data/factors/stock/daily/{year}/{trade_date}.parquet
```

If the file does not exist, it is created. If it exists, the writer loads it,
merges the newly computed columns, and rewrites the file. Existing unrelated
columns are preserved.

Insufficient lookback history usually means the factor column contains `null`
for that date. Labels are different: if the future lookahead is not available,
the label engine skips that target date and writes no label file.

## Validation Checklist

Before committing a factor change:

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo check --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor --profile
```

Common issues:

- `missing required column ts_code`: a daily file is malformed or the wrong file
  layout was read.
- stale metadata: rerun `metadata` after adding or renaming factors.
- all-null output: check lookback, missing input dates, industry membership, and
  whether PIT data is available on the target date.
- `STATUS_CONTROL_C_EXIT`: the process was interrupted from the terminal; it is
  not a Rust panic.
