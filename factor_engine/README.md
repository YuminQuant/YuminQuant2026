# YuminQuant Factor Engine MVP

This is the first Rust factor engine scaffold for YuminQuant. It focuses on
price-volume factors and reads the existing local parquet data lake directly.

## Factor Layout

Factors are grouped by asset domain. Each concrete factor lives in its own
`.rs` file:

```text
src/factor/
  chn_stock/
    daily/
      return_1d.rs
      momentum_20d.rs
      volume_ratio_20d.rs
      volatility_20d.rs
    minute/
      return_1m.rs
  future/
    daily/
      return_1d.rs
      momentum_20d.rs
      volatility_20d.rs
    minute/
      return_1m.rs
  registry.rs
src/operators/
  ts_mean.rs
  ts_std_dev.rs
  ts_zscore.rs
  ts_corr.rs
```

Each concrete factor file owns its factor expression. Shared math and
time-series building blocks live under `src/operators/`, for example
`ts_mean`, `ts_std_dev`, `ts_zscore`, and `ts_corr`. The registry is generated
at build time by scanning the factor directory.

## Design Decisions

- Source data remains in the Python-managed `data/` directory.
- Daily data is stored by year in the source lake, so the loader uses the
  trading calendar to compute the warmup start date and only opens the years
  that overlap the requested window.
- Minute data is loaded by trading day, matching the existing source layout.
- Factor requirements are grouped by dataset and projected columns are loaded
  once per run. This is the memory model intended for large factor batches.
- Factor outputs are written as wide daily files:

```text
data/factors/{asset_class}/{frequency}/YYYY/YYYYMMDD.parquet
```

Each output file contains key columns plus all factor columns computed or
previously stored for that date. This avoids producing one file per
factor/date pair when the factor count grows.

Factor metadata is written to:

```text
data/factors/factor_metadata.parquet
```

The metadata table includes tags, dependency metadata, descriptions, and the
output column name. Run the `metadata` command after adding, removing, or
editing factor metadata.

## Examples

From the repository root:

```powershell
cargo run --manifest-path factor_engine/Cargo.toml -- metadata
cargo run --manifest-path factor_engine/Cargo.toml -- list --asset stock --frequency daily
cargo run --manifest-path factor_engine/Cargo.toml -- plan --asset stock --frequency daily --start-date 20260105 --end-date 20260109
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260105 --end-date 20260109
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset future --frequency minute_1m --start-date 20260424 --end-date 20260424
```

Run specific factors:

```powershell
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260105 --end-date 20260109 --factors stock.daily.pv.return_1d
```

Run by tags:

```powershell
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260105 --end-date 20260109 --tags momentum
```

Run factors in batches by source path:

```powershell
python factor_engine\scripts\run_factor_batches.py --asset chn_stock --frequency daily --start-date 20260105 --end-date 20260109 --batch-num 20
```

Batch mode computes one group of factors at a time and appends/merges those
columns into the same per-date wide parquet files. This keeps memory bounded by
the selected batch instead of keeping every factor result in memory.
