# Factor Development Guide

## Workflow

1. Create one factor per `.rs` file under `src/factor/{asset}/{frequency}/`.
2. Use a short snake_case factor ID equal to the file stem.
3. Declare dependencies precisely; the engine loads only requested columns for
   the current factor batch.
4. Run `metadata` after adding or editing factor specs.
5. Use `plan` to confirm selected factors and loaded columns, then use `run`.

## Daily Panel Expressions

`DailyPanel` is the preferred interface for nested time-series and
cross-sectional expressions:

```rust
let panel = DailyPanel::from_table(data.daily(DatasetId::StockDailyPv)?, context)?;
let factor = panel
    .column("close")?
    .ts(|values| ts_pctchg(values, 1))?
    .ts(|values| ts_sum(values, 20, 20))?
    .cs(|values| cs_pctrank(values, true))?;
Ok(factor.to_factor_series(self.spec()))
```

Use `ts_binary` for multi-column time-series expressions such as
`ts_corr(volume, close)`, and `cs_binary` for multi-column cross-sectional
expressions such as regression residuals.

## Minute To Daily Expressions

Minute-to-daily factors use two compute layers. A factor exposes
`intraday_raw_specs()` and `minute_compute()` in the same `.rs` file that
contains its final `compute()`. The engine materializes missing raw daily
vectors from minute data into the local cache, loads the requested raw columns
into `DataPool`, and then calls the ordinary final `compute()`:

```rust
let panel = DailyPanel::from_table(
    data.intraday_daily_raw("ret_over_sqrt_vol_mean")?,
    context,
)?;
let raw = panel.column("ret_over_sqrt_vol_mean")?;
let factor = raw.ts(|values| ts_mean(values, 20, 20))?;
Ok(factor.to_factor_series(self.spec()))
```

Final factors declare raw dependencies in `FactorSpec::intraday_raw_dependencies`.
The raw expression is discovered from registered factors, so concrete raw
formula code should stay in factor files. `factor/common/` only provides
generic intraday grouping, window, cache-table, and numeric helpers.

Raw cache parquet output is enabled by default and lives under
`data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet` for
stock raw factors. Use `--refresh-minute-cache` to force raw recomputation.

For factors that do not need daily post-processing, keep the same shape:
`minute_compute()` creates the raw value and `compute()` simply returns the raw
column as the final factor. Cross-day minute concat factors are intentionally
kept for a later design pass.

```rust
let panel = DailyPanel::from_table(
    data.intraday_daily_raw("ret_over_sqrt_vol_mean")?,
    context,
)?;
let factor = panel.column("ret_over_sqrt_vol_mean")?;
Ok(factor.to_factor_series(self.spec()))
```

## Financial Statements

Financial statement helpers are point-in-time. They prefer `f_ann_date` over
`ann_date` and only expose records whose disclosure date is on or before the
target trading date. Report type preference is explicit: income factors can
prefer adjusted single-quarter reports before regular single-quarter reports,
while balance sheet factors can prefer consolidated point-in-time reports. For
duplicate `(ts_code, end_date, report_type)` records, the as-of version with the
latest disclosure date is used; ties prefer `update_flag=1`.

The `roe_8q` demo uses the latest 8 valid quarters and applies the regulatory
deadline rule for Q1/Q4, Q2, and Q3:

```text
mean(n_income_attr_p / total_hldr_eqy_exc_min_int, 8 quarters)
```

## Operators

- Time-series operators live in `src/operators/time_series/`.
- Cross-sectional operators live in `src/operators/cross_sectional/`.
- Keep factor expressions in factor files. Put reusable math in operators and
  data-shaping helpers in `factor/common/`.

## Profiling

Pass `--profile` to `run` to print per target-date batch and factor batch
timings. The engine processes one trading date at a time, reusing that date's
loaded data across the selected factors in the current factor batch:

```powershell
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260105 --end-date 20260130 --factors pe_zscore_60d,roe_8q --profile
```

The profile includes the execution stage, load, compute, write milliseconds,
row count, and non-null count for each factor. Intraday daily factors show
`intraday_raw_materialize_window_N` for the in-memory raw step, followed by
`intraday_daily_postprocess_lookback_N` for final daily expressions.

## Common Pitfalls

- Short factor IDs are scoped by asset and frequency. Use `--asset` and
  `--frequency` with `--factors return_1d`.
- Old output parquet files may still contain long columns such as
  `stock__daily__pv__return_1d`. Clear the target output date range before
  regenerating if you want only short columns.
- `ts_sum(ts_pctchg(close, 1), 20)` is a sum of 20 one-period returns, not
  `close_t / close_{t-20} - 1`.
