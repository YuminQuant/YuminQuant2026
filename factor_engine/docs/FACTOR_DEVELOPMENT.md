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

## Financial Statements

Financial statement helpers are point-in-time. They prefer `f_ann_date` over
`ann_date` and only expose records whose disclosure date is on or before the
target trading date. For duplicate `(ts_code, end_date)` records, the as-of
version with the latest disclosure date is used; ties prefer `update_flag=1`.

The `roe_8q` demo uses the latest 8 disclosed quarters:

```text
sum(n_income_attr_p, 8 quarters) / 2 / mean(total_hldr_eqy_exc_min_int, 8 quarters)
```

## Operators

- Time-series operators live in `src/operators/time_series/`.
- Cross-sectional operators live in `src/operators/cross_sectional/`.
- Keep factor expressions in factor files. Put reusable math in operators and
  data-shaping helpers in `factor/common/`.

## Profiling

Pass `--profile` to `run` to print per date batch and factor batch timings:

```powershell
cargo run --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260105 --end-date 20260130 --factors pe_zscore_60d,roe_8q --profile
```

The profile includes load, compute, write milliseconds, row count, and non-null
count for each factor.

## Common Pitfalls

- Short factor IDs are scoped by asset and frequency. Use `--asset` and
  `--frequency` with `--factors return_1d`.
- Old output parquet files may still contain long columns such as
  `stock__daily__pv__return_1d`. Clear the target output date range before
  regenerating if you want only short columns.
- `ts_sum(ts_pctchg(close, 1), 20)` is a sum of 20 one-period returns, not
  `close_t / close_{t-20} - 1`.
