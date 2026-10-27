# Daily factor storage: manual and ML partitions

Both the local filesystem and the HDFS artifact backend use the same layout:

```text
data/factors/stock/daily/<year>/<YYYYMMDD>.parquet       # manual factors
data/factors/stock/daily/<year>/<YYYYMMDD>_ml.parquet    # ml_alpha factors
```

Each file contains unique `trade_date, ts_code` keys and its own factor columns.
ML prediction, grouped prediction and report-text factors all write the `_ml`
partition. Model signals under `data/models` retain their existing layout.

## Reading

`FactorFiles` is the public logical daily reader in Python and Rust. It projects
only requested columns, routes them by ownership, and joins required partitions
on `trade_date, ts_code` with an outer join. Invalid, null or duplicate row keys
raise an error. Factor-frame models, their graph/sequence histories, Rust
backtests and the Python backtest bridge use this logical view. Date scans
recognize both names and deduplicate dates, including dates with only ML output.

Python callers can use:

```python
from yq_ml_alpha.data.factor_files import FactorFiles

factors = FactorFiles("data/factors/stock/daily")
frame = factors.read(20260105, ["some_manual_factor", "logsig_alpha_v"])
```

Factor metadata now publishes `storage_partition="manual"|"ml"` for every row.
Old metadata infers ownership from the reserved `model_generated` tag. Two IDs
cannot claim the same output column, and an ML writer cannot update a registered
manual column. Existing `factor_metadata.rust.parquet` and
`factor_metadata.ml_alpha.parquet` remain the metadata sources; their union is
published as `factor_metadata.parquet`.

For old mixed daily files, an ML column is read from the manual file until that
column exists in the ML file. Once present in `_ml`, it owns the whole date's
cross-section: explicit nulls and absent stocks never fall back to old values.
Writers do not pad other ML columns merely because they exist on another date;
this prevents a new null column from masking an unmigrated legacy factor.

## Writing and concurrency

Each writer locks only its own physical daily file, merges that partition's
existing columns, and publishes a complete Parquet via atomic replacement.
Manual production and ML prediction can run concurrently. Several ML writers
still share the `_ml.parquet.lock` lock; grouped prediction merges multiple ML
columns with one write per date. A missing ML file obtains its initial stock keys
from the configured daily price source, without reading the manual factor table.

Splitting output does not remove input dependencies: a factor-combination model
must still wait for the manual input columns it actually consumes.

## Migrating existing mixed files

Readers support the old format immediately; installing the code does not bulk
rewrite the historical library. To physically move legacy ML columns, refresh
factor metadata and use the production Rust command through the regular entry:

```powershell
# Inspect first. --factor-root is the daily root, not the metadata root.
python scripts/yq.py migrate-factor-files `
  --start-date 20160101 --end-date 20260424 `
  --factor-root data/factors/stock/daily --dry-run

# Apply the same range by omitting --dry-run.
python scripts/yq.py migrate-factor-files `
  --start-date 20160101 --end-date 20260424 `
  --factor-root data/factors/stock/daily
```

The same Rust command works through a deployment adapter with an HDFS backend.
Migration takes locks in the fixed order manual then ML, preserves existing ML
columns (including nulls), publishes ML first, and only then removes moved columns
from the original file. It is safe to rerun after interruption; it does not delete
an existing ML score or infer ownership from column names.

`remove-factor-columns` and `scripts/remove_factor_columns.py` scan both names
and remove requested columns from both physical locations, including leftover
legacy copies. Removal preserves row keys and locks each rewritten file.
