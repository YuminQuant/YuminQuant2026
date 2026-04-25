# YuminQuant

YuminQuant is a local quantitative data lake and data synchronization toolkit built around Tushare and parquet files. The current project focuses on collecting market, calendar, static, financial, and alternative datasets into a structured local directory so that future factor research and backtesting can read from a consistent offline source.

## Current Scope

The existing codebase mainly contains:

- Tushare client and configuration management.
- Downloaders for calendars, A-shares, ETFs, futures, options, indexes, Hong Kong stocks, US stocks, financial statements, dividends, and analyst reports.
- Incremental update scripts for keeping local parquet data up to date.
- Documentation for downloader granularity, update commands, and the Rust factor engine scaffold.

The local `data/` directory is intentionally not tracked by Git. It can be large and machine-specific.

## Project Structure

```text
YuminQuant/
  data_manager/
    core/                 # Config, logger, Tushare client, downloader base class
    downloader/           # Asset-specific downloader implementations
  factor_engine/          # Rust factor engine scaffold
  scripts/
    init_*.py             # Historical initialization scripts
    update_incremental.py # Incremental update entry point
  docs/
    DOWNLOAD_GRANULARITY.md
    UPDATE_COMMANDS.md
    FACTOR_MODULE_DESIGN.md
  config.example.toml     # Sanitized config template
  config.toml             # Local private config, ignored by Git
```

## Setup

Create or activate a Python environment, then install the required runtime packages:

```powershell
pip install pandas numpy pyarrow tqdm tomli tushare
```

Copy the example configuration and fill in your own Tushare token:

```powershell
copy config.example.toml config.toml
```

Then edit `config.toml`:

```toml
[api]
tushare_token = "YOUR_TUSHARE_TOKEN"
```

The default data root in the template is:

```text
D:/yuminwu_workspace/Internship/YuminQuant/data
```

Change `paths.base_data_dir` if you keep the data lake elsewhere.

## Updating Data

The recommended update order is:

1. Static data and calendars.
2. Daily data.
3. Minute data.

For the current main A-share and futures pipeline, a safe update flow from `20260214` is:

```powershell
python scripts\update_incremental.py --groups calendar stock_static future_static
python scripts\update_incremental.py --groups stock_daily future_daily --start-date 20260214
python scripts\update_incremental.py --groups stock_minute future_minute --start-date 20260214
```

To update all static tables:

```powershell
python scripts\update_incremental.py --groups static
```

See [docs/UPDATE_COMMANDS.md](docs/UPDATE_COMMANDS.md) for more command examples.

## Downloader Granularity

Different downloaders use different request patterns:

- Many daily datasets are fetched by trading day as full-market cross sections.
- Stock, ETF, and futures minute datasets are organized by trading day, with internal code batching.
- Index time series are usually fetched by index code.
- Financial datasets are usually fetched by reporting period or announcement date.
- Static datasets are usually full snapshots.

See [docs/DOWNLOAD_GRANULARITY.md](docs/DOWNLOAD_GRANULARITY.md) for details.

## Git And Data Safety

The repository ignores:

- `config.toml`, because it contains local secrets.
- `data/`, because it contains large generated parquet data.
- Python caches and local editor files.

Before pushing to a remote repository, check:

```powershell
git status
git ls-files
```

This helps confirm that no token or large local data file is being tracked.
