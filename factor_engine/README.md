# YuminQuant Factor Engine / Rust 因子引擎

`factor_engine` 是 YuminQuant 的 Rust 计算核心，负责正式因子、分钟 raw 缓存、Barra CNE6 暴露、future-return labels、截面回测和事件驱动策略模拟。

`factor_engine` is the Rust compute core for formal factors, minute-to-daily raw cache, Barra CNE6 exposures, future-return labels, cross-sectional backtests, and event-driven strategy simulation.

## 输出 / Outputs

```text
data/factors/{asset}/{frequency}/{year}/{trade_date}.parquet
data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet
data/derived/stock/bar/{bar_size}m/{year}/{trade_date}.parquet
data/barra/{asset}/daily/CNE6/{year}/{trade_date}.parquet
data/label/{asset}/{frequency}/{year}/{trade_date}.parquet
data/backtest/stock/daily/{factor_id}/{returns,ic,factor_stats,holdings,industry_weights,barra_exposure,index_group_returns}.parquet
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

宽表输出会保留已有无关列，只更新本次计算列。Wide parquet outputs preserve unrelated existing columns and overwrite only selected columns.

## 基础命令 / Core Commands

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- help
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine\Cargo.toml -- plan --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors utd
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors utd --profile
```

常用因子运行方式 / Common factor runs:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --tags GFZQ --factor-batch-size 20 --profile
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --tags XYZQ --factor-batch-size 20 --date-batch-size 120 --profile --refresh-minute-cache
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset future --frequency daily --start-date 20260424 --end-date 20260424 --profile
```

Important flags:

```text
--factors a,b,c
--tags XYZQ,GFZQ
--factor-batch-size N     run default 64
--date-batch-size N       run default 1
--threads N
--profile
--refresh-minute-cache
```

## 派生数据 / Derived Data

`derive-bar` 会基于原始 1min 数据生成可复用的股票分钟派生 bar。该命令会并行处理多个交易日；
`--date-batch-size N` 用于控制并发日期数量，默认值为 `20`。

`derive-bar` builds reusable stock minute bars from raw 1m data. It processes
multiple trading days in parallel; `--date-batch-size N` controls concurrent
dates and defaults to `20`.

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 15 --start-date 20110101 --end-date 20260424
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 120 --start-date 20260424 --end-date 20260424 --date-batch-size 20
```

股票分钟 `bar_size` 必须是 240 的因子，且满足 `1 < bar_size <= 120`；其中
`120` 表示上午一根 bar、下午一根 bar。输出路径为
`data/derived/stock/bar/{bar_size}m/{year}/{trade_date}.parquet`。

Allowed stock minute `bar_size` values are divisors of 240 with
`1 < bar_size <= 120`; `120` means one morning bar and one afternoon bar.
Output is written to `data/derived/stock/bar/{bar_size}m/{year}/{trade_date}.parquet`.

Rust 因子 raw provider 可以通过通用 `stock.derived.bar` 数据源和 `bar_size`
请求派生 bar。已迁移的 5m provider 会优先读取 `data/derived/stock/bar/5m`；
如果派生文件缺失、必要列缺失或结构不兼容，则输出 warning 并回退到原始 1min 数据。
1min fallback 采用懒加载：只有缺少或无法读取派生 5m bar 的交易日才会额外读取原始 1min。
fallback 以交易日为粒度，因此同一个交易日不会混合使用派生 bar 和原始分钟数据。

Rust factor raw providers can request derived bars through the generic
`stock.derived.bar` data source with a `bar_size`. Migrated 5m providers prefer
`data/derived/stock/bar/5m` and fall back to raw 1m data with a warning when the
derived file or required columns are unavailable. The 1m fallback is lazy-loaded:
raw 1m data is loaded only for trading dates whose derived 5m bar is missing or
unreadable. The fallback is date-level, so a single trading day is not mixed
between derived and raw minute sources.

当前已迁移到派生 bar 的 provider 只包含标准、非重叠 5m 家族：`patv`、DBZQ
`volume_price_distribution` / `significant_up_volume_return_distribution`、GFZQ
`str_5min_ma*`，以及 `umr_minute_volatility` / `umr_minute_skewness`。需要
`09:30` 锚点、交错子网格、rolling 5m、3m bar 或完整 1min 矩阵的因子仍然保留在原始
1min 数据路径上。

Current derived-bar migrated providers are the standard non-overlapping 5m
families: `patv`, DBZQ `volume_price_distribution` /
`significant_up_volume_return_distribution`, GFZQ `str_5min_ma*`, and
`umr_minute_volatility` / `umr_minute_skewness`. Anchored, staggered, rolling, 3m,
or full 1m-matrix factors intentionally remain on raw 1m data.

增量更新 workflow 会在 `stock_minute` 后接入 `stock_derived_bar`，默认生成 5m/15m bar：

The incremental workflow includes `stock_derived_bar` after `stock_minute` and
builds 5m/15m bars by default:

```powershell
python scripts\update_incremental.py --groups stock_minute stock_derived_bar --start-date 20260424 --end-date 20260424
python scripts\update_incremental.py --groups stock_derived_bar --derived-bar-sizes 5,15 --start-date 20260424 --end-date 20260424
```

## Barra 与 Label / Barra And Labels

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- barra-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- barra-list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine\Cargo.toml -- barra-run --asset stock --frequency daily --start-date 20200101 --end-date 20201231 --families DIVIDEND_YIELD,GROWTH,LIQUIDITY,MOMENTUM,QUALITY,SENTIMENT,VALUE,VOLATILITY --date-batch-size 240 --profile

cargo run --release --manifest-path factor_engine\Cargo.toml -- label-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-run --asset stock --frequency daily --start-date 20260101 --end-date 20260424 --label-batch-size 20 --profile
```

`--exposure-batch-size N` 控制 Barra 每批计算多少个 exposure family，默认 `1`。较大的值可以复用数据，但内存更高。

`--exposure-batch-size N` controls how many Barra exposure families are computed per batch. Larger batches reuse loaded data but use more memory.

## 截面回测 / Cross-Sectional Backtest

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --factors utd --groups 10 --rebalance 5
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --tags XYZQ --groups 10 --rebalance weekly --factor-batch-size 10 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --all-factors --groups 10 --rebalance 5 --factor-batch-size 10 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors mdl_000006 --factor-root data\models --factor-fill ffill --groups 10 --rebalance 20
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --all-factors --factor-root data\barra\stock\daily\CNE6 --groups 10 --rebalance 20
```

Backtest 参数 / Backtest flags:

```text
--factors a,b,c              Explicit factor columns.
--tags XYZQ                  Select formal factors by metadata tag.
--all-factors                All non-deprecated formal factors, or all columns under --factor-root.
--factor-root data\models    External direct daily factor root.
--factor-fill none|ffill     Forward-fill missing low-frequency factor snapshots only.

--label future_vwap_return_1d
--groups 5|10|20
--rebalance daily|5|10|weekly|biweekly|monthly|quarterly

--universe mkt_all|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id
--benchmark mkt_mean|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id

--neutralize none
--neutralize sector
--neutralize barra:SIZE
--neutralize barra:SIZE+sector
--neutralize barra:all+sector

--detail none|holdings|industry_weights|all
--detail-sector sw_l1|ci_l1

--exclude-limit true|false
--exclude-st true|false
--limit-side both|up|down
--factor-batch-size 10
--date-batch-size 120
--threads N
```

`sector` 表示申万一级行业。`barra:all` 只代表 CNE6 一级风格暴露：

`sector` means Shenwan level-1 sector. `barra:all` means only primary CNE6 style exposures:

```text
DIVIDEND_YIELD, GROWTH, LIQUIDITY, MOMENTUM, QUALITY, SENTIMENT, SIZE, VALUE, VOLATILITY
```

它不会把二级/三级 Barra 列如 `DTOP`、`BTOP`、`Beta` 放进回归右端。如果目标因子本身也在控制变量里，会先从控制变量里移除。

Secondary or tertiary Barra columns such as `DTOP`, `BTOP`, and `Beta` are not included. If the target factor is also in the control list, it is removed before neutralization.

Backtest outputs:

```text
data/backtest/stock/daily/{factor_id}/returns.parquet
data/backtest/stock/daily/{factor_id}/ic.parquet
data/backtest/stock/daily/{factor_id}/factor_stats.parquet
data/backtest/stock/daily/{factor_id}/holdings.parquet
data/backtest/stock/daily/{factor_id}/industry_weights.parquet
data/backtest/stock/daily/{factor_id}/barra_exposure.parquet
data/backtest/stock/daily/{factor_id}/index_group_returns.parquet
```

`holdings` 和 `industry_weights` 只在 `--detail holdings|industry_weights|all` 时写出。多头端点由 `sign(mean(RankIC))` 决定：非负取 `group_N`，负数取 `group_1`。

`holdings` and `industry_weights` are written only when detail output is requested. The long endpoint is selected by `sign(mean(RankIC))`.

`barra_exposure` and `index_group_returns` are default diagnostics. `index_group_returns` always targets `000300.SH`, `000905.SH`, and `000852.SH`; missing index members or benchmark data are represented as `NaN` rows instead of aborting the main backtest.

IC decay uses the selected 1d label only. For each factor date, the engine computes Pearson IC against the same 1d future-return cross-section shifted by 0..19 trading days and writes `horizon=1..20`. RankIC is computed only for `horizon=1` to keep decay diagnostics lighter.

### External Parquet Schema Repair

If an external `--factor-root` fails in backtest with Arrow schema errors such
as `LargeUtf8`, `trade_date is i64`, or non-`float32` value columns, normalize
the files in place with:

```powershell
python scripts\cast_output_value_columns.py --root C:\Users\Devin\Desktop\Pred --start-date 20110101 --end-date 20260424 --dry-run
python scripts\cast_output_value_columns.py --root C:\Users\Devin\Desktop\Pred --start-date 20110101 --end-date 20260424
```

The script scans parquet files by date, casts key columns to Rust-compatible
types (`trade_date=int32`, `ts_code/trade_time=utf8`), casts numeric output
columns to `float32`, and atomically replaces only files that need changes.

## Strategy Run / 事件驱动策略

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\strategy_001.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml --detail true
```

`strategy-run` 读取策略 TOML，按 daily 或 minute bar 事件推进账户。策略只下单，引擎负责撮合、费用、滑点、持仓和 PnL。普通 `on_bar` 订单默认下一根 bar open 成交；`on_session_open` 可用于第一根 bar 开盘成交。

`strategy-run` reads a strategy TOML and drives the account by daily or minute bar events. Strategies place orders; the engine handles execution, costs, positions, and PnL. Normal `on_bar` orders fill at the next bar open by default.

Output:

```text
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

分钟策略默认 `detail=false`，每个交易日写一行日终 snapshot。传 `--detail true` 或 TOML `[output] detail = true` 时，逐分钟写出。

Minute strategies default to daily snapshots. Use `--detail true` or `[output] detail = true` for one row per minute bar.

Strategy development guide: [STRATEGY_README.md](STRATEGY_README.md)

## 执行模型 / Execution Model

普通因子按 `date_batch x factor_batch` 执行。每个 batch 合并依赖、按 lookback 读取所需日期和列、构造 `DataPool`、并行计算、立即写出并释放内存。

Ordinary factor runs execute by `date_batch x factor_batch`. Each batch merges dependencies, loads needed dates and columns, builds `DataPool`, computes in parallel, writes output, then releases memory.

分钟日频因子分两层：

Minute-to-daily factors have two layers:

1. `minute_compute()` 将单日分钟数据降维成日频 raw，写入 `_cache/intraday_daily`。
2. `compute()` 读取日频 raw，再做 `ts_mean`、`cs_zscore`、neutralization 等后处理。

Use `--refresh-minute-cache` when raw formulas change.

### Deprecated Factors And Intraday Raw / deprecated 因子与分钟 raw

中文规则：

- `deprecated` 因子不会进入 `--all-factors`、`--tags` 等批量选择，也不会贡献新的 raw requirements。
- 如果 deprecated 因子独占某个 raw id，该 raw 不应被计算、不应写入 `_cache/intraday_daily`，也不应参与最终 `compute()`。
- 如果 active 因子仍依赖同一个 raw id，则该 raw 继续正常计算；deprecated 只影响因子选择，不会阻断 active 因子的共享依赖。
- 多 raw provider 必须按本次 `raw_ids` 请求集合计算。允许保留必要共享前置状态，例如分钟收益、5min bar、状态矩阵；但不能顺手计算未请求的 sibling raw 指标分支。
- 开发多 raw provider 时优先使用 `RequestedRawIds`，在具体指标分支前判断 `requested.contains(raw_id)` 或 `requested.contains_any([...])`。

English rules:

- Deprecated factors are excluded from broad selection such as `--all-factors` and `--tags`, and therefore should not add new raw requirements.
- A raw id used only by deprecated factors should not be computed, written to `_cache/intraday_daily`, or consumed by final `compute()`.
- If an active factor still depends on the same raw id, that raw remains active; deprecation only removes retired factor selection.
- Multi-raw providers must be driven by the current requested `raw_ids`. Shared setup such as minute returns, 5-minute bars, or state matrices is allowed, but unrequested sibling metric branches must not be computed opportunistically.
- Prefer `RequestedRawIds` in new multi-raw providers and guard concrete metric branches with `requested.contains(raw_id)` or `requested.contains_any([...])`.

## 开发入口 / Development Guides

- Factor development: [FACTOR_DEVELOPMENT_README.md](FACTOR_DEVELOPMENT_README.md)
- Short factor development navigation: [docs/FACTOR_DEVELOPMENT.md](docs/FACTOR_DEVELOPMENT.md)
- Strategy development: [STRATEGY_README.md](STRATEGY_README.md)

常用验证命令 / Common validation:

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo check --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
```
