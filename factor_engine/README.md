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
data/backtest/stock/daily/{returns,ic,factor_stats,holdings,industry_weights}/
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

## Derived Data / Derived Bars

`derive-bar` builds reusable stock minute bars from raw 1m data. It processes
multiple trading days in parallel; `--date-batch-size N` controls concurrent
dates and defaults to `20`.

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 15 --start-date 20110101 --end-date 20260424
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 120 --start-date 20260424 --end-date 20260424 --date-batch-size 20
```

Allowed stock minute `bar_size` values are divisors of 240 with
`1 < bar_size <= 120`; `120` means one morning bar and one afternoon bar.
Output is written to `data/derived/stock/bar/{bar_size}m/{year}/{trade_date}.parquet`.

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
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --factor-fill ffill --groups 10 --rebalance 20
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
data/backtest/stock/daily/returns/{factor_id}.parquet
data/backtest/stock/daily/ic/{factor_id}.parquet
data/backtest/stock/daily/factor_stats/{factor_id}.parquet
data/backtest/stock/daily/holdings/{factor_id}.parquet
data/backtest/stock/daily/industry_weights/{factor_id}.parquet
```

`holdings` 和 `industry_weights` 只在 `--detail holdings|industry_weights|all` 时写出。多头端点由 `sign(mean(RankIC))` 决定：非负取 `group_N`，负数取 `group_1`。

`holdings` and `industry_weights` are written only when detail output is requested. The long endpoint is selected by `sign(mean(RankIC))`.

## Strategy Run / 事件驱动策略

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\ml_xgb_top20.toml
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
