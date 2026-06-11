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

### Multi-output Factor Providers / 多输出因子 Provider

适用场景：

- 多个正式因子共享同一套昂贵前置计算，例如财务向量、截面相似度矩阵、peer network、同一套复杂截面状态。
- 不适用于普通单因子；普通因子继续只写 `spec()` 和 `compute()`。

后续开发流程：

1. 在 `factor/common/` 下写一个共享 provider 模块，例如 `financial_similarity.rs`。
2. 共享 provider 模块负责：
   - 定义所有正式输出 id 常量。
   - 提供 `spec(kind)` 或类似 helper 生成各正式因子的 `FactorSpec`。
   - 提供 `compute_requested(requested_ids, context, data)`，一次性构造共享前置状态，并只计算本次请求的输出分支。
3. 在 `factor/chn_stock/daily/` 下为每个正式输出保留一个很薄的 wrapper 文件，例如：
   - `f_momentum_80pec.rs`
   - `link_new.rs`
4. 每个 wrapper 的 `spec()` 返回自己的正式因子 metadata。
5. 同一组 wrapper 的 `compute_provider_key()` 必须返回同一个稳定 key，例如：
   `stock|daily|financial_similarity`。
6. 每个 wrapper 的 `compute_many(requested_ids, context, data)` 转发到共享 provider 的 `compute_requested(...)`。
7. 每个 wrapper 的 `compute()` 只作为单因子兼容入口，通常用单个自身 id 调用 `compute_requested(...)`，并取回对应 `FactorSeries`。
8. 在共享 provider 内部必须做 requested-aware 分支：
   - 共享前置计算可以统一做。
   - 只请求 `link_new` 时，不应计算 `f_momentum_80pec` 独有的 top-peer Ret20 分支。
   - 只请求某个输出时，不应返回未请求 sibling factor。
9. 如果某个 sibling factor 后续被 `deprecated`，它不会进入 `requested_ids`；provider 独有分支也不应运行，除非 active factor 仍需要同一共享前置状态。

Development workflow:

1. Add a shared provider module under `factor/common/`, for example `financial_similarity.rs`.
2. The shared provider should:
   - Define all formal output id constants.
   - Provide `spec(kind)` or an equivalent helper for each formal `FactorSpec`.
   - Provide `compute_requested(requested_ids, context, data)`, which builds shared setup once and computes only requested output branches.
3. Keep one thin wrapper file per formal output under `factor/chn_stock/daily/`, for example:
   - `f_momentum_80pec.rs`
   - `link_new.rs`
4. Each wrapper's `spec()` returns its own formal factor metadata.
5. Wrappers in the same provider group must return the same stable `compute_provider_key()`, for example:
   `stock|daily|financial_similarity`.
6. Each wrapper's `compute_many(requested_ids, context, data)` delegates to the shared provider.
7. Each wrapper's `compute()` remains the single-factor compatibility entry point; it usually calls `compute_requested(...)` with only its own id and returns the matching `FactorSeries`.
8. The shared provider must be requested-aware:
   - Shared setup may be computed once.
   - If only `link_new` is requested, do not compute the top-peer Ret20 branch unique to `f_momentum_80pec`.
   - Do not return unrequested sibling factors.
9. If a sibling factor is later marked `deprecated`, it will not enter `requested_ids`; deprecated-only branches should not run unless an active factor still needs the shared setup.

Engine 兼容性：

- 普通旧因子只需要实现 `spec()` 和 `compute()`；`provided_specs()`、`compute_provider_key()`、`compute_many()` 都有默认实现，因此旧代码无需改动。
- 默认 `provided_specs()` 返回单个 `spec()`；默认 `compute_provider_key()` 返回该因子的 registry key；默认 `compute_many()` 只在本因子被请求时调用原来的 `compute()`。
- 如果多个正式因子共享昂贵的前置计算，例如同一组财务向量、同一个截面相似度矩阵或同一套 peer 网络，应让这些 wrapper 返回相同的 `compute_provider_key()`，并由 provider 覆盖 `compute_many(requested_ids, context, data)` 一次性返回多个 `FactorSeries`。
- 引擎会在同一个 factor batch 内按 `compute_provider_key()` 分组，每个 provider 只调用一次 `compute_many()`；随后校验返回结果必须覆盖所有 requested factor id、不得返回未请求或重复因子，并按原请求顺序写出。
- `compute_many()` 必须 requested-aware：只计算本次 `requested_ids` 需要的独有分支。共享前置状态可以保留，但不能顺手计算未请求 sibling factor 的昂贵指标分支。
- `deprecated` 因子不会进入 selected factors，因此也不会进入 provider 的 `requested_ids`；除非 active 因子仍共享同一正式输出或同一必要前置计算，否则 deprecated 独有分支不应被计算。

Engine compatibility:

- Legacy single-output factors only need `spec()` and `compute()`. `provided_specs()`, `compute_provider_key()`, and `compute_many()` have backward-compatible defaults, so existing factors do not need changes.
- By default, `provided_specs()` returns one `spec()`, `compute_provider_key()` returns the factor registry key, and `compute_many()` delegates to the original `compute()` only when the factor id is requested.
- When multiple formal factors share expensive setup, such as financial vectors, cross-sectional similarity matrices, or peer networks, their thin wrappers should return the same `compute_provider_key()` and override `compute_many(requested_ids, context, data)` to return all requested `FactorSeries` in one provider call.
- The engine groups selected factors in the same factor batch by `compute_provider_key()`, calls each provider once, validates that all requested ids are returned with no extra or duplicate factors, and then restores the original request order before writing.
- `compute_many()` must be requested-aware: compute only branches needed by the current `requested_ids`. Shared setup is allowed, but expensive sibling-factor metric branches must not be computed opportunistically.
- Deprecated factors do not enter selected factors and therefore do not enter provider `requested_ids`; deprecated-only branches should not run unless an active factor still shares the same required output or setup.

### Sparse DataRequest Date Policy / 稀疏日期读取策略

中文规则：

- `DataRequest::new(...)` 保持旧语义，但在加载前会按当前因子自己的 context 解析成显式日期。
- 如果因子只需要一组特定日期，在 `requirements_for_context(context)` 内计算 `Vec<i32>`，再传给 `DataRequest::explicit_dates(...)`。
- 周末点、月末点、季末点、固定检查日、事件日都不需要新增专用构造器；它们只是不同的本地日期生成函数。
- 同一 dataset 的多个请求在物理 IO 层会按列和日期做 union，避免重复读取。
- compute 层会按 provider 自己声明的请求创建隔离视图；一个因子的稀疏/长窗口日期不会污染另一个因子的 `DailyPanel` 或日频 raw table。

English rules:

- `DataRequest::new(...)` keeps the old meaning, but before loading it is resolved into explicit dates using the current factor's context.
- If a factor only needs a selected set of dates, compute a `Vec<i32>` in `requirements_for_context(context)` and pass it to `DataRequest::explicit_dates(...)`.
- Week ends, month ends, quarter ends, fixed checkpoints, and event dates do not need dedicated constructors; they are just different local date generators.
- Multiple requests for the same dataset are merged by columns and date union at the physical IO layer.
- At compute time, each provider receives a request-scoped `DataPool` view, so one factor's sparse or long-window dates do not pollute another factor's `DailyPanel` or daily raw table.

示例 / Example:

```rust
fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
    let dates = custom_event_or_sample_dates(context);
    vec![DataRequest::explicit_dates(DatasetId::StockDailyPv, &["close"], dates)]
}
```

### Financial Factor Update Policy / 财务因子更新策略

财务因子不能仅凭 `fundamental` tag 自动切换更新频率，必须由 factor/provider 显式声明 `update_policy()`。

Financial factor update frequency is never inferred from the `fundamental` tag. Each factor or provider must explicitly declare `update_policy()`.

Current panel rule: financial, main-business, dividend, and analyst event
factors use `data.stock_universe_panel()?` as the output grid. Do not request
`DataRequest::new(DatasetId::StockDailyPv, &["close"])` only to obtain a panel.
Only keep `StockDailyPv` dependencies when price/return data is a real formula
input, then align those columns to the stock universe panel by
`(trade_date, ts_code)`.

可选策略 / Policies:

- `Daily`：默认策略。估值、价格、收益率、市值、换手率等快变量每日计算；旧因子默认走这个路径。
- `FinancialEventSnapshot`：纯财报慢因子。只有当当前交易日区间内出现 `ann_date/f_ann_date` 财务事件时，重算完整截面、截面标准化、回归和中性化；非事件日仍逐日输出，但回放最近一次事件日的最终因子截面。
- `FinancialEventStateDailyFast`：快慢混合因子。财务向量、F-Link、peer/network 等慢状态仅在财务事件日更新；Ret20、价格、行业市值中性化等快分支仍每日计算。

- `Daily`: default behavior. Valuation, price, return, market-cap, turnover, and other fast variables are computed every day.
- `FinancialEventSnapshot`: pure slow financial factors. On event dates, recompute the full cross-section, transforms, regressions, and neutralization. On non-event dates, still output daily rows by replaying the most recent final factor cross-section.
- `FinancialEventStateDailyFast`: mixed slow/fast factors. Slow financial vectors, F-Link, peer sets, or network state update only on financial events; fast branches such as Ret20, prices, and neutralization still run daily.

`trade_date` 仍由 `TradingCalendar` 控制。财务事件按 PIT 口径 `f_ann_date.or(ann_date) <= trade_date` 生效；非交易日公告会映射到后续第一个目标交易日，因为 schedule 检查的是 `(last_processed_trade_date, current_trade_date]` 区间内是否有事件。

`trade_date` remains driven by `TradingCalendar`. Financial events follow the PIT rule `f_ann_date.or(ann_date) <= trade_date`; non-trading-day disclosures map to the next target trading day because the schedule checks events in `(last_processed_trade_date, current_trade_date]`.

开发新财务因子时 / When adding a new financial factor:

1. 纯财报慢因子使用 `FinancialEventSnapshot` 和 `EventDrivenCrossSectionCache`。
2. 快慢混合因子使用 `FinancialEventStateDailyFast`，只缓存慢状态，快变量每日重新计算。
3. 估值指标、价格收益指标和其他日频快变量保持 `Daily`。
4. 财务科目查表和股票级慢指标公式必须放入 provider state，例如 `InstrumentAlignedSnapshotCache<T>` + `cached_financial_stock_snapshots_for_date(...)`；不要放在单次函数调用的局部 cache 中。
5. Barra CNE6 的 `growth / quality / value / dividend_yield` 财报或分红 slow legs 使用同一规则；分析师 legs 仍按日频快分支处理。
6. 多输出 provider 必须继续 requested-aware；同 provider 的 wrapper 共享 `compute_provider_key()`，并使用同一种 state 类型。

1. Use `FinancialEventSnapshot`, `EventDrivenCrossSectionCache`, and `compute_financial_event_snapshot_streaming_on_panel(...)` for pure slow statement factors.
2. Use `FinancialEventStateDailyFast` for mixed factors: cache only slow state, recompute fast variables daily.
3. Keep valuation, price/return, and other daily fast variables on `Daily`.
4. Put financial statement lookups and stock-level slow formulas in provider state, for example `InstrumentAlignedSnapshotCache<T>` plus `cached_financial_stock_snapshots_for_date(...)`; do not keep them in one-shot local function caches.
5. Use `data.stock_universe_panel()?` for financial/event output grids; there is no PV-anchor replay compatibility API.
6. Barra CNE6 `growth / quality / value / dividend_yield` statement or dividend slow legs follow the same PIT/cache rule; analyst legs remain daily fast branches.
7. Multi-output providers must remain requested-aware; wrappers sharing a provider key must use the same state type.

股票级财报慢指标应优先使用 provider-state 的 `InstrumentAlignedSnapshotCache<T>` + `cached_financial_stock_snapshots_for_date(...)`。`skip_fn` 负责剔除 `.BJ`、非在市或不在股票池的股票；`marker_fn` 声明会影响 snapshot 的 PIT 记录链和 synthetic marker；`compute_fn` 只写股票级慢指标公式。截面 rank、OLS/ridge、网络降维和中性化仍放在事件日或每日的后续步骤中。

For stock-level slow financial metrics, prefer provider-state
`InstrumentAlignedSnapshotCache<T>` plus
`cached_financial_stock_snapshots_for_date(...)`. `skip_fn` handles `.BJ`,
non-present, or out-of-universe stocks; `marker_fn` declares PIT record chains
and synthetic markers that can change the snapshot; `compute_fn` contains only
the stock-level slow formula. Cross-sectional rank, OLS/ridge, network
reductions, and neutralization stay in the event-date or daily post-processing
stage.

完整开发范式见 / Full development pattern: [FINANCIAL_FACTOR_DEVELOPMENT_README.md](FINANCIAL_FACTOR_DEVELOPMENT_README.md).

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
- Financial factor development: [FINANCIAL_FACTOR_DEVELOPMENT_README.md](FINANCIAL_FACTOR_DEVELOPMENT_README.md)
- Short factor development navigation: [docs/FACTOR_DEVELOPMENT.md](docs/FACTOR_DEVELOPMENT.md)
- Strategy development: [STRATEGY_README.md](STRATEGY_README.md)

常用验证命令 / Common validation:

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo check --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
```
