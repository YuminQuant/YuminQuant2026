# Strategy README / 事件驱动策略开发教程

Strategy 模块用于真实策略模拟。它和 `backtest` 不同：`backtest` 是不扣费的快速因子检验，输出组合收益和 IC；`strategy-run` 是事件驱动账户模拟，策略主动下订单，引擎负责撮合、费用、持仓、保证金和 PnL。

The strategy module is for concrete trading simulations. It differs from `backtest`: `backtest` is a fast factor test with returns and IC; `strategy-run` is an event-driven account simulator where strategies place orders and the engine handles execution, costs, positions, margin, and PnL.

## 快速运行 / Quick Runs

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\strategy_001.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml --detail true
```

输出 / Output:

```text
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

`holdings.parquet` 按交易日写账户日结。`detail=true` 额外写出 `trades.parquet`，每笔实际成交一行，股票和期货共用结构。

`holdings.parquet` holds one end-of-day snapshot per trading session. `detail=true` enables a separate `trades.parquet` with one row per fill across stocks and futures.

期货配置 `[output].product_pnl = true` 时还会输出 `product_pnl.parquet`，按交易日和品种记录毛盈亏、手续费、滑点及净盈亏，并校验品种净盈亏合计与账户权益变化一致。CTA 分析入口为根目录 `cta_strategy.ipynb` 和 `analysis_tools/yq_analysis/cta_report.py`。输出目录同时保存输入配置的副本 `run_config.toml`。

For futures, `product_pnl = true` under `[output]` enables daily product attribution with an account reconciliation check. Use `cta_strategy.ipynb` or `analysis_tools/yq_analysis/cta_report.py` for analysis. The output directory also contains a copy of the input configuration as `run_config.toml`.

## 事件循环 / Event Loop

固定时序 / Fixed timing:

```text
load current bar
-> call on_start() once, at the first available frame
-> fill pending orders at current bar open
-> on the first frame of the session, call on_session_open() and fill its orders
-> call on_bar_open() and fill its orders at current bar open
-> update account and mark to market at current bar close
-> process stock lifecycle exits when due
-> call strategy.on_bar()
-> queue new orders for next bar open
-> call on_session_end() on the last frame of the trading date
-> write fills when detail is enabled and the account snapshot at session end
```

每次撮合后调用 `on_fill()`。`on_session_open()` 在当日首帧的待成交订单处理之后调用，其订单可在首根 bar open 成交。`on_bar_open()` 在每帧开盘阶段调用，其订单在当前 bar open 撮合。普通 `on_bar()` 订单进入队列，在下一帧开盘尝试撮合；若该帧缺少对应合约行情，订单不会自动保留到后续帧。未成交或部分成交的剩余数量需要策略自行重试。

`on_fill()` follows each matching pass. `on_session_open()` runs after pending orders on the first frame; its orders can fill at that frame's open. `on_bar_open()` runs on every frame and submits orders for the current open. Orders from `on_bar()` are attempted on the next frame. Missing bars and unfilled balances are not automatically retried.

开盘回调虽然可以通过 `ctx.market()` 访问整根 bar，但只能检查开盘时已知的信息，不能使用当前 bar 的 high、low、close 或完整分钟成交量来决定下单。完整分钟成交量由撮合层用于成交上限。`on_start()`、`on_fill()`、`on_session_end()` 和 `on_end()` 产生的订单当前不会进入撮合队列；需要下单时在上述支持下单的回调中提交。

Open callbacks expose the full bar through the context, but strategy decisions must use only information available at the open. The matcher uses full-minute volume as a fill cap. Orders created in `on_start()`, `on_fill()`, `on_session_end()` or `on_end()` are currently discarded.

## Strategy Trait / 策略接口

```rust
// Main replay callbacks; see strategy.rs for the complete trait.
pub trait Strategy {
    fn name(&self) -> &'static str;
    fn on_start(&mut self, ctx: &mut StrategyContext) -> Result<()> { Ok(()) }
    fn on_session_open(&mut self, ctx: &mut StrategyContext, event: &SessionOpenEvent) -> Result<()> { Ok(()) }
    fn on_bar_open(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> { Ok(()) }
    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()>;
    fn on_fill(&mut self, ctx: &mut StrategyContext, event: &FillEvent) -> Result<()> { Ok(()) }
    fn on_session_end(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> { Ok(()) }
    fn on_end(&mut self, ctx: &mut StrategyContext) -> Result<()> { Ok(()) }
}
```

准备阶段接口见 [strategy.rs](../src/strategy/strategy.rs)：

| 接口 / Hook | 用途与顺序 / Purpose and order |
|---|---|
| `prepare_run(config, dates)` | 初始化整次运行 / Initialize the run |
| `prepare_future_run(config, calendar, dates)` | 期货初始化，可加载映射和预热交易日 / Initialize futures mappings and warmup dates |
| `select_future_contracts(config, date, contracts)` | 每日期货行情加载前选择合约；引擎额外并入已有持仓合约 / Select contracts before daily IO; the engine also includes held contracts |
| `session_files(config, date, symbols)` | 选定合约后声明分日额外输入 / Declare extra daily inputs after selection |
| `on_session_inputs(date, inputs)` | 输入读取完成后，在回放线程接收数据 / Receive loaded inputs on the replay thread |
| `prepare_session(config, account, date)` | 每日回放前准备状态；股票还在行情加载前用其返回值筛选股票，不能使用当天信号进行这次预选 / Prepare daily state; stocks also use its pre-IO return value for selection without today's signal |
| `input_batch_days()` | 默认 1；仅符合引擎条件的股票分钟运行使用批量输入，不能跨策略决策边界 / Defaults to 1; batching is restricted to eligible stock minute runs and must respect decision boundaries |

`session_files` 返回 `SessionFile`，`on_session_inputs` 接收按名称索引的 `SessionInputs`。也可以在 TOML 中使用 `[session_inputs.<name>]` 声明额外 Parquet 输入，具体示例见 [策略配置说明](../../strategy_config/README.md)。

策略可访问 / A strategy can access:

```rust
ctx.config()                      // TOML config and strategy params
ctx.account()                     // cash, equity, positions, PnL
ctx.market()                      // current bars and prices
ctx.order_quantity(symbol, qty)   // signed quantity order
ctx.order_target_quantity(symbol, target_qty)
ctx.order_value(symbol, value)
ctx.order_target_value(symbol, target_value)
```

`order_quantity` 的正数是买入/做多，负数是卖出/做空。股票默认不做空；期货支持多空和反手。

Positive `order_quantity` buys or goes long; negative values sell or go short. Stock is long-only by default; futures support long, short, and reversal.

期货 CTA 使用显式方向接口，与 SMA 的净持仓接口分开：

```rust
ctx.order_future_side(symbol, OrderSide::Buy, lots, signal_time);   // open long
ctx.order_future_side(symbol, OrderSide::Sell, lots, signal_time);  // close long
ctx.order_future_side(symbol, OrderSide::Short, lots, signal_time); // open short
ctx.order_future_side(symbol, OrderSide::Cover, lots, signal_time); // close short
ctx.account().future_position(symbol, side); // Buy/Sell -> long; Short/Cover -> short
ctx.account().futures_positions();          // both sides, including held old contracts
```

显式方向订单要求当天有可用的 `future_day` 合约参数，由 `[future].daily_costs = true` 加载。多空仓位分别存储、保证金分别计提，不要用 `position_quantity()` 或 `order_target_quantity()` 查询或调整这些仓位。显式方向撮合检查最小手数、当前分钟可用成交量、保证金及配置启用的涨跌停限制；同一次撮合内按提交顺序扣减该合约可用成交量。先平后开时应依次提交平仓、开仓订单。

Explicit-side CTA orders require daily contract specifications (`daily_costs = true`). Long and short positions are stored separately; use `future_position()` rather than net-position helpers. Matching checks lot size, remaining minute volume, margin and configured price limits, processing orders in submission order. Minimum opening quantities use daily API records first and dated fallback rules otherwise; closing minimums are independent. See [quantity sources and effective dates](FUTURE_ORDER_LIMITS.md).

## 当前策略 / Built-In Strategies

### 股票 Top Signal Equal Weight

配置 / Config:

```text
strategy_config/stock/strategy_001.toml
```

逻辑 / Logic:

- 读取 `data/models/{year}/{date}.parquet` 中的 `signal_id`，例如 `mdl_000019`。
- 每 `rebalance_days` 个交易日调仓一次。
- 剔除 `.BJ`，选择因子值最高的 `top_n` 只股票。
- 未建仓的买入席位遇到分钟开盘涨停时，按信号排名递补可买股票。递补后，本轮调仓取消原股票的买单。
- `[strategy].limit_replacement_candidates` 指定最多加载多少只候补股票，默认等于 `top_n`，设为 `0` 关闭递补。
- 按账户权益等权目标市值下单。

It reads a signal column, rebalances every `rebalance_days`, selects top `top_n` non-BJ stocks, and targets equal value weights.

Run:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\strategy_001.toml
```

### 期货 SMA

配置 / Config:

```text
strategy_config/future/ag_sma_20.toml
```

逻辑 / Logic:

- 分钟结算，按 `[market].products = "AG"` 过滤 AG 合约。
- 每个合约独立维护 SMA window。
- `close > SMA` 时目标多头；`close < SMA` 时目标空头。
- 已有同向仓位时按 `add_quantity` 加仓。
- 保证金占用受 `[future].max_margin_ratio` 约束。

It filters AG contracts, maintains one SMA per contract, goes long above SMA, short below SMA, adds to same-direction positions, and respects the configured margin cap.

Run:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml --detail true
```

### 期货模型 CTA / Model CTA

实现为 [cta_strategy_001.rs](../src/strategy/strategies/future/cta_strategy_001.rs) 中的 `CtaStrategy001`，注册名为 `future::cta_strategy_001`，配置为 [mdl_100000.toml](../../strategy_config/future/mdl_100000.toml)。

- 从 `<model_root>/future/minute_1m/YYYY/YYYYMMDD.parquet` 读取 `trade_date, ts_code, trade_time` 和 `[strategy].signal_id` 指定的信号列（示例为 `mdl_100000`）；`model_root` 默认是数据根目录下的 `models`，可由 `[paths].model_root` 覆盖。日期不符或合约时间键重复会报错，缺失文件或非有限信号不产生交易信号。
- 按品种分别计算前五个交易日普通主力信号的日 P1/P3/P97/P99，再分别取五日均值。该品种任一天缺少分位数时不开仓。当天信号只在对应 `on_bar` 使用。
- 严格高于 P99 做多，严格低于 P1 做空；每侧一个最小交易单位，同向不加仓。`on_bar` 保存意图，在该合约下一根可用行情的 `on_bar_open` 提交；反向时先平后开，成交仍受成交量、保证金和涨跌停限制。
- 按合约日盘收盘时间提前五分钟停止新开仓并逐分钟尝试平仓；隔夜余仓在下一交易日优先平仓，保留旧合约行情。夜盘按所属交易日处理。
- 信号列名通过必填的 `[strategy].signal_id` 配置；五日窗口、开平仓分位数和提前五分钟平仓仍固定在实现中。`strategy_class` 选择策略代码，`strategy_id` 标识实验输出，二者都不选择信号列。

The model CTA reads minute predictions, averages five prior sessions' daily main-contract P1/P99 thresholds, and submits one minimum lot per side on strict threshold crossings. It stores close-time signals for the contract's next available open, uses P3/P97 exits without a holding-time timeout, closes before reversing, and retries liquidation in the last five minutes and on the next trading date. Select the signal column with required `[strategy].signal_id`; threshold and timing parameters remain fixed in code.

```toml
strategy_class = "future::cta_strategy_001"
strategy_id = "mdl_100000"

[strategy]
signal_id = "mdl_100000"
```

期货的 spread 与滑点只接受 `[execution].spread_ticks`、`slippage_ticks`；旧的 `slippage_bps` 会报错。每手金额为两者之和乘以最小变动价位和合约乘数。交易所手续费仍取每日结算快照。股票保留 bps 成本配置。

Futures spread and slippage are configured in ticks only; `slippage_bps` is rejected. Exchange fees still come from daily snapshots. Stock bps settings are unchanged.

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\mdl_100000.toml --detail true
```

## CTA 开仓池 / CTA entry pool

### CTA 001 平仓规则

P99/P1 开多/开空，每侧一个当日 min_lots，同向不加仓。
多头信号严格小于 P3 平仓，空头严格大于 P97 平仓；各分位数沿用
前五个交易日逐日分位数的均值。信号触发后下一根可用 bar 执行，
旧方向未平完可开反向仓，未成交平仓逐 bar 重试。

001 不设持仓分钟上限，保留尾盘强平和隔夜继续平仓。
配置为 `strategy_config/future/mdl_100000.toml`，信号由 `[strategy].signal_id`
选择。当前只保留 CTA 001；研究策略 002、003 已删除，不再注册。
Slurm 不设置数组并发上限（`max_parallel = 0`）。

`mdl_100000.toml` 在 `[future]` 下设置 `pool_id = "liquidity_001"`。CTA 从 `future_data/basic/mapping_pool/<pool_id>/YYYY.parquet` 读取当天允许开仓的实际主力合约，并与 `[market]` 品种或合约筛选同时生效。当天缺池映射时不开新仓。已有持仓仍加载其合约的行情和结算参数，可以按原规则平仓，包括换年出池或主力换月后的隔夜强平。P99/P1 继续使用过去五个交易日的普通主力信号。省略 `pool_id` 时，开仓使用原主力映射。

Set `pool_id = "liquidity_001"` under `[future]` in `mdl_100000.toml`. New entries require the exact contract in that day's pool and must pass any `[market]` filter. A missing pool session disables entries. Held contracts remain eligible for standard exits, including overnight liquidation after a pool change or contract roll. P99/P1 still use the ordinary main-contract signals from the previous five sessions. Omitting `pool_id` selects the ordinary main-contract mapping for entries.

## holdings.parquet 字段 / holdings.parquet Columns

核心字段 / Core columns:

```text
strategy_id, asset_class, trade_date, trade_time
cash, account_pnl, realized_pnl_cum, net_realized_pnl_cum, unrealized_pnl
gross_market_value, net_market_value, margin_required, available_margin
position_count, buy_count, sell_count, short_count, cover_count
symbols_json, quantities_json, signed_quantities_json, directions_json
avg_costs_json, market_values_json, unrealized_pnls_json
multipliers_json, margin_ratios_json, margin_values_json
trade_symbols_json, trade_sides_json, trade_quantities_json
trade_signed_quantities_json, trade_prices_json, trade_realized_pnls_json, trade_net_pnls_json
```

JSON 字段为空时写 `"[]"`。`directions_json` 和 `trade_sides_json` 使用数值方向，股票多头为正，期货空头为负。

JSON fields use `"[]"` when empty. Direction fields are numeric to keep parsing simple.

## 新增策略 / Add A Strategy

1. 在对应 asset class 目录新建 `.rs`：

   Create a `.rs` file under the relevant asset class:

   ```text
   factor_engine/src/strategy/strategies/stock/my_strategy.rs
   factor_engine/src/strategy/strategies/future/my_cta.rs
   ```

2. 实现 `Strategy` trait，并提供构造函数读取配置。构造函数名不受 trait 限制：可参考 `CtaStrategy001::from_config()`（返回 `Result<Self>`）或 `SmaStrategy::from_context_config()`。自定义 `[strategy]` 参数从 `config.strategy_params` 读取并校验。

   Implement `Strategy` and a configuration constructor; its name is not prescribed. Read and validate custom parameters from `config.strategy_params`.

3. 在对应资产目录的 `mod.rs` 声明模块，例如在 `strategies/future/mod.rs` 添加 `pub mod my_cta;`。然后在 `engine.rs` 导入策略类型，并在 `create_strategy()` 中注册 `strategy_class` 字符串：

   Declare the module in the asset class `mod.rs`, then import its type and register the class string in `engine.rs::create_strategy()`:

   ```rust
   // engine.rs imports
   use crate::strategy::strategies::future::my_cta::MyCta;

   // create_strategy() match arm; constructor returns Result<Self>
   "future::my_cta" => Ok(Box::new(MyCta::from_config(config)?)),
   ```

4. 在 `strategy_config/{asset}/` 下写 TOML，确保 `strategy_class` 与工厂注册字符串一致，例如 `future::my_cta`。用不同的 `strategy_id` 隔离实验输出。

   Add a TOML config under `strategy_config/{asset}/`.

5. 运行：

   Run:

   ```powershell
   cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\my_strategy.toml
   ```

## 验证 / Validation

仓库仍包含 Rust 测试，`engine.rs` 通过 `#[cfg(test)]` 引用 `strategy/tests.rs`，其他模块也有内嵌测试。现有部分测试使用系统临时目录，运行前需确认临时目录及工具缓存均位于 D: 等允许的磁盘，不能写入 C:。新增验证脚本、测试夹具、基准和生成产物统一放在被忽略的仓库 `tmp/` 下，不提交或推送；`ml_alpha/tests/` 必须保持未跟踪。

Rust tests remain in the repository, including `strategy/tests.rs` referenced by `engine.rs`. Some use the system temporary directory; ensure temporary paths and tool caches are on an allowed drive before running them. Keep new validation scripts, fixtures, benchmarks and generated artifacts under ignored `tmp/`, never commit or push them, and keep `ml_alpha/tests/` untracked.

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo check --release --all-targets --manifest-path factor_engine\Cargo.toml
```

建议验证点 / Suggested checks:

- `on_bar` 订单在下一根 bar open 成交。
- `on_session_open` 订单在当天第一根 bar open 成交。
- 买入、卖出、开空、平空、反手 PnL 手算一致。
- 股票 lot size、现金不足、费用和滑点生效。
- 期货 multiplier、margin ratio、max margin cap 生效。
