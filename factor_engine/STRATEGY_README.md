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

`holdings.parquet` 每行是一个结算 bar。日频策略每天一行；分钟策略默认 `detail=false`，每个交易日一行日终 snapshot；`detail=true` 时每分钟一行。

`holdings.parquet` has one row per settlement bar. Daily strategies write one row per day. Minute strategies default to one end-of-day snapshot per trading day; `detail=true` writes one row per minute.

## 事件循环 / Event Loop

固定时序 / Fixed timing:

```text
load current bar
-> fill pending orders at current bar open
-> update account and mark to market at current bar close
-> call strategy.on_bar()
-> queue new orders for next bar open
-> write holding snapshot
```

`on_session_open()` 用于每天第一根 bar 前的信号，适合集合竞价或开盘前逻辑；该事件里的订单可在当天第一根 bar open 成交。普通 `on_bar()` 的订单默认下一根 bar open 成交。

`on_session_open()` is for pre-open signals and can fill at the first bar open. Normal `on_bar()` orders fill at the next bar open by default.

## Strategy Trait / 策略接口

```rust
pub trait Strategy {
    fn name(&self) -> &'static str;
    fn on_start(&mut self, ctx: &mut StrategyContext) -> Result<()> { Ok(()) }
    fn on_session_open(&mut self, ctx: &mut StrategyContext, event: &SessionOpenEvent) -> Result<()> { Ok(()) }
    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()>;
    fn on_fill(&mut self, ctx: &mut StrategyContext, event: &FillEvent) -> Result<()> { Ok(()) }
    fn on_end(&mut self, ctx: &mut StrategyContext) -> Result<()> { Ok(()) }
}
```

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

## 当前策略 / Built-In Strategies

### 股票 Top Signal Equal Weight

配置 / Config:

```text
strategy_config/stock/strategy_001.toml
```

逻辑 / Logic:

- 读取 `data/models/{year}/{date}.parquet` 中的 `signal_id`，例如 `mdl_000006`。
- 每 `rebalance_days` 个交易日调仓一次。
- 剔除 `.BJ`，选择因子值最高的 `top_n` 只股票。
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

2. 实现 `Strategy` trait，并提供 `from_context_config()` 读取 `[strategy]` 参数。

   Implement `Strategy` and provide a constructor that reads `[strategy]` params.

3. 在对应 `mod.rs` 中注册 `strategy_class` 字符串，例如 `stock::my_strategy`。

   Register the `strategy_class` in the asset class `mod.rs`.

4. 在 `strategy_config/{asset}/` 下写 TOML。

   Add a TOML config under `strategy_config/{asset}/`.

5. 运行：

   Run:

   ```powershell
   cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\my_strategy.toml
   ```

## 测试 / Tests

```powershell
cargo fmt --manifest-path factor_engine\Cargo.toml
cargo test --manifest-path factor_engine\Cargo.toml strategy
cargo check --manifest-path factor_engine\Cargo.toml
```

建议测试点 / Suggested tests:

- `on_bar` 订单在下一根 bar open 成交。
- `on_session_open` 订单在当天第一根 bar open 成交。
- 买入、卖出、开空、平空、反手 PnL 手算一致。
- 股票 lot size、现金不足、费用和滑点生效。
- 期货 multiplier、margin ratio、max margin cap 生效。
