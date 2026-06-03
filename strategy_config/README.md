# Strategy Config README / 策略配置说明

`strategy_config/` 存放 `strategy-run` 使用的 TOML 配置，按 asset class 分目录。

`strategy_config/` stores TOML files for `strategy-run`, grouped by asset class.

```text
strategy_config/
  stock/
    strategy_001.toml
  future/
    ag_sma_20.toml
    cta_001.toml
```

## 运行 / Run

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\strategy_001.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml --detail true
```

CLI `--detail true|false` 会覆盖 TOML `[output].detail`。

CLI `--detail true|false` overrides `[output].detail`.

## 通用字段 / Common Fields

```toml
asset_class = "stock"                  # stock | future | multi_asset
strategy_id = "strategy_001"            # output folder name
strategy_class = "stock::strategy_001"

start_date = 20260105
end_date = 20260424
initial_cash = 100000.0

[clock]
bar_frequency = "daily"                 # daily | minute

[execution]
fill_price = "next_open"
buy_commission_bps = 3.0
sell_commission_bps = 5.0
short_commission_bps = 0.0
cover_commission_bps = 0.0
stamp_tax_bps = 0.0
slippage_bps = 0.0
lot_size = 100
```

`fill_price = "next_open"` 表示普通 `on_bar` 订单在下一根 bar 的 open 撮合。

`fill_price = "next_open"` means normal `on_bar` orders fill at the next bar open.

## 股票 Top20 示例 / Stock Top20 Example

```toml
asset_class = "stock"
strategy_id = "strategy_001"
strategy_class = "stock::strategy_001"

start_date = 20140401
end_date = 20260424
initial_cash = 100000.0

[clock]
bar_frequency = "daily"

[execution]
fill_price = "next_open"
buy_commission_bps = 3.0
sell_commission_bps = 5.0
stamp_tax_bps = 0.0
slippage_bps = 0.0
lot_size = 100

[paths]
model_root = "data/models"

[strategy]
signal_id = "mdl_000006"
signal_root = "data/models"
rebalance_days = 5
top_n = 20
cash_buffer = 0.0
```

`[strategy]` 是策略私有参数。引擎只负责解析为字符串 map，具体含义由策略 `.rs` 自己决定。

`[strategy]` contains strategy-owned params. The engine passes them as strings; each strategy interprets them.

## 期货 SMA 示例 / Future SMA Example

```toml
asset_class = "future"
strategy_id = "ag_sma_20"
strategy_class = "future::sma"

start_date = 20260105
end_date = 20260424
initial_cash = 8000000.0

[clock]
bar_frequency = "minute"

[market]
products = "AG"

[output]
detail = false

[execution]
fill_price = "next_open"
buy_commission_bps = 0.0
sell_commission_bps = 0.0
short_commission_bps = 0.0
cover_commission_bps = 0.0
stamp_tax_bps = 0.0
slippage_bps = 0.0
lot_size = 1

[future]
default_margin_ratio = 0.12
max_margin_ratio = 0.30
mark_price = "close"

[future.margin_by_product]
AG = 0.12

[strategy]
product = "AG"
window = 20
capital_per_contract = 500000
add_quantity = 1
```

`[market].products` 会先过滤行情时间轴，避免其他品种的分钟 bar 污染策略时间。多品种可写 `"AG,IF,IC"`。

`[market].products` filters the market clock first. Multiple products can be written as `"AG,IF,IC"`.

## 输出 / Output

```text
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

分钟策略默认 `detail=false`，输出每日一行日终 snapshot；`detail=true` 输出每分钟一行。

Minute strategies default to daily snapshots; `detail=true` writes one row per minute.
