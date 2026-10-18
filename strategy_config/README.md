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

### strategy_001 固定持有期 / Fixed holding batches

T 日收盘信号在 T+1 开盘买入，T+6 开盘卖出，持有 5 个交易日。首个交易日使用上一交易日信号，后续每 `rebalance_days` 个交易日先平旧批次、再买新批次；重复入选也卖出重买并计费。不能卖出的旧持仓保留，下一调仓再次尝试。回测结束时未到期的持仓按末日收盘估值，不强制提前平仓。

严格按信号选择 Top N，不因缺行情/涨停改买后续排名。仓位由策略按开盘价、100 股整手和费用计算，每只约 1/N 预算；未成交份额和整手余款留现金，不重新分配，不借款。`cash_buffer=0` 仅取消主动现金预留，不保证成交后满仓。冻结持仓占用资金时缩减新批次预算，不融资。

股票账户自动读取 `chn_stock_data/dividend/*.parquet` 的已实施方案：登记日收盘锁定权益，除权日记送转股和税前现金股利，派息日前记应收款，送转股上市日前不可卖。真实未复权价格用于成交与估值；不再把除权降价单独当作亏损。`dividend_receivable` 纳入净资产但不可交易，`dividend_income_cum` 为累计税前现金股利。使用 `cash_div_tax`，不推算个人持有期差异化红利税；零碎送转股向下取整。缺派息/上市日期时警告并使用除权日。该模块处理分红送转，不模拟配股认购、合并换股等其他公司行动。

本轮独立参数配置（初始资金均为 10 万、信号为 `mdl_000009`）：

```powershell
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/stock/strategy_001_top10_5d.toml
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/stock/strategy_001_top20_5d.toml
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/stock/strategy_001_top50_5d.toml
```

输出目录使用各自的 `strategy_id`，不会相互覆盖。

```text
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

分钟策略默认 `detail=false`，输出每日一行日终 snapshot；`detail=true` 输出每分钟一行。

Minute strategies default to daily snapshots; `detail=true` writes one row per minute.

## 性能与执行约定 / Performance and Execution Semantics

- `detail=false` 的分钟策略仍逐 bar 更新持仓估值、执行策略和撮合，只在当日最后一根 bar 构造日终快照；四类成交计数累计全天。`detail=true` 仍保存每根 bar 的浮盈浮亏和持仓明细。
- 持仓输出达到 256 行或约 8 MiB 快照内容时，分批写入临时 Parquet 文件，完成后替换正式文件。单行可能超过阈值；阈值约束快照缓冲，不是进程总内存上限。
- 期货日线优先使用日文件；缺失时缓存当前年度文件并建立日期索引，切换年度缓存时释放旧数据。同一次回测期间年度缓存不感知文件更新。
- 期货分钟品种/合约过滤在 Parquet 解码阶段执行；浮点行情列可直接借用时不再复制。过滤仍保留 products 与 symbols 的并集语义。
- 收盘涨跌停标志用于全日成交过滤，是有意采用的悲观假设。缺行情、涨跌停导致不成交时不保留订单；资金不足产生的未成交余量也不会自动重试，需策略重新下单。
- 数据降级与部分无效配置会打印 `[strategy warning]` 到 stderr，保留原有默认值和错误处理方式。公共层未新增风控规则。

Minute valuation remains per bar in both output modes. Daily output only serializes the final snapshot and aggregates daily fill counts. Holdings are streamed to a temporary Parquet file and published on successful completion. Annual fallback data is cached per run; minute symbol filters are pushed into the Parquet reader. Existing pessimistic execution and strategy-owned risk rules are unchanged.

### 当前研究：Top50 与三个对照

默认 `strategy_001.toml` 使用 Top50。四组独立配置：

| 配置 | 信号排名范围 | 涨跌停限制 |
|---|---|---|
| `strategy_001_top50_5d.toml` | 1–50 | 保留 |
| `strategy_001_top50_exclude10_5d.toml` | 11–50 | 保留 |
| `strategy_001_top50_exclude20_5d.toml` | 21–50 | 保留 |
| `strategy_001_top50_no_limits_5d.toml` | 1–50 | 关闭，仅用于假设可成交的测试 |

`[strategy].exclude_top_n` 默认 0，必须小于 `top_n`。先取原 Top N，再剔除头部，不向后补位；余额仍遵循约等权整手分配。`[execution].respect_price_limits` 默认 true，设置 false 同时关闭策略分配和成交引擎中的涨停买入/跌停卖出限制；资金、整手、未上市送转股锁定、费用与行情缺失处理保持原样。

初始资金 10 万元，信号 `mdl_000009`，持有 5 个交易日；NAV/PnL 都包含分红送转。研究结果位于本地 `strategy.ipynb`。
