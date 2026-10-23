# Strategy Configuration / 策略配置

本目录按资产类别保存事件驱动策略的 TOML 配置。策略代码定义交易逻辑，配置指定数据、日期、资金、成交假设和策略参数；配置本身不下载数据或生成模型信号。

This directory stores TOML configurations for event-driven strategies by asset class. Code defines trading logic; configuration selects data, dates, capital, execution assumptions and strategy parameters. It does not download data or generate model signals.

## 常用入口 / Common entrypoints

从仓库根目录运行，先检查配置中的路径、信号和日期。示例所需数据必须已生成。
Run from the repository root after checking paths, signals and dates. Required input data must already exist.

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/stock/strategy_001.toml
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/future/ag_sma_20.toml
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/future/ag_sma_20.toml --detail true
```

股票目录提供不同持仓数量、排名筛选及成交假设的配置；期货目录提供 SMA 和 CTA 示例。选择适合研究的问题，不将示例参数视为推荐投资组合。

Stock examples vary portfolio size, rank filters and execution assumptions; futures examples include SMA and CTA strategies. Select settings for your research question, not as recommended portfolios.

## 配置内容 / Configuration contents

| 字段或分组 / Field or section | 用途 / Purpose |
|---|---|
| `asset_class / strategy_class` | 资产与策略实现 / Asset class and implementation |
| `strategy_id` | 输出目录标识；不同实验使用不同 ID / Output identifier; use separate IDs for experiments |
| `start_date / end_date / initial_cash` | 时间范围与初始资金 / Date range and starting capital |
| `[paths]` | 输入和输出路径 / Input and output paths |
| `[clock]` | 日频或分钟时钟 / Daily or minute clock |
| `[execution]` | 成交价、费用、整手与涨跌停假设 / Fill prices, costs, lot size and price limits |
| `[market] / [future]` | 行情范围与期货保证金设置 / Market scope and futures margin settings |
| `[strategy]` | 策略自身解释的参数 / Strategy-specific parameters |
| `[output]` | 输出明细粒度 / Output detail level |

`fill_price = "next_open"` 表示普通 `on_bar` 订单在下一根 bar 开盘撮合；具体回调和订单行为见策略开发文档。关闭涨跌停限制的配置仅用于假设性比较，不代表真实可成交。

With `fill_price = "next_open"`, ordinary `on_bar` orders fill at the next bar open. Consult the development guide for callback and order semantics. Disabling price limits is a hypothetical comparison, not a claim of executable trades.

## 输出与分析 / Outputs and analysis

```text
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

根目录可由配置调整。分钟策略默认输出日终快照；`--detail true` 覆盖配置并输出分钟明细，不改变逐 bar 执行频率。相同策略 ID 的重跑可能替换既有结果。

The root is configurable. Minute strategies default to daily snapshots; `--detail true` overrides configuration to write minute details without changing per-bar execution. Reusing an ID may replace earlier results.

分析时使用与回测相同的初始资金。资金不足、整手、缺行情及成交限制会影响实际持仓，不保证信号选中的股票全部成交。

Use the same starting capital when analyzing results. Cash constraints, lot sizes, missing prices and execution limits can prevent selected signals from becoming positions.

- [策略开发与执行口径 / Strategy development and execution](../factor_engine/docs/STRATEGY_README.md)
- [报告与绘图 / Reporting and plotting](../analysis_tools/README.md)
- [引擎命令 / Engine commands](../factor_engine/README.md)
