# Factor Engine / Rust 计算引擎

本模块负责股票和期货的因子生产、收益标签、Barra CNE6 暴露、派生数据、截面回测及事件驱动策略。它不负责下载原始数据或训练 Python 模型。

This module produces stock and futures factors, return labels, Barra CNE6 exposures and derived datasets, and runs cross-sectional backtests and event-driven strategies. Raw-data downloading and Python model training are separate modules.

## 准备与入口 / Setup and entrypoints

安装 Rust stable，按[项目说明](../README.md)配置数据目录。以下命令从仓库根目录运行。具体参数以 `help` 和当前配置为准。

Install Rust stable and configure the data root using the [project guide](../README.md). Run commands from the repository root. Consult `help` and the current configuration for supported options.

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- help
cargo run --release --manifest-path factor_engine/Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine/Cargo.toml -- list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine/Cargo.toml -- plan --asset stock --frequency daily --start-date 20260420 --end-date 20260424 --factors ret20
python scripts/yq.py run --asset stock --frequency daily --start-date 20260420 --end-date 20260424 --factors ret20 --date-batch-size 120 --profile
```

使用 `--factors a,b` 或 `--tags tag` 选择因子，不能同时指定。实现或注册发生变化后刷新 metadata。模型生成因子由 Python 推理入口生产，不通过 Rust `run` 训练。

Select factors with either `--factors a,b` or `--tags tag`, not both. Refresh metadata after implementation or registry changes. Model-generated factors are produced by Python inference, not trained by Rust `run`.

## 派生数据 / Derived datasets

统一入口生产前自动刷新一次 metadata；底层生产直接查代码 registry，不依赖 metadata 文件。分片、远程 IO、Snappy 和 `--single-task` 规则见[计算与存储](docs/ARTIFACT_STORAGE.md)。原 Cargo 命令仍可使用。
The unified entrypoint refreshes metadata once; production selects code registries without a metadata-file gate. See [execution and storage](docs/ARTIFACT_STORAGE.md) for slicing, remote IO, Snappy and single-task behavior. Cargo commands remain available.

支持分钟聚合 bar、一致预期日频宽表，以及成交量 logsignature 特征。
Supported datasets include aggregated minute bars, daily consensus wide tables and volume logsignature features.

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-bar --asset stock --source minute --bar-size 5 --start-date 20260420 --end-date 20260424
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-consensus --start-date 20260420 --end-date 20260424 --date-batch-size 20
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-logsig --asset stock --start-date 20260420 --end-date 20260424 --threads 2
```

标准 5m 因子可优先读取派生 bar，缺失时按日懒加载 1m fallback；锚点、交错及 rolling 口径不能直接替换。一致预期目前使用可替换的同权聚合，不声称复现未公开的机构权重。Logsignature 固定为 5m/20 日/order10、226 维；需要完整窗口，当前 float32 产物不宜与旧精度产物混用。

Standard 5m factors can prefer derived bars and lazily load a daily 1m fallback; anchored, staggered and rolling definitions are not interchangeable. Consensus uses replaceable equal weighting rather than undisclosed institution weights. Logsignature uses a fixed 5m/20-day/order-10 definition with 226 features and complete windows; do not mix current float32 and older precision datasets.

## 标签与风险暴露 / Labels and risk exposures

先生成回测或中性化所依赖的数据；标签需要未来行情，数据末尾可能无法生成有效值。
Generate dependencies needed by backtesting or neutralization first. Return labels require future prices and may be missing near the data boundary.

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- label-metadata
cargo run --release --manifest-path factor_engine/Cargo.toml -- label-run --asset stock --frequency daily --start-date 20260420 --end-date 20260424
cargo run --release --manifest-path factor_engine/Cargo.toml -- barra-metadata
cargo run --release --manifest-path factor_engine/Cargo.toml -- barra-run --asset stock --frequency daily --start-date 20260420 --end-date 20260424 --families SIZE --profile
```

## 回测与策略 / Backtests and strategies

截面回测评估因子分组收益与 IC；策略模拟使用独立账户、成交和持仓规则。两者不是同一种收益口径。

Cross-sectional backtests evaluate factor portfolios and IC. Strategy simulation applies account, execution and holding rules; its returns have different semantics.

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- backtest --asset stock --frequency daily --start-date 20260401 --end-date 20260424 --factors ret20 --groups 10 --rebalance 5
cargo run --release --manifest-path factor_engine/Cargo.toml -- strategy-run --config strategy_config/stock/strategy_001.toml
```

回测可配置 `--factor-root`、`--label`、`--universe`、`--benchmark` 和 `--neutralize`。先确认所选标签、股票池、指数和风险暴露均已就绪；策略字段见[策略配置](../strategy_config/README.md)。

Backtests support configurable factor roots, labels, universes, benchmarks and neutralization. Verify that required labels, constituents, indexes and exposures exist. See [strategy configuration](../strategy_config/README.md) for simulation settings.

## 输出 / Outputs

```text
data/factors/{asset}/{frequency}/{year}/{date}.parquet
data/label/{asset}/{frequency}/{year}/{date}.parquet
data/barra/{asset}/daily/CNE6/{year}/{date}.parquet
data/derived/stock/bar/{bar_size}m/{year}/{date}.parquet
data/derived/stock/consensus/{date}.parquet
data/derived/stock/logsig_v/{year}/{date}.parquet
data/backtest/stock/daily/{factor_id}/
data/strategy/{asset}/{strategy_id}/
```

路径中的 `data` 代表配置的数据根目录。正式因子宽表按键合并，保留未更新列；本次显式空值会覆盖旧值。派生数据及策略文件各有写出规则，不能套用同一合并假设。

Here `data` denotes the configured data root. Formal factor wide tables merge by key, preserve untouched columns and replace explicit null updates. Derived and strategy outputs have their own write policies.

## 性能与数据边界 / Performance and data boundaries

- `--threads` 控制线程数，`--date-batch-size` 和 `--factor-batch-size` 控制批量；用 `--profile` 观察加载、计算和写出时间，不把 batch 大小理解为内存硬上限。
  Threads and batch sizes tune execution; use profiling to distinguish loading, computation and writing. Batch size is not a hard memory cap.
- 财报按公告日 PIT 查询，跨 batch 有界复用年度索引与股票状态；快变量及后处理遵循各因子定义，不统一按 tag 改变频率。
  Financial readers enforce announcement-date visibility and reuse bounded indexes/state. Fast inputs and postprocessing retain factor-specific schedules.
- 标准入口默认本地 IO。可选产物后端不是通用远程文件系统，详见[存储扩展](docs/ARTIFACT_STORAGE.md)。
  Standard entrypoints use local IO. Optional artifact backends are not universal remote filesystems; see [storage extensions](docs/ARTIFACT_STORAGE.md).

## 开发文档 / Development guides

- [因子接口与 provider / Factor interfaces and providers](docs/FACTOR_DEVELOPMENT_README.md)
- [财务因子与 PIT / Financial factors and PIT](docs/FINANCIAL_FACTOR_DEVELOPMENT_README.md)
- [基金持仓因子 / Fund-holdings factors](docs/FUND_HOLDINGS_FACTOR_DEVELOPMENT_README.md)
- [一致预期指标 / Consensus indicators](docs/CONSENSUS_DERIVED_README.md)
- [策略开发 / Strategy development](docs/STRATEGY_README.md)

```bash
cargo check --manifest-path factor_engine/Cargo.toml
cargo test --manifest-path factor_engine/Cargo.toml
```

验证脚本与生成数据保留在忽略目录，不提交。Only source/documentation changes belong in commits; keep validation artifacts untracked.
