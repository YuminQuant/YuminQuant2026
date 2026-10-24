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

## 计算资源与共享输入 / Compute resources and shared inputs

因子、Barra 和标签生产通过 `--threads N` 指定计算线程数；未指定时，默认使用当前进程可用 CPU 配额的一半，至少为 1。显式线程数不得超过进程可用 CPU 或部署层声明的配额。线程数是上限，不保证所有阶段都能占满 CPU；IO 并发独立控制，不随计算线程数增大。

Factor, Barra and label production accept `--threads N`. The default is half the effective process CPU allocation, with a minimum of one. Explicit values cannot exceed the process/deployment allocation. This is a concurrency ceiling, not a promise of full CPU utilization. IO concurrency is independent of compute threads.

```bash
python scripts/yq.py run --asset stock --frequency daily --start-date 20260413 --end-date 20260424 --factors ret20 --threads 8 --date-batch-size 120 --profile
python scripts/yq.py barra-run --asset stock --frequency daily --start-date 20260413 --end-date 20260424 --threads 8 --exposure-batch-size 2 --profile
```

部署适配器可以申请 30 个 CPU、向引擎传入 `--threads 28`。计算批次和纯函数算子共用同一个 Rayon pool；股票内部时序、财务事件状态机仍按时间顺序推进。已有 `ts/cs` 的 `FnMut` 接口保持串行兼容；可并行的纯函数使用 `ts_parallel`、`ts_binary_parallel`、`cs_parallel`、`cs_binary_parallel` 或 `cs_by_group_parallel`。Barra 的主要时序及标准化步骤已接入这些接口。分钟 bar 的 `date-batch-size` 只控制日期批量，不再决定线程数。

A deployment adapter may reserve 30 CPUs and pass `--threads 28`. Providers and pure operators share one Rayon pool; within-stock chronology and financial event state machines remain sequential. Existing `ts/cs` methods retain their serial `FnMut` contract. Pure kernels opt into the parallel variants listed above, already used by the main Barra time-series and standardization paths. Minute-bar `date-batch-size` controls date batching, not thread count.

普通因子和 Barra 在每个 date batch 内共享已加载的日频输入，按请求合并物理读取；每个计算批次仍只看到自身声明的日期与列，稀疏请求不会混入其他请求的时点。普通因子只提前加载会被多个计算批次复用的依赖，单次依赖仍按需加载；单个 provider 不增加共享层。不再重复构建相同的行情表；panel-only 请求共享 panel，需要行表时仅保留所需日期/列，或按计算批次构建所需 panel，避免全历史行表与等价 panel 同时常驻。财务共享 PIT、report type、预热窗口及跨 batch 慢状态口径不变。

Factors and Barra share daily inputs within each date batch, merging physical reads while retaining request-specific dates and columns. Sparse requests do not inherit other requests' dates. Ordinary factors preload only dependencies reused across compute batches; single-use inputs remain lazy and single-provider runs bypass sharing. Panel-only requests share panels; row tables retain only required dates/columns, or panels are built per compute batch, avoiding equivalent full-history representations. Shared financial PIT indexes, report preferences, warmup windows and cross-batch slow states retain their existing semantics.

Parquet 随机读使用文件生命周期内的有界内存块缓冲，复用页头/页体重叠读取；不合并小文件，不增加节点磁盘数据缓存。输出仅累计目标日的列式数据，默认达到约 256 MiB 后分批发布；未达到阈值时，同日多批因子/暴露只合并写出一次。锁、Snappy、原子发布及保留已有其他列的规则不变。`YQ_OUTPUT_BUFFER_BYTES` 可调整发布缓冲阈值；它不是整个进程的内存限制。`--profile` 的读取字节/文件打开/区间读取/缓冲命中为客户端计数，不等于 DataNode 的全局 RPC 数。

Parquet random reads use a bounded, per-file in-memory block buffer shared by page-header and page-body reads. Small files remain unchanged; no node disk data cache is introduced. Target-date columns are accumulated and flushed at an approximate 256 MiB threshold; batches below that threshold merge each day's outputs once. Locking, Snappy, atomic publication and preservation of unrelated columns remain unchanged. `YQ_OUTPUT_BUFFER_BYTES` changes this flush threshold, not the total process memory limit. Profile bytes, opens, range reads and cache hits are client-side counters, not global DataNode RPC counts.

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
