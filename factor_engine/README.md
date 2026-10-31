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

任务独立 raw 缓存、清理边界及显式共享准备方式见
[Raw 缓存位置](docs/RAW_CACHE_PIPELINE.md)。未配置时保留原有本地缓存路径。
See [raw cache placement](docs/RAW_CACHE_PIPELINE.md) for task-local caching,
cleanup boundaries and explicit shared preparation. Default local paths are unchanged.

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

## 回测输入读取 / Backtest input reads

回测在本地默认串行读文件；远程存储可通过通用 `ArtifactBackend::read_options()` 配置 IO worker、打开速率和预读预算。通用默认值为 4 个 worker、每秒启动 10 次打开、128 MiB，实际值由适配器配置决定。远程 worker 并发读取所需列的压缩数据，复用同一次打开取得的 footer/schema；解压和 Table 转换受 `--threads` 约束，结果按原日期顺序消费。完成后立即补充新文件，不再等待整波文件。文件状态和来源规划也复用 IO 并发预算；适配器若已限速，通过 `paces_read_opens()` 避免引擎重复限速。本地不启用该流水线，不创建节点磁盘数据缓存。

预读按投影列的压缩大小、未压缩大小和行数估算内存，并在读取数据列之前预留预算；预算覆盖读取中、解压中和等待消费的结果。单个超预算文件独占推进，避免死锁；估算不是进程 RSS 的硬上限。另有最多 `worker × 1 MiB` 的准备期区间缓冲及 footer/元信息开销；回测必需的最终 panel、公共标签和已有共享缓存不计入预读预算。`--profile` 输出预留峰值、超预算文件数和结果超过估算的次数。读取失败、消费失败或可捕获 panic 会停止派发并释放待处理数据。

打开速率可配置为 `0`，表示不额外限制每秒打开次数；文件并发和内存上限仍然生效。固定速率过低会使暖缓存的小文件读取变慢，不能仅凭打开次数推断提速比例。

Backtests read local files serially by default. A remote adapter configures IO workers, open rate and prefetch budget through `ArtifactBackend::read_options()`. Generic defaults are four workers, ten open starts per second and 128 MiB; the adapter selects the effective settings. Remote workers fetch only projected compressed column chunks, reusing the open reader and footer/schema. Decompression and Table conversion respect `--threads`; results are consumed in original date order, with replacement work dispatched as files complete rather than at wave barriers. File status/source planning uses the IO budget too. An adapter that already paces opens declares `paces_read_opens()` to avoid double pacing. No dataset cache is written to node disk.

Memory is reserved before payload reads using projected compressed/uncompressed sizes and row counts, covering fetching, decoding and queued results. A single oversized file runs exclusively to avoid deadlock; estimates are not hard process RSS limits. Preparation also uses up to `workers × 1 MiB` of range buffers plus footer/metadata overhead. Required final panels, common labels and existing shared caches are outside this budget. Profile counters report peak reservations, oversized files and results exceeding estimates. Read/consumer failures and caught panics stop dispatch and release pending data.

An open rate of `0` disables the additional per-second cap, not the file-concurrency or memory limits. An overly low rate can slow warm-cache small-file reads; fewer opens alone do not establish a wall-time speedup.

manual/ML 因子表在同一次打开中完成 schema 获取和数据读取，列归属、按股票和日期键合并、空值及 forward-fill 规则不变。同一远程回测的多个 factor batch 可共享最多约 128 MiB 的 Barra 投影表；按文件长度和修改时间失效，预算不足则正常重读，不积累全历史缓存。`--profile` 新增同时打开句柄峰值及打开延迟 P50/P95 的直方图区间上界，仍是客户端指标，不代表 NameNode/DataNode 的全局请求数。

Manual/ML factor schema and projected data share one open, retaining ownership, keyed merges, nulls and forward-fill semantics. Multiple factor batches in one remote backtest can share approximately 128 MiB of projected Barra tables. Entries are invalidated by file size/mtime; inputs outside the budget are reread normally rather than retained for the full history. Profile output includes peak live read handles and histogram upper bounds for open-latency P50/P95. These remain client-side measurements, not global NameNode/DataNode request counts.

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

标签生产与因子复用相同的 date-batch 共享输入、受控线程池和有界批量写出工具。每个标签仍只看到自身声明的未来交易日窗口，只写目标日；不会把标签未来收益混入因子历史预热。多个标签批次在预算内合并后更新同一日宽表，保留其他列，统一使用 Snappy 和原子发布。本机和远程后端使用相同的计算与写出链路。

Label production reuses the factor engine's date-batch shared inputs, bounded thread pool and buffered output tools. Each label retains its declared future window and writes target dates only; future label returns never enter factor warmup data. Label batches share daily-wide writes within the buffer budget, preserving other columns with Snappy and atomic publication. Local and remote backends use the same execution path.

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

## 手工与 ML 因子的日文件

手工因子写入 `<year>/<YYYYMMDD>.parquet`，ML 因子写入同目录的 `<YYYYMMDD>_ml.parquet`。
公共读取层按列归属选择文件，回测和历史填充兼容两类文件及旧混合文件。
旧列物理迁移使用 `migrate-factor-files --start-date YYYYMMDD --end-date YYYYMMDD [--dry-run]`；
`remove-factor-columns` 会处理两类日文件。详见 [双文件存储协议](../ml_alpha/docs/FACTOR_FILE_PARTITIONS.md)。
