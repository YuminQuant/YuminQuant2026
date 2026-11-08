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

## 期货分钟复权 / Futures minute adjustment

```bash
python scripts/yq.py run --asset future --frequency minute_1m --start-date 20260424 --end-date 20260424 --factors yq_001
python scripts/update_incremental.py --groups future_mapping --start-date 20260423 --end-date 20260424
```

- `--future-main-source open-interest` 读取 `future_data/basic/mapping_oi/YYYY.parquet`。先运行 `derive-future-main-oi --start-date YYYYMMDD --end-date YYYYMMDD --date-batch-size 120`，区间需覆盖预热日。首个有持仓数据的交易日按当日最大持仓量初始化，并沿用到下一交易日。此后使用上一交易日的持仓量，只切换到更远月份，且新合约持仓量达到旧主力的 1.1 倍。旧主力缺有效持仓量时从更远月份中重新选择。缺上一交易日数据的映射留空。
- `--future-main-source fut-mapping` 是默认选项，优先读取 `future_data/basic/mapping/YYYY.parquet`，缺少日期时兼容旧版按日文件。下载时应覆盖所选因子的预热日期。输出的 `ts_code` 为目标交易日实际选中的合约代码。
- `--future-main-source all-contracts` 对每个真实合约独立计算。再指定 `--future-adjust-daily false` 可计算未复权的全合约分钟因子。
- `--future-main-only true` 默认仅计算并输出目标日的主力合约。目标日缺少可用映射时回退到全合约；设为 `false` 则所有日期都计算全合约。
- `--future-warmup-source main-sequence` 默认用各历史日的主力为目标日主力预热。设为 `same-contract` 则用目标日实际合约的历史。全合约输出使用各合约自己的历史。历史日没有映射时，主力序列预热使用目标日实际合约在该日的历史。
- `oi_diff` 始终先在实际合约内做差，日首根使用该合约上一交易日最后一根 bar 的原始持仓量。差分得到的序列再按所选预热方式拼接。
- 新期货分钟因子采用有限回望。EMA 使用最近 N 根 bar 的归一化指数权重，ATR 的指数平滑也使用有限窗口。同一日期批次的期货分钟行情在因子批次间共享。
- `--future-adjust-daily` 默认开启，连接上一交易日日盘收市和下一交易日的首个开市段。夜盘结束到同一交易日日盘开市、早盘短休和午休分别由 `--future-adjust-night`、`--future-adjust-morning`、`--future-adjust-noon` 控制，默认关闭。
- 每个目标日以声明的预热范围内最早可用的分钟数据为基准。启用的边界按上一段收盘价除以下一段开盘价，把比例累计乘到后续价格。边界价格缺失或非正时跳过该次比例调整。期货分钟输出按日替换本次计算的因子列。

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

默认 `--exclude-limit true --limit-side both`：用信号日 T 的因子、股票池和 ST 标记确定各组及目标权重。调仓时，T+1 收盘涨跌停只阻止新增仓位；已有同方向仓位最多保留至目标权重，减仓不受此规则限制。未成交权重保留为零收益现金至下次调仓，不补选、不重新归一化。没有有效收益的持仓不把权重分摊给其他股票。默认基准 `mkt_mean` 每日对通过 T 日 ST 和 T+1 涨跌停筛选且收益有效的股票求等权平均；显式指定的加权基准也按 T+1 涨跌停过滤，并按剩余有效权重归一化。`--exclude-limit false` 对组合和基准均关闭涨跌停筛选。末个信号日缺少 T+1 交易过滤表时跳过该日的限价约束；区间内缺表仍报错。这是用收盘状态近似成交约束，不是逐笔成交模拟；不改变策略模拟的执行规则。

With the defaults `--exclude-limit true --limit-side both`, groups and target weights use only the T signal, universe, and ST status. At rebalance, T+1 closing limit status blocks increases in exposure; an existing same-side position can be kept up to its target, and reductions are allowed. Unfilled allocation stays in zero-return cash until the next rebalance, with no replacements or renormalization. Missing returns do not redistribute portfolio weight. The market-mean benchmark uses signal-date ST status and finite returns, not T+1 limits. If the final signal date has no T+1 trade-filter file, its limit constraint is skipped; missing files inside the range remain errors. Closing limits proxy execution constraints, not individual fills; strategy execution is unchanged.

### 期货分钟因子评价

```powershell
cargo run --release --manifest-path factor_engine/Cargo.toml -- backtest --asset future --frequency minute_1m --start-date 20260423 --end-date 20260424 --factors yq_001,yq_002
```

- 因子和标签分别从配置的 `factor_root` 和 `label_root` 读取，可用 `--factor-root` 和 `--label-root` 覆盖。两边均使用根目录下的 `future/minute_1m/{year}/{trade_date}.parquet`，按 `trade_date + trade_time + ts_code` 对齐。每个请求的交易日都需要有对应的两张表。每次运行选择一列标签，默认为 `future_open_return_5m`，可用 `--label` 切换。输出的 Parquet 元数据记录标签列名。使用同一输出目录重跑会替换该因子的摘要。
- 因子选择支持 `--factors`、`--tags`、`--all-factors` 三选一。标签按全部匹配筛选，`--tags pv,yq` 选择这批 YQ 期货分钟因子。标签读取当前期货分钟因子注册表。与股票入口一致，指定外部 `--factor-root` 时用 `--factors` 或 `--all-factors`。
- 支持 `--factor-batch-size`、`--date-batch-size`、`--threads` 和 `--profile`。日历在启动时读取一次，五日窗口跨日期批次延续。股票专用选项如 `--benchmark`、`--barra`、`--neutralize` 和 `--detail` 可传入，在期货分钟回测中不参与计算。
- `--products` 可传 `RB` 或 `RB.SHF`，筛选同时作用于 All、板块和品种评价。未指定时使用全部品种。品种由实际交割合约代码归并，如 `RB2610.SHF` 归入 `RB.SHF`。同一品种若有多个期限，品种评价合并这些合约的分钟样本。
- 默认模式 `--quantile-mode trailing-5` 使用每个品种、因子在 T−5 至 T−1 的每日上下分位数的等权均值，写入 `data/backtest/future/minute_1m/{factor_name}/summary.parquet`。`--quantile-mode daily` 使用当日分位数，写入 `data/backtest/future/minute_1m/daily/{factor_name}/summary.parquet`。默认双端为 P05/P95，`--quantile 90` 可切换到 P10/P90。列顺序均为 `trade_date, ts_code, ic, cosic, long_rtn, short_rtn, trigger_count, eligible_contract_count`，模式和分位参数记录在 Parquet 元数据中。`--output-dir` 可覆盖因子目录的上一级。
- 每个交易日为每个品种输出一行，`ts_code` 存品种代码。`ic` 和 `cosic` 分别为 Pearson IC 和余弦相似度，使用该品种当日的有效分钟配对。`long_rtn` 和 `short_rtn` 分别为因子值达到上阈值及以上、下阈值及以下的有效标签均值，沿用标签的单位和符号。分位点由当日所有有限因子值线性插值得到。两条阈值重合且当日有有效因子与标签配对时，上下尾收益均记为 0。没有有效配对时，收益留空。`trigger_count` 记录上下端合计的有效合约分钟触发数，All 行为品种合计。有阈值和有效配对但没有触发时记 0，缺少阈值或有效配对时留空。
- 每个交易日还有一行 `ts_code=all`。将全部合约当日的分钟样本拼接，直接对有效因子值和标签计算一次 `ic` 和 `cosic`。两端收益分别在当日有对应有效收益的品种间等权平均。余弦相似度至少需要两个有效配对，任一向量全为零时为空。
- 板块行用 `ts_code=sector:<板块缩写>` 标识，按同一口径在板块内跨合约计算 IC 和 CosIC。板块行的两端收益列留空，报告侧使用各品种定向后的收益汇总板块收益。
- `eligible_contract_count` 记录当日有可用阈值且至少有一个有效因子标签配对的合约数。品种和 All 行记录此分母，板块报告按品种汇总。
- 五日模式只用因子预热，最多读取回测起点之前五个交易日。前五个交易日有任一天缺少该品种的有效分位数时，当日两端收益留空。阈值可用且当日有有效配对时，某一端没有达到该端阈值的有效配对，则该端收益记为 0。IC 和 CosIC 的计算口径相同。
- `--quantile 50` 使用中位数作为分界。因子值严格高于或低于中位数的样本分别进入上端和下端，等于中位数的样本不归入两端。
- 板块分类、品种中英文名称和交易所代码见[期货板块](docs/FUTURE_SECTORS.md)。

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


### Minute stock strategy

`stock::strategy_001` reads the preceding session's signal and rebalances at
the configured interval (default five sessions). The sample configuration is
`strategy_config/stock/strategy_001.toml`. Net mode orders target minus actual
shares. Round-trip mode exits and reopens each batch. Whole lots, fees and the
opening equity cash reserve constrain buys. `allocation_mode = "rank_carry"`
optionally passes unused budget to lower ranks in net mode.

The default TWAP window is 09:30–09:59. At each minute open, the dated price
bands and daily `pre_close` identify the 10% or 20% limit regime. When the open
reaches either ±9% or ±18% respectively, a pending buy requests its full
remaining quantity. A pending sell accelerates only on the positive side.
Other limit regimes and unavailable reference prices use normal scheduling.
Acceleration still respects limits, volume, sellable shares, fees, lots and
available cash. All remaining quantities are due on every eligible bar after
the window. With `retry_next_day = true`, any remainder carries to the next
session; new targets replace pending orders at the next rebalance.

Minute replay runs through the close with either output detail setting.
Signals, projected minute bars, bands and previous closes use bounded
read-ahead and account events consume them in calendar order. A single account
should cover the full date range so cash and rebalance phase persist across years.

Runs save `run_config.toml`, `fills.csv`, and in net mode `rebalance.csv`.
The generic report entry point is `analysis_tools/scripts/report_strategy.py`.
Research configurations and outputs belong in the ignored
`research/strategy001/` directory.

Stock runs enable `[execution] force_delisting_exit = true` by default. The event
uses `delist_date` from `chn_stock_data/info/stock_basic.parquet` (the first
available exchange session on or after that date if the date is closed). That
symbol is excluded from regular orders starting that session. At
`delisting_exit_time = "15:00"` by default, all remaining shares are assumed
sold at the latest valid account mark, including at limit-down. Minute runs can
configure the cutoff as `HH:MM`. Daily runs use the close. When the cutoff has no
quote, the account's last mark supplies the price without retaining a price
history. These simulated sales use normal selling costs, credit cash, and
appear in the regular fill accounting and `lifecycle.csv`. Set
`force_delisting_exit = false` to disable the rule. An absent stock metadata file
emits a warning. A missing quote alone does not trigger an exit.
