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

因子、Barra 和标签生产通过 `--threads N` 指定计算线程数；未指定时，默认使用当前进程可用 CPU 配额，至少为 1。显式线程数不得超过进程可用 CPU 或部署层声明的配额。线程数是上限，不保证所有阶段都能占满 CPU；IO 并发独立控制，不随计算线程数增大。

Factor, Barra and label production accept `--threads N`. The default is the effective process CPU allocation, with a minimum of one. Explicit values cannot exceed the process/deployment allocation. This is a concurrency ceiling, not a promise of full CPU utilization. IO concurrency is independent of compute threads.

```bash
python scripts/yq.py run --asset stock --frequency daily --start-date 20260413 --end-date 20260424 --factors ret20 --threads 8 --date-batch-size 120 --profile
python scripts/yq.py barra-run --asset stock --frequency daily --start-date 20260413 --end-date 20260424 --threads 8 --exposure-batch-size 2 --profile
```

部署适配器可以申请 30 个 CPU、向引擎传入 `--threads 28`。计算批次和纯函数算子共用同一个 Rayon pool；股票内部时序、财务事件状态机仍按时间顺序推进。已有 `ts/cs` 的 `FnMut` 接口保持串行兼容；可并行的纯函数使用 `ts_parallel`、`ts_binary_parallel`、`cs_parallel`、`cs_binary_parallel` 或 `cs_by_group_parallel`。Barra 的主要时序及标准化步骤已接入这些接口。分钟 bar 的 `date-batch-size` 只控制日期批量，不再决定线程数。

A deployment adapter may reserve 30 CPUs and pass `--threads 28`. Providers and pure operators share one Rayon pool; within-stock chronology and financial event state machines remain sequential. Existing `ts/cs` methods retain their serial `FnMut` contract. Pure kernels opt into the parallel variants listed above, already used by the main Barra time-series and standardization paths. Minute-bar `date-batch-size` controls date batching, not thread count.

普通因子和 Barra 在每个 date batch 内共享已加载的日频输入，按请求合并物理读取；每个计算批次仍只看到自身声明的日期与列，稀疏请求不会混入其他请求的时点。普通因子只提前加载会被多个计算批次复用的依赖，单次依赖仍按需加载；单个 provider 不增加共享层。不再重复构建相同的行情表；panel-only 请求共享 panel，需要行表时仅保留所需日期/列，或按计算批次构建所需 panel，避免全历史行表与等价 panel 同时常驻。财务共享 PIT、report type、预热窗口及跨 batch 慢状态口径不变。

Factors and Barra share daily inputs within each date batch, merging physical reads while retaining request-specific dates and columns. Sparse requests do not inherit other requests' dates. Ordinary factors preload only dependencies reused across compute batches; single-use inputs remain lazy and single-provider runs bypass sharing. Panel-only requests share panels; row tables retain only required dates/columns, or panels are built per compute batch, avoiding equivalent full-history representations. Shared financial PIT indexes, report preferences, warmup windows and cross-batch slow states retain their existing semantics.

Parquet 随机读使用文件生命周期内的内存块缓冲，复用页头和页体的重叠读取。因子、标签和 Barra 每个计算批次完成后立即合并写入当日文件，释放该批结果后进入下一批。远端落盘统一最多 50 个并发写入，使用文件锁、Snappy 压缩和原子发布，保留未更新的已有列。`--profile` 记录客户端读取字节数、打开文件数、区间读取数和缓冲命中数。

Parquet random reads use a per-file block buffer shared by page-header and page-body reads. Factor, label and Barra results are merged into daily files and released after each compute batch. Remote publication allows up to 50 concurrent writes and uses file locking, Snappy compression and atomic replacement while preserving untouched columns. Profile bytes, opens, range reads and cache hits count client activity.

## 期货分钟复权 / Futures minute adjustment

期货时序因子按排序后的交易分钟序列计算。夜盘到日盘、早盘短休、午休和跨交易日均连接到上一根可用 bar，预热数据参与该序列。公共连续标记仅在组装后序列的第一根 bar 为 0，其后为 1。原始字段缺失、零方差、样本不足等情况按各因子公式处理。价格由复权开关调整，默认日间复权开启，日内各休盘复权关闭。

```bash
python scripts/yq.py run --asset future --frequency minute_1m --start-date 20260424 --end-date 20260424 --factors yq_001
python scripts/update_incremental.py --groups future_mapping --start-date 20260423 --end-date 20260424
```

- `--future-main-source open-interest` 读取 `future_data/basic/mapping_oi/YYYY.parquet`。先运行 `derive-future-main-oi --start-date YYYYMMDD --end-date YYYYMMDD --date-batch-size 120`，区间需覆盖预热日。首个有持仓数据的交易日按当日最大持仓量初始化，并沿用到下一交易日。此后使用上一交易日的持仓量，只切换到更远月份，且新合约持仓量达到旧主力的 1.1 倍。旧主力缺有效持仓量时从更远月份中重新选择。缺上一交易日数据的映射留空。
- `--future-main-source fut-mapping` 是默认选项，优先读取 `future_data/basic/mapping/YYYY.parquet`，缺少日期时兼容旧版按日文件。下载时应覆盖所选因子的预热日期。输出的 `ts_code` 为目标交易日实际选中的合约代码。
- `--future-main-source all-contracts` 对每个真实合约独立计算。再指定 `--future-adjust-daily false` 可计算未复权的全合约分钟因子。

### YQ 分钟趋势因子

| 因子 ID | 公式 | 窗口 |
| --- | --- | --- |
| `yq_026–027` | `(KAMA − EMA) / ATR` | 15、30 |
| `yq_028–030` | `sign(momentum) × ER` | 5、15、30 |
| `yq_031–033` | `sigmoid((U − C)/ATR) − sigmoid((C − L)/ATR)` | 5、15、30 |
| `yq_034–036` | `exp(−abs(C − U)/ATR) − exp(−abs(C − L)/ATR)` | 5、15、30 |
| `yq_037–039` | `tanh(max(C − U, 0)/ATR) − tanh(max(L − C, 0)/ATR)` | 5、15、30 |
| `yq_040–042` | `exp(−abs(C − U)/ATR) × tanh((U_t − U_{t−n})/ATR)` | 5、15、30 |
| `yq_043–045` | `exp(−abs(C − L)/ATR) × tanh((L_t − L_{t−n})/ATR)` | 5、15、30 |
| `yq_046–048` | `(C_t − LL)/(HH − LL)` | 5、15、30 |
| `yq_049–050` | `ER_s × d × (1 + sign(Vol_f/Vol_s) + sign(d × ΔOI_s))` | `(f, s) = (5, 15)、(15, 30)` |
| `yq_051` | `M × (1 + 0.5 × M × A)` | MACD `(15, 30, 5)`，ATR `30` |

所有新因子都带 `pv` 和 `yq` 标签。KAMA 每次仅使用最近 `2n` 根收盘价，以前 `n` 根的均值为初值，再用 `n` 次自适应更新。KAMA 的 ER 周期为 `n`，平滑系数使用标准快慢参数 2 和 30。EMA 使用最近 `n` 根收盘价的归一化指数权重。[KAMA 平滑公式](https://ta-lib.org/functions/kama.html)

布林带使用 `MA ± 1.5σ`，其中 `σ` 为总体标准差，带宽、ATR 和斜率滞后都使用对应窗口 `n`。`HH` 和 `LL` 使用 `t−n` 到 `t−1` 的最高价和最低价，区间宽度为零时留空。所有以 ATR 为分母的 YQ 分钟因子在 ATR 为零时输出 0。ER 的价格路径长度为零时输出 0。

`yq_049–050` 中 `d = sign(C_t − C_{t−f})`，`Vol_n` 是含当前分钟的最近 `n` 根 bar 的成交量之和。`Vol_s` 为零时成交量项取 0。`ΔOI_s` 为最近 `s` 根 bar 的合约内持仓量差分之和，主力换月时沿用合约内差分口径。

`yq_051` 使用 `DIF = EMA_15(C) − EMA_30(C)`、`DEA = EMA_5(DIF)` 和 `H = DIF − DEA`。`M = tanh(DIF / ATR_30)`，`A = tanh(H / ATR_30)`。三次 EMA 均使用有限窗口。
- `--future-main-only true` 默认仅计算并输出目标日的主力合约。目标日缺少可用映射时回退到全合约；设为 `false` 则所有日期都计算全合约。
- `--future-warmup-source main-sequence` 默认用各历史日的主力为目标日主力预热。设为 `same-contract` 则用目标日实际合约的历史。全合约输出使用各合约自己的历史。历史日没有映射时，主力序列预热使用目标日实际合约在该日的历史。
- `oi_diff` 始终先在实际合约内做差，日首根使用该合约上一交易日最后一根 bar 的原始持仓量。差分得到的序列再按所选预热方式拼接。
- 新期货分钟因子采用有限回望。EMA 使用最近 N 根 bar 的归一化指数权重，ATR 的指数平滑也使用有限窗口。同一日期批次的期货分钟行情在因子批次间共享。
- `--future-adjust-daily` 默认开启，连接上一交易日日盘收市和下一交易日的首个开市段。夜盘结束到同一交易日日盘开市、早盘短休和午休分别由 `--future-adjust-night`、`--future-adjust-morning`、`--future-adjust-noon` 控制，默认关闭。
- 每个目标日以声明的预热范围内最早可用的分钟数据为基准。启用的边界按上一段收盘价除以下一段开盘价，把比例累计乘到后续价格。边界价格缺失或非正时跳过该次比例调整。期货分钟输出按日替换本次计算的因子列。

## 回测输入读取 / Backtest input reads

本地文件读写串行，计算线程独立配置。远程可并发读独立文件及写独立目标文件，使用配置的日期批次和同文件写锁。HDFS 部署的进程内读、写并发上限各为 50，读取操作每秒限流 1,000 次，写入使用独立并发额度；这不是集群级 RPC 上限。远程预读复用 reader 和 footer/schema，投影结果按输入顺序消费；PIT 和依赖前序状态的计算顺序不变。

Local file I/O is serial; compute threads are configured independently. Remote I/O can read independent files and write independent destinations concurrently within the configured batches, with per-file write locks. The HDFS deployment allows up to 50 reads and 50 writes per process, with reads paced at 1,000 client operations per second and writes using independent slots, not a cluster-wide RPC limit. Prefetch reuses the reader and footer/schema and delivers projections in input order, preserving PIT and sequential state updates.

manual/ML 因子表在同一次打开中完成 schema 获取和数据读取，列归属、按股票和日期键合并、空值及 forward-fill 规则不变。同一远程回测的多个 factor batch 共享 Barra 投影表，按文件长度和修改时间失效，运行结束释放。`--profile` 新增同时打开句柄峰值及打开延迟 P50/P95 的直方图区间上界，仍是客户端指标，不代表 NameNode/DataNode 的全局请求数。

Manual/ML factor schema and projected data share one open, retaining ownership, keyed merges, nulls and forward-fill semantics. Multiple factor batches in one remote backtest share projected Barra tables until the run ends. Entries are invalidated by file size/mtime. Profile output includes peak live read handles and histogram upper bounds for open-latency P50/P95. These remain client-side measurements, not global NameNode/DataNode request counts.

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

### 期货流动性主力池 / Futures liquidity pool

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-future-main-pool --start-date 20260105 --end-date 20260109
```

日期参数指定要产出的交易日。程序以目标年份上一年的最后一个交易日为窗口终点，向前取 240 个交易日（含终点），通过 `future_data/basic/mapping` 逐日匹配实际主力合约及其期货日线 `vol`。纳入所有具备完整 240 日有效记录且日均成交量至少为 10,000 手的品种。有效的零成交量计入均值。

次年沿用入池品种，每日从原主力映射取实际合约代码。池 ID 为 `liquidity_001`，年度表写入 `data/future_data/basic/mapping_pool/liquidity_001/YYYY.parquet`。同目录自动写入中英文 `README.md`。字段及补跑口径见[池说明](docs/FUTURE_MAPPING_POOL_LIQUIDITY_001.md)。

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

默认 `--exclude-limit true --limit-side both`：先用 T 日信号、股票池和 ST 标记分组并确定权重，再将 T+1 收盘涨跌停股票的目标权重置零。其余权重不变，不补选，空余仓位按零收益现金计至下次调仓。各基准每日将涨跌停股票的收益贡献置零，保留筛选前的有效收益样本作为归一化基数。末日缺少 T+1 过滤表时跳过涨跌停筛选；区间内缺表仍报错。这是事后筛选，不模拟实际成交，不改变策略执行规则。

By default, groups and weights are formed from T-day inputs, then stocks that close at either price limit on T+1 receive zero target weight. Other weights are unchanged and unused allocation earns zero until the next rebalance. Daily benchmarks zero the contribution of limit-hit stocks without changing the pre-screen normalization over valid returns. The final signal date can skip this screen if its T+1 filter file is unavailable; interior gaps remain errors. This is an ex-post screen, not a fill simulation, and does not change strategy execution.

### 期货分钟因子评价

```powershell
cargo run --release --manifest-path factor_engine/Cargo.toml -- backtest --asset future --frequency minute_1m --start-date 20260423 --end-date 20260424 --factors yq_001,yq_002
```

- 因子和标签分别从配置的 `factor_root` 和 `label_root` 读取，可用 `--factor-root` 和 `--label-root` 覆盖。两边均使用根目录下的 `future/minute_1m/{year}/{trade_date}.parquet`，按 `trade_date + trade_time + ts_code` 对齐。每个请求的交易日都需要有对应的两张表。每次运行选择一列标签，默认为 `future_open_return_5m`，可用 `--label` 切换。输出的 Parquet 元数据记录标签列名。使用同一输出目录重跑会替换该因子的摘要。
- 因子选择支持 `--factors`、`--tags`、`--all-factors` 三选一。标签按全部匹配筛选，`--tags pv,yq` 选择这批 YQ 期货分钟因子。标签读取当前期货分钟因子注册表。与股票入口一致，指定外部 `--factor-root` 时用 `--factors` 或 `--all-factors`。
- 支持 `--factor-batch-size`、`--date-batch-size`、`--threads` 和 `--profile`。日历在启动时读取一次，五日窗口跨日期批次延续。股票专用选项如 `--benchmark`、`--barra`、`--neutralize` 和 `--detail` 可传入，在期货分钟回测中不参与计算。
- 远端数据按存储后端的读取并发预取，期货分钟评价按交易日顺序消费结果。计算线程数独立于 HDFS I/O 上限。
- `--future-pool-id` 默认为 `liquidity_001`，按 `future_data/basic/mapping_pool/<pool_id>/YYYY.parquet` 中每个交易日的实际主力合约筛选 All、板块和品种样本。任一评价日缺池映射时报告日期并终止。`--products` 可传 `RB` 或 `RB.SHF`，与池筛选取交集。`--future-pool-id none` 使用全市场口径，同品种若有多个期限，合并分钟样本。
- 默认模式 `--quantile-mode trailing-5` 使用每个品种、因子在 T−5 至 T−1 的每日上下分位数的等权均值。启用池筛选时，每天的历史分位数用当日普通主力合约的因子值计算，跨年新入池品种可以使用入池前的数据。`--quantile-mode daily` 使用评价日有效样本的当日分位数。默认双端为 P05/P95，`--quantile 90` 可切换到 P10/P90。
- 默认结果写入 `data/backtest/future/minute_1m/{factor_name}/summary.parquet`，当日分位模式在 `minute_1m` 下增加 `daily/`。`--output-dir` 可覆盖因子目录的上一级。原有统计列之后写入 `holding_rtn` 和 `holding_minutes`。Parquet 元数据记录池 ID、标签、持仓收益标签、模式和分位参数。
- 每个交易日为每个品种输出一行，`ts_code` 存品种代码。`ic` 和 `cosic` 分别为 Pearson IC 和余弦相似度，使用该品种当日的有效分钟配对。`long_rtn` 和 `short_rtn` 分别为因子值达到上阈值及以上、下阈值及以下的有效标签均值，沿用标签的单位和符号。分位点由当日所有有限因子值线性插值得到。两条阈值重合且当日有有效因子与标签配对时，上下尾收益均记为 0。没有有效配对时，收益留空。`trigger_count` 记录上下端合计的有效合约分钟触发数，All 行为品种合计。有阈值和有效配对但没有触发时记 0，缺少阈值或有效配对时留空。
- 每个交易日还有一行 `ts_code=all`。将全部合约当日的分钟样本拼接，直接对有效因子值和标签计算一次 `ic` 和 `cosic`。两端收益分别在当日有对应有效收益的品种间等权平均。余弦相似度至少需要两个有效配对，任一向量全为零时为空。
- 板块行用 `ts_code=sector:<板块缩写>` 标识，按同一口径在板块内跨合约计算 IC 和 CosIC。板块行的两端收益列留空，报告侧使用各品种定向后的收益汇总板块收益。
- `eligible_contract_count` 记录当日有可用阈值且至少有一个有效因子标签配对的合约数。品种和 All 行记录此分母，板块报告按品种汇总。
- 五日模式只用因子预热，最多读取回测起点之前五个交易日。前五个交易日有任一天缺少该品种的有效分位数时，当日两端收益留空。阈值可用且当日有有效配对时，某一端没有达到该端阈值的有效配对，则该端收益记为 0。IC 和 CosIC 的计算口径相同。
- 持仓检验按实际合约、交易日独立计算。方向由本次回测区间内品种日 IC 的等权均值决定，使用所选评价标签计算的 IC。IC 均值为负时，因子值严格高于上阈值则在次根开盘切换为空头，严格低于下阈值则切换为多头。IC 均值非负或不可用时，高阈值对应多头、低阈值对应空头。阈值之间、恰好等于阈值、信号缺失时维持原仓位。同方向连续触发维持一份仓位。反手在同一开盘时点先结束旧持仓，再开始新持仓。交易日以空仓开始，在末根 K 线开盘平仓。每天把持仓期间的 `future_open_return_1m` 按方向带符号累加得到 `holding_rtn`（bp）。扫描时先累计高多低空的基准收益，区间品种 IC 均值确定后对负方向品种取反，与每分钟反向持仓等价。持仓收益按合约等权汇总至品种，再使用调整后的品种收益等权汇总至 All 和板块。方向使用完整回测区间统计，属于样本内因子评价。
- `holding_minutes` 是当天每笔已闭合持仓的平均交易分钟数。开多至平多、开空至平空分别计一笔。反手的同一开盘时点结束旧仓并开始新仓。时长只累计两个开盘时点连续相差 1 分钟的持仓区间，跳过夜盘至日盘、早盘短休、午休等间隔。品种、All 和板块均以持仓总分钟数除以已闭合持仓笔数。当天无持仓时留空。持仓区间缺收益时，该合约当天的 `holding_rtn` 留空，仍统计时长。老标签文件缺 1 分钟收益列时，这两列留空。
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

标签生产与因子复用相同的 date-batch 共享输入、受控线程池和有界批量写出工具。每个标签仍只看到自身声明的未来交易日窗口，只写目标日；不会把标签未来收益混入因子历史预热。每个标签计算批次完成后更新对应日宽表，保留其他列，统一使用 Snappy 和原子发布。本机和远程后端使用相同的计算与写出链路。

Label production reuses the factor engine's date-batch shared inputs, bounded thread pool and buffered output tools. Each label retains its declared future window and writes target dates only; future label returns never enter factor warmup data. Each label compute batch publishes its daily-wide results, preserving other columns with Snappy and atomic publication. Local and remote backends use the same execution path.

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

## Batch and I/O controls / 批量和 I/O

日期批量和因子批量由 `--date-batch-size`、`--factor-batch-size` 控制。股票分钟派生、raw 物化沿用配置的日期批量。计算使用配置的 CPU 线程，默认使用可用 CPU。内存、合并、缓存与文件数量不施加额外预算。远端读取与写入各 50 并发，读取客户端操作每秒 1,000 次。期货评价 summary 按因子批次并发写出。

Configured date and factor batch sizes control work batches. Stock minute derivation and raw materialization use the configured date batch size. Compute defaults to the available CPUs. Memory, merge, cache and file-count budgets are removed. Remote reads and writes each have 50 independent slots, and reads are paced at 1,000 client operations per second. Futures evaluation publishes factor summaries concurrently within each batch.
