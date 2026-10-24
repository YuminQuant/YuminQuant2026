# Storage and Execution / 存储与计算入口

## Unified entrypoint / 统一入口

从仓库根目录运行。因子参数与原 Rust CLI 保持一致；底层 Cargo 命令仍可使用。
Run from the repository root. Factor arguments match the Rust CLI; direct Cargo commands remain available.

```bash
python scripts/yq.py run --asset stock --frequency daily --start-date 20260401 --end-date 20260424 --factors ret20 --date-batch-size 120 --factor-batch-size 10 --threads 2 --profile
python scripts/yq.py --dry-run run --asset stock --frequency daily --start-date 20260401 --end-date 20260424 --factors ret20
python scripts/yq.py --single-task run --asset stock --frequency daily --start-date 20260401 --end-date 20260424 --factors roe_enhance
python scripts/yq.py factor-train --config ml_alpha/factors/bar_gru_15m.toml
python scripts/yq.py factor-predict --config ml_alpha/factors/bar_gru_15m.toml
```

默认构建 release；`--no-build` 复用已构建程序，`--binary PATH` 指定本地二进制。`--dry-run` 读取日历并打印计划，但不刷新 metadata、不生产数据。
Release builds are the default. Use `--no-build` to reuse a build or `--binary PATH` to select a local executable. Dry runs inspect the calendar and plan without refreshing metadata or producing data.

## Metadata / 注册目录

生产入口先刷新一次相关 metadata。Rust、ML 来源分别为 `factor_metadata.rust.parquet` 和 `factor_metadata.ml_alpha.parquet`，合并表为 `factor_metadata.parquet`，两端规则相同。来源 ID 冲突报错，不相互覆盖。
The entrypoint refreshes the relevant metadata once. Both backends use separate Rust and ML source tables plus the merged `factor_metadata.parquet`; colliding IDs fail rather than overwrite each other.

底层因子、标签和 Barra 生产按代码 registry 选择实现，不要求 metadata 存在或检查是否过期。Metadata 是注册目录，并不保证每个历史日期已生成全部列。回测/模型仍可使用描述信息；部署层可通过 `YQ_METADATA_SNAPSHOTS` 指向提交端生成的只读 metadata 副本，避免并发读取正式目录。
Factor, label and Barra production select code registries without a metadata existence/freshness gate. Metadata describes registration, not historical column availability. Backtests/models may still need descriptions; deployment launchers can supply read-only submission-time copies through `YQ_METADATA_SNAPSHOTS`.

## Task boundaries / 分片边界

因子、标签、Barra 按交易日历每 `date-batch-size` 个目标日划分独立任务，最后一片允许不足。默认本机顺序执行；外部调度器可并发提交同一计划。每片只写自己的目标日期，并自行读取回看历史。
Factors, labels and Barra are split by trading-calendar target dates. The last slice may be shorter. Local execution is sequential; an external scheduler can dispatch the same plan concurrently. Each task reads its own history and writes only its target dates.

任务内保留现有财务慢指标/hold 状态；任务之间不传 checkpoint。独立分片的首日初始化可能与一次连续运行的 hold 状态不同，比较两端时应使用相同分片。需要连续账户或 hold 状态时使用 `--single-task`；策略、回测和训练默认单任务。预测保留原模型窗口，不按因子日期强行拆分。
Existing slow/hold state survives within a task, not across tasks. Initialization can differ from a single uninterrupted hold run, so compare backends using identical slices. Use `--single-task` when continuous state is required. Strategies, backtests and training default to one task; inference retains its model windows.

共享分钟 raw 先由整个区间的唯一准备任务生成，再运行日期消费者。它是正式可复用辅助产物，不是节点数据库缓存。`factor-batch-size` 仍控制每个任务内部的因子批量。
One preparation task generates shared minute raw artifacts before date consumers start. These are reusable outputs, not node database caches. Factor batching remains internal to each task.

每个独立子任务从自己的首个目标日向前读取预热交易日，不依赖前一个任务的内存。20 日有限窗口均值需 19 日历史；先算 5 日指标再做 20 日均值则需 23 日历史。引擎取普通 lookback、DataRequest 回看与 raw 依赖回看的最大值；生成历史 raw 时还会补足 raw 自身的分钟窗口。仅写目标日。历史文件缺失、新股历史不足或 EMA 等无限历史递推，不保证与无限长连续运行完全相同。
Each independent task reads its own warmup. A 20-session finite mean needs 19 prior sessions; a 5-session metric followed by a 20-session mean needs 23. The engine takes the maximum of ordinary, DataRequest and raw-dependency lookbacks, adding the raw producer's own minute window for missing raw. Only target dates are written. Missing source history, new stocks and infinite-history recursions such as EMA can still differ from an uninterrupted run.

## Independent training tasks / 独立训练任务

训练按模型窗口拆分，不按因子 date-batch-size 切训练集。独立入口先用完整配置和日历规划窗口，保留原窗口 ID、训练/验证日期及标签成熟期清洗，再让每个子任务只训练一个模型。本机顺序执行，外部调度器可并发执行同一计划。规划不加载训练张量、不实例化模型。
Training splits by model windows, not factor date batches. The dedicated entrypoint plans against the full configuration/calendar, retaining IDs, train/validation dates and label-maturity purging. Each child fits one model. Local tasks run sequentially; deployment schedulers may dispatch the same plan concurrently. Planning does not load tensors or instantiate models.

```bash
python scripts/train_models.py factor-train --config ml_alpha/factors/bar_gru_15m.toml --dry-run
python scripts/train_models.py factor-train --config ml_alpha/factors/bar_gru_15m.toml --resume
python scripts/train_models.py model-train --config ml_alpha/models/mdl_000001.toml
python scripts/yq.py --split-training-windows factor-train --config ml_alpha/factors/bar_gru_15m.toml
```

通过配置 train_scheme.refit_frequency 选择更新频率。带 validation_lookback 的日历窗口支持 yearly_end、semiannual_end、quarterly_end、monthly_end，分别在年、半年、季度、月开始前一个交易日收盘拟合，并按配置回看历史。年度窗口继续使用 annual_YYYY，标识与日期口径不变。其他训练方案保留现有 refit 规则，包括固定交易日间隔；static 配置仍只有一个模型。
Select cadence with train_scheme.refit_frequency. Calendar windows with validation_lookback support yearly, semiannual, quarterly and monthly refits at the preceding period's final trading close. Existing annual IDs and dates remain unchanged. Other schemes retain their refit rules, including fixed trading-day steps; static configurations still produce one model.

子任务写各自的 artifact_dir/window_id/，不写预测列、不并发写汇总诊断；全部成功后由单独步骤汇总诊断。可用 --window-id annual_2020 单独重试；factor-train 的 --resume 继续校验模型 manifest。预测使用原配置寻找同一模型，不拆分有时序状态的预测。原 scripts/yq.py 训练命令不带拆分参数时仍是单任务。
Each child writes its own artifact_dir/window_id/, without predictions or concurrent aggregate-diagnostic writes. A final step aggregates diagnostics only after all children succeed. Use --window-id annual_2020 for a retry; factor resume still validates manifests. Prediction locates the same models with the original configuration and retains sequential state. Ordinary training through scripts/yq.py remains single-task unless splitting is requested.

## Paths and adapters / 路径与适配器

公共配置 `[paths]` 支持 `base_data_dir`，以及可选 `factor_root`、`label_root`、`barra_root`。默认仍为本地 `data/` 布局。`YQ_ENGINE_CONFIG` 可指定引擎配置；显式 CLI 配置优先。
Public configuration supports `base_data_dir` and optional factor, label and Barra roots. Local `data/` defaults remain unchanged. `YQ_ENGINE_CONFIG` selects an engine configuration unless explicitly overridden.

部署启动器和公司配置必须保留在仓库外。可在未跟踪的项目 TOML 中选择适配器，公共仓库不包含企业客户端、服务器名、凭证或鉴权根目录：
Keep deployment launchers and organizational configuration outside Git. An ignored project TOML can select an adapter; public code contains no enterprise client, hostnames, credentials or authorized root:

```toml
[execution]
deployment_module = "deployment_package.runner"
module_root = "/absolute/private/module-directory"
deployment_config = "/absolute/private/storage.toml"
```

部署脚本负责同一计划的调度、CPU/内存/IO 并发参数、认证续期、失败日志与节点清理。只允许代码、二进制、配置和认证材料临时落在节点；数据库、特征和结果不暂存到节点。
Deployment scripts own scheduling, resource/IO limits, renewable authentication, failure logs and cleanup. Only code, executables, configuration and credentials may be staged on nodes, never datasets, features or computed results.

## IO contract / 读写契约

Rust 接口位于 [`storage_io.rs`](../src/storage_io.rs)，经 `artifact_io` 导出。`ArtifactBackend` 提供 stat/list、随机读取、流式写入、删除、重命名和锁。`ReadAt` 接入 Parquet `ChunkReader`，支持 footer、列投影与现有过滤；流式 Parquet 与文本写入也经同一后端。
Rust exposes `ArtifactBackend` through `artifact_io`: stat/list, random reads, streaming writes, removal, rename and locking. `ReadAt` implements Parquet's chunk reader, preserving footer reads, column projection and filtering; streaming tables and text use the same backend.

后端启动时注册一次；本地无需注册。远程客户端由进程复用、限制 IO 并发，不扫描本地数据缓存或构造全任务数据清单。DataRequest 日期选择、provider 日期隔离、PIT 和 report type 逻辑不变。
Register a backend once at startup; local execution needs no registration. Remote clients are process-reused and IO-limited, without node data-cache scans or full input manifests. DataRequest isolation and PIT/report-type semantics are unchanged.

Python 使用 [`output.artifact_io`](../../ml_alpha/yq_ml_alpha/output/artifact_io.py) 读取表格、日历、特征及模型。远程训练使用当前窗口内存张量，预算由 `YQ_TENSOR_MEMORY_BYTES` 控制；超限明确失败，不回退节点 memmap。可复用特征和模型写入配置的远程目录。本地原有 memmap 选择不变。
Python routes tables, calendars, features and models through `output.artifact_io`. Remote tensor assembly stays in RAM under `YQ_TENSOR_MEMORY_BYTES`; excess fails explicitly, never spills to node memmaps. Reusable features/models go to remote storage. Existing local memmap choices remain available.

新表格 Parquet 默认 Snappy，不批量改写历史文件。宽表修改在锁内按股票/日期键合并、写同后端临时文件再原子发布；异常不替换旧文件。
New table Parquets default to Snappy, without rewriting history. Wide-table updates lock, merge by keys, write a same-backend temporary file and publish atomically. Failed writes leave the previous file intact.

```bash
python scripts/yq.py remove-factor-columns --start-date 20260401 --end-date 20260424 --columns obsolete_factor
```

删除列也使用锁和原子发布，不能删除股票/日期键；不会删除注册信息。`--profile` 额外报告读取字节、打开文件数、目录/状态请求和 IO 耗时。通用计数是逻辑 IO 调用，不等于 NameNode RPC 次数。
Column removal uses the same locking/publication contract, protects row keys and does not unregister factors. Profiling reports bytes, opens, directory/status calls and IO timing. Generic counters represent logical calls, not exact NameNode RPCs.
