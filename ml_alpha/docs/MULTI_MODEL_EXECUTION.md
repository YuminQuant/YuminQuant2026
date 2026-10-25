# Multi-model execution / 多模型执行

Training is scheduled by `(model configuration, original window_id)`. Prediction
is scheduled by output destination and disjoint target dates. Execution settings
are separate from scientific configuration and do not change artifact manifests.

训练按“模型配置、原始 window_id”拆分；预测按输出目录和互不重叠的目标日期拆分。
执行设置独立于模型配置，不改变训练窗口、标签成熟期或模型 manifest。

Models declaring `training_requires_previous_window=True` retain one sequential
training task for all their windows. Other models may still train concurrently.
声明依赖前序窗口的模型保留单任务顺序训练，不妨碍其他模型并发。

## Entry points / 入口

```bash
python scripts/yq.py factor-train --config ml_alpha/factors/a.toml --config ml_alpha/factors/b.toml --execution-config ml_alpha/execution.example.toml --list-tasks
python scripts/yq.py factor-train --config ml_alpha/factors/a.toml --config ml_alpha/factors/b.toml --execution-config ml_alpha/execution.example.toml
python scripts/yq.py factor-predict --config ml_alpha/factors/a.toml --config ml_alpha/factors/b.toml --execution-config ml_alpha/execution.example.toml
```

The filenames above are placeholders. `model-train/model-predict` accept the same
options for signal configurations. Existing single-model commands remain valid.
`--list-tasks` only plans; `--task-id ID` executes exactly one planned task.

上面的文件名为示意。信号模型使用同样的 `model-train/model-predict` 参数。
原单模型命令保留；`--list-tasks` 只规划，`--task-id ID` 只执行一个规划任务。

## Resources / 资源

Copy the generic execution example to a chosen configuration path. Local defaults
are serial: `train_max_parallel=1`, `predict_max_parallel=1`,
`model_predict_workers=1`. The first two settings control independent process/task
concurrency; the third controls compatible CPU models within a prediction task.
Neural/GPU models remain sequential within a task. CPU model concurrency requires
explicit class capability and bounded native thread pools, not just a thread pool
around unrestricted BLAS kernels. `threads` is the per-task compute budget.

执行示例默认本地串行。前两个并发参数控制独立训练/日期任务，第三个控制任务内
兼容 CPU 模型的并发；神经网络/GPU 模型任务内顺序推理。CPU 并发要求模型显式
声明能力，同时限制 BLAS 原生线程，防止线程池叠加。`threads` 是单任务计算预算。

Cache limits apply to raw Arrow tables/footer, pending scalar scores and resident
model weights separately. `memory_budget_mib` checks process resident RAM at stage
boundaries; it is not an OS allocation limiter. Tensor assembly also uses the
existing spool budget. Oversized inputs fail explicitly rather than silently
changing model batch semantics or spilling remote data to disk. Concurrent tasks
each need their own reservation; these settings do not reserve cluster resources.

各缓存分别限制原始 Arrow/footer、待写分数和模型权重；总内存在阶段边界检查，
并非操作系统级分配上限。张量组装继续遵守既有 spool 预算。超限明确失败，
不静默改变模型批次，也不将远程数据溢写节点磁盘。并发任务各自需要内存，
执行参数本身不代表已经申请集群资源。

## Windows and sharing / 窗口与共享

Models may have different refit frequencies, prediction ranges and batch sizes.
The planner never edits their date configurations. Each prediction computes its
original complete input batch; only owned target dates are written. Boundary
context can therefore overlap between tasks, but output date ownership does not.
Missing-history, label maturity and model activation rules remain unchanged.

不同 refit、预测范围和批次可以共存，不裁剪训练配置。预测仍计算原始完整输入
批次，只写任务拥有的日期；任务边界可能重复计算上下文，但正式输出日期不重叠。
历史不足、标签成熟和模型生效规则不变。

`AlphaModel` defaults to conservative capabilities. `prediction_replay_safe=True`
allows independent date tasks; otherwise the complete prediction range is one
task. `prediction_inputs_readonly=True` permits bundle sharing only when feature
order, sequence settings, preprocessing, filters, universe, data version and exact
input dates match. `prediction_cpu_parallel=True` additionally permits concurrent
CPU calls. The baseline linear/mean models and reviewed bar-GRU input paths declare
their respective capabilities; extensions do not become safe merely by sharing a
file path. Report-event models retain their existing replay workflow.

默认采取保守能力声明：日期拆分、只读输入共享、CPU 并发分别显式声明。
共享还必须匹配特征顺序、序列、预处理、过滤、股票池、版本和实际输入日期。
线性/均值基线及已审阅的 bar-GRU 路径声明对应能力，不能只因路径相同就共享；
研报事件模型继续使用原有回放入口。

Raw tabular projections may union columns, never dates seen by a model. Each
consumer receives only its requested columns and retains its own transformations.
Unknown/mutable inputs are isolated. Builder cleanup inside a task releases owned
feature state without clearing the task's byte-bounded raw read cache.

原始表可合并列投影，但不扩大模型看到的日期；每个消费者只获得自己的列，
预处理仍隔离。未知或可修改输入不共享。Builder 清理自身特征状态时，不再清空
联合任务的有界原始读取缓存。

## Publication and paths / 发布与路径

The coordinator retains scalar scores only until every selected model covering a
date has completed. It joins on `(trade_date, ts_code)` and calls `write_many`
once per date. Uncovered columns remain untouched; covered empty predictions
retain existing NaN-overwrite behavior. Locks, Snappy and atomic per-file publish
remain active. A failed prediction batch is not partially flushed; a multi-date
batch is not a cross-file transaction, so retries must be idempotent.

协调器仅暂存标量分数，待对应日期的模型全部完成后按股票日期键合并，一天写一次。
未覆盖列不改；覆盖但为空的预测沿用 NaN 覆盖规则。文件锁、Snappy 和单文件原子
发布保留。预测失败不提前写该批的不完整模型列；跨日期不是整体事务，重试须幂等。

Both local and remote paths preserve the configured relative layout:

本机和远程保留相同的相对路径结构：

```text
data/model_workspace/<run>/artifacts/<window_id>/model.pkl
data/model_workspace/<run>/artifacts/<window_id>/model.manifest.json
data/factors/stock/daily/<year>/<date>.parquet
data/models/<year>/<date>.parquet
```

Configured non-default roots remain supported. Storage relocation is a deployment
concern; company adapters, credentials and scheduling settings stay outside Git.
Matching relative layouts do not bypass existing data/backend identity checks on
model artifacts. Training artifacts are not automatically portable across data
versions or relocated sources with different manifest identities.

非默认根目录仍由配置决定。存储路径映射由部署层负责，公司适配器、鉴权和调度
设置不入 Git。相对路径一致不代表绕过模型数据身份检查；不同数据版本或不同身份
的数据源不能直接混用模型 manifest。

Logs include task resources, shared-bundle counts, stage RAM and managed-reader
open/footer/cache counters. These are task-side observations, not cluster-wide
storage service load measurements.
日志记录资源、共享输入次数、阶段内存及读取器打开/footer/缓存计数；这些只是
任务侧观测，不能据此声称已验证整个存储集群无压力。
