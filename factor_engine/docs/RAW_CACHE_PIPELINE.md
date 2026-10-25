# Raw Cache Placement / Raw 缓存位置

## Task-Local Cache / 任务独立缓存

An optional `[paths].intraday_raw_root` redirects only minute-derived daily raw
files and their metadata. Source inputs, final factors and factor metadata keep
their existing roots. Without this setting, the existing persistent cache under
`factor_root/_cache/intraday_daily` is unchanged.

可选配置 `[paths].intraday_raw_root` 只改变分钟降频 raw 及其 metadata 的位置，
不改变原始输入、正式因子及因子 metadata 的路径。未配置时仍使用原有的
`factor_root/_cache/intraday_daily` 持久缓存，本地命令行为不变。

```toml
[paths]
base_data_dir = "/project/data"
factor_root = "/project/data/factors"
intraday_raw_root = "/scratch/task-unique/intraday_raw"
```

The regular `run` command calculates missing raw with its declared warmup,
reuses the cache within the task, and writes final factors. Parallel annual jobs
must use different raw roots, including different paths when assigned to the
same node. Warmup can overlap because no raw files are shared. This deliberately
recomputes overlapping history rather than coordinating cross-task writers.

正常 `run` 命令按已声明的预热计算缺失 raw，在任务内复用，再生成正式因子。
并行年度子任务必须使用不同 raw 根目录，即使分配到同一节点也不能共用。
跨年预热可以重叠，因为没有共享 raw 文件；代价是各任务独立计算重叠的历史。

The deployment owns temporary-directory creation and cleanup after success,
failure and catchable cancellation. Only the explicit raw directory may be local
when inputs/final outputs use a remote backend. Node loss or an uncatchable kill
requires an external terminal-task cleanup; shell traps alone cannot guarantee it.

部署层负责创建独立目录，并在成功、失败及可捕获取消后清理。远程模式只允许
显式 raw 目录落本地，原始数据和正式产物仍通过远程后端读取及写出。
节点掉线或不可捕获强杀须由外部机制清理已结束任务残留，不能仅靠 shell trap 保证。

## Explicit Shared Preparation / 显式共享准备

The existing `run` and `raw-run` commands keep their default behavior. Shared parallel
preparation is opt-in and does not change factor formulas or PIT visibility.

原有 `run`、`raw-run` 的默认行为不变。并行准备为显式选项，不改变公式或 PIT 可见性。

## Three Steps / 三个步骤

1. Publish the selected raw metadata once with `raw-metadata`, before workers run.
2. Prepare disjoint date ranges with `raw-run --owned-target-dates true`.
3. Run factors with `run --prepared-raw-only true` after all owners covering their
   target dates and raw lookback have succeeded.

1. 工作任务启动前，用 `raw-metadata` 一次发布所选 raw 的版本描述。
2. 用 `raw-run --owned-target-dates true` 准备互不重叠的日期区间。
3. 当目标日期及 raw 回看所需的所有日期所有者都成功后，用
   `run --prepared-raw-only true` 计算正式因子。

```bash
cargo run --release --manifest-path factor_engine/Cargo.toml -- raw-metadata --asset stock --frequency daily --start-date 20251201 --end-date 20260130 --factors cvar95_week
cargo run --release --manifest-path factor_engine/Cargo.toml -- raw-run --asset stock --frequency daily --start-date 20251201 --end-date 20251231 --factors cvar95_week --owned-target-dates true
cargo run --release --manifest-path factor_engine/Cargo.toml -- raw-run --asset stock --frequency daily --start-date 20260101 --end-date 20260130 --factors cvar95_week --owned-target-dates true
cargo run --release --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260101 --end-date 20260130 --factors cvar95_week --prepared-raw-only true
```

Use the same selection, versions and storage root for all steps. The example
prepares December history for January; production planners must derive the actual
lookback from the selected factors rather than assume a one-month boundary.

各步骤须使用相同因子选择、版本和存储根目录。示例用十二月作为一月的预热历史；
生产调度必须按因子的实际回看计算日期依赖，不能固定假设只需要一个月。

## Guarantees / 约束

- Ownership applies to published raw dates, not source-read dates. Providers still
  load their original source warmup and auxiliary dependencies, but cannot publish
  raw rows outside the owned interval.
- Owned preparations never update the shared raw metadata. A completed date with
  no source observations is represented by an empty, typed cache; a missing leg
  is represented by null values. Existing missing-history/min-periods behavior is
  retained without requiring fallback writes in readers.
- Prepared-only factor runs validate cache completeness, fail on missing files,
  columns or incompatible versions, and never calculate/write raw as a fallback.
- A cache file has exactly one preparation owner; final factor files have their
  own disjoint target-date ownership. File locks and atomic publication remain.

- 所有权限制的是 raw 写出日期，不限制读取历史源数据；原始预热和辅助依赖照常读取，
  但不得向责任区间外发布 raw 行。
- 所有者任务不更新共享 raw metadata。源数据不存在的已完成日期保存为有类型的空表；
  单个指标缺失保存为空值，沿用历史不足和 min_periods 规则。
- 正式只读任务检查缓存完整性；缺文件、缺列或版本不兼容即报错，绝不回退补写。
- 每个 raw 日期只有一个准备任务负责；最终因子也按目标日期分配唯一写入者，保留锁与原子发布。

Concurrency and memory requests belong to the scheduler. A yearly factor may
depend on its own raw owner and earlier owners needed for warmup; it need not wait
for future years. Bound the total active pipeline, not merely each stage in
isolation, to avoid doubling concurrent IO when preparation and factors overlap.

并发和内存由调度层控制。年度因子依赖当年及预热涉及的历史所有者，不必等待未来年份。
准备和正式任务交错时，应限制整条流水线的同时运行数，不能把两个阶段的并发上限简单叠加。
