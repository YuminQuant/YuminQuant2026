# 单模型数据与预测工作流优化

因子范围、股票筛选、采样、训练窗口、标签到期隔离、网络结构、损失、batch_size
和 Optuna 次数继续由原模型配置和训练流程决定。

## 1. 序列整数索引复用

`DailyTensorSequence` 为每日截面建立连续 float32 数组和股票整数行号，并为各目标日
保留历史交集。后续 epoch 按行号组装当前股票批次，张量使用后释放。完整重叠窗口
无需提前展开成常驻张量。模型自身的窗口内归一化仍在原有位置执行。

## 2. HDFS 训练输入准备

分窗提交时，`__all__` 按因子根路径和训练截止日扫描一次。公共任务规划返回有序列清单，
提交器通过 `YQ_FACTOR_COLUMN_CATALOG` 交给各训练子任务，各模型再应用原有的列过滤。

因子历史按目标日期的历史并集跨窗口预读，标签也并发加载。手工和 ML 分区分别在一次
Parquet 打开中投影，按原有列归属及键规则合并。历史截面标准化后在进程内跨 epoch
复用，历史读取保留 Parquet 元数据并避免另存一份原始 Arrow 表。HDFS 继续使用
独立的 50 路读写和每秒 1000 次读取节流。本地读取保持串行，这套路径只在内存中缓存数据。

## 3. 模型窗口内的预测会话

公共 `AlphaModel.prediction_session()` 在一个模型窗口的日期批次之间保留设备模型。
BarGRU、MultiBarGRU、ResidualGRU、Logsig、门控与 ASTGNN 不再每批迁回 CPU。
Chip-age 在会话内复用网络；单种子不重复加载同一份权重，多种子仍顺序复用一个网络。
会话结束（包括异常）将网络迁回 CPU，并清理会话状态。
普通 BarGRU、MultiBarGRU、ResidualGRU 移除逐训练日的 `empty_cache`；
保留模型结束、阶段切换等资源交接位置的清理。没有启用混合精度或改变优化器。

## 4. 多列合并写出

`DailyWideWriter.write_many()` 接收多个因子的分数和各自覆盖日期，一天只读取、合并、
压缩、替换一次宽表。未覆盖该日期的列保持原值；覆盖日期没有分数时写 NaN，
与单因子覆盖语义一致。单列入口也使用相同原子写入和文件锁。
`.parquet.lock` 是很小的持久锁文件，保留文件以维持跨进程锁的同一标识；
进程退出后操作系统自动释放锁。未接入此锁的外部脚本/Rust 写入任务不受保护，
不应与它们同时修改相同正式日文件。

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = "$PWD\ml_alpha"
& $py -B -u -m yq_ml_alpha factor-predict `
  --config ml_alpha\factors\astgnn_week.toml `
  --config ml_alpha\factors\astgnn_msnew.toml
```

要求已有模型，输出路径、日历、预测范围和预测窗口边界一致。
先顺序预测各模型，再统一写当前窗口；GPU 同时只驻留一个模型。
CPU 只保留当前窗口的标量分数，股票代码使用 categorical。
不训练、不产生永久分数暂存文件；报告文本事件模型继续使用专门回放入口。
单个 `--config` 兼容原用法。分别执行单因子 CLI 仍各写一次宽表，不会自动合并。

## 5. 单日快照与窗口索引替代重叠窗口文件

BarPanel、RawBar、MultiBarPanel、ResNetPanel 训练默认
`materialize.tensor_storage="indexed"`，ASTGNN 也显式启用。

1. 当前训练/验证 bundle 内，单日、单分支基础数据只保存一次。
2. 保存目标日、源日期、合格股票和最终标签筛选后的行位置。
3. 模型访问批次时复用原 provider 重建目标日窗口；同日各股票块复用这个窗口。
4. 切换日期释放上一窗口；训练完成或失败后关闭 bundle，删除临时快照。

继续调用原 provider 的复权、聚合、缩放与完整历史筛选。截面预处理先在原始合格
股票范围内完成，再筛选标签有效行；图模型保留无标签节点。基础快照保持输入精度，
需要时使用 float64，最终模型批次仍为 float32。训练顺序与 batch_size 不变。

临时目录为 `data/model_workspace/<模型或因子>/cache/scratch/indexed-*/`。
存储器拒绝解析后位于 C 盘的路径，不使用系统默认临时目录，没有 Pickle 或跨模型永久缓存。
快照量随不同源日数增长，不随窗口展开次数增长。正常/异常退出会清理；
强制终止进程可能留下未清理的目录。

快照读取缓存使用 256 MiB 计费预算；单个完整目标日窗口、股票索引、provider 源日缓存、
模型批次和操作系统页缓存另占内存。每轮会重新组装窗口，因此减少临时写入和存储的同时，
增加了部分窗口组装工作，实际训练耗时需要后续测量。

旧模式保留：

- `tensor_storage="mmap"`：保存展开后的 float32 窗口，减少每轮窗口组装。
- `tensor_storage="memory"`：完整展开张量放内存。
- 预测不创建 indexed 快照，继续按日期批次加载和释放。
- Chip-age/HIST 等 lazy sequence 模型使用第 1、2 项通路，不切换到此模式。

日志报告源快照数、实际临时文件体积和样本行数。存储模式不属于模型 manifest 的训练
语义，不需要仅因为执行模式切换重新训练已有模型。
