# 单模型数据与预测工作流优化

因子范围、股票筛选、采样、训练窗口、标签到期隔离、网络结构、损失、batch_size
和 Optuna 次数继续由原模型配置和训练流程决定。

## 1. 序列整数索引复用

`DailyTensorSequence` 为每日截面建立连续 float32 数组和股票整数行号，并为各目标日
保留历史交集及其行映射。FactorGCL 的两路共用预处理后的每日数组。后续 epoch 按行号组装当前股票批次，张量使用后释放。完整重叠窗口
无需提前展开成常驻张量。模型自身的窗口内归一化仍在原有位置执行。

## 2. HDFS 训练输入准备

分窗提交时，公共任务规划统一解析 `__all__` 和标签筛选。提交器通过
`YQ_FACTOR_COLUMN_CATALOG` 传递有序列清单，子任务跳过规划阶段列扫描。
按日期维护因子文件索引，读取同一个 Parquet 时在同批输入范围内复用文件状态。

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

## 5. 按日保留准备好的 BarPanel 张量

BarPanel、RawBar、MultiBarPanel 和 ResNetPanel 默认使用 `tensor_storage="memory"`。
训练和验证按日期组装、归一化一次，股票批次只从当天的 float32 数组取行。完整窗口
无需再次拼成一块大数组。HDFS 训练与预测的 ResNet 输入都按目标日期的源数据并集预读，
并发读取标签，预测按日期批次释放张量。本地读取保持串行。

显式指定 `tensor_storage="indexed"` 或 `"mmap"` 可沿用本地临时存储路径。
存储模式不改变复权、聚合、截面预处理、筛选、标签或模型 manifest。
