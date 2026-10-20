# 东方证券 ResNet-GRU：week 与 msnew

依据东方证券 2023-08-24《因子选股系列之九十六：基于残差网络的端到端因子挖掘模型》
第 6–7 页。实现两个独立端到端因子：`resnet_gru_week`、`resnet_gru_msnew`，券商标签
`DFZQ`。本轮不实现旧版人工分钟特征 `ms`、l2、跨数据集非线性加权或指数组合优化。

## 输入与单位

### week

- 截至样本日 T 的最近 150 个市场交易日，按由远到近顺序分为 30 组，每组连续 5 日。
  分组随 T 移动，不是自然周，也不是把每 5 日聚合成一根周 K 线。
- 每张图 6×5，通道顺序 open/high/low/close/vwap/turnover_rate_f。
- 日线 VWAP = amount × 10 / vol（日线 amount 千元、vol 手）；价格乘当日 adj_factor，
  全部 150 日价格统一除以样本日 T 的复权收盘价。并非各个 5 日图片分别除以自身最后收盘价。
- 用户确认使用自由流通换手率 turnover_rate_f，保留源数据百分数，不截断、不做指数变换。
- 读取 daily/pv、daily/adj_factor、daily/basic；无需额外派生、落盘 150 日重叠窗口。

### msnew

- 最近 30 个交易日，每日 8 根完整 30 分钟 K 线，组成 30 张 5×8 图片。
  通道顺序 open/high/low/close/amount。
- 价格除以该日 daily/pv.pre_close，使用行情源的前收参考价处理除权日基准；不从未来日期
  获取价格或复权因子。成交额除以该日流通市值：minute amount / (circ_mv × 10000)，
  分钟金额单位元，daily/basic.circ_mv 单位万元，结果为比例。
- 从 Rust 公共 `derive-bar --bar-size 30` 的输出读取。按 bar_index 0–7 排序，每根必须
  minute_count=30；不完整日不补齐、不跨午休拼接，不在训练进程中重新聚合分钟数据。
- Rust 已支持半小时聚合、日期并发及 Snappy 日度 Parquet，本轮复用，未重复新增派生命令。

所有日期按股票代码对齐。源文件或必要字段缺失会报出路径；单只股票缺失、非有限值、
非正价格/市值、非法换手或不完整历史，则不形成该样本。预测和训练使用相同构造过程。
股票池沿用 mkt_all，剔除 ST、北交所，源历史中的 ST/BJ 筛选也保持一致。

## 网络、目标与约定

每张图片使用同一个残差块：

```text
主路：Conv1d(kernel=3, channels=32, padding=0) → ReLU
旁路：时间轴两端各裁剪1格 → Conv1d(kernel=1, channels=32)
两路相加 → 展平 → 30 步两层 GRU(hidden=64, dropout=0.1)
→ 最后时点隐藏状态 → Linear(64,64) → BatchNorm(64)
→ 固定权重 1/64 求和 → BatchNorm(1) → 每只股票一个最终分数
```

图片内只沿时间轴卷积，字段为通道；普通 Conv1d 并不保证通道置换不变，必须固定字段顺序。
week 图片编码后维度 32×3=96；msnew 为 32×6=192。
不添加图片跨股票注意力或跨数据集融合。

训练目标为研报公式：标准化收益标签的 MSE，加 `lambda/(N*K*K) * ||Z.T @ Z||_F`，
其中 Z 为内部 BN 后的 K=64 维因子，保留 Gram 矩阵对角线。默认 lambda=1.0。
标签在同一训练日期 batch 内中心化、除以总体标准差；常量标签 batch 不训练。
BN 预测使用保存的运行统计量，推理分批不会重新拟合标准化参数。

研报未披露卷积核/通道数、激活函数和全部训练超参。本次用户确认单残差块、核3、
32通道、ReLU、两层64维GRU；lambda=1.0 是可配置的工程初值，不声称为研报最优参数。
不改变原有 rawbar_multi_gru 的负 IC 训练目标或最终等权输出实现。

沿用项目训练方案与显存约束：

- 每年年末重训；近 36 个月前 33 个月训练、后 3 个月验证，每 5 个交易日采样。
- 10 日 VWAP 标签，lookahead_days=11；公共层截断未到期标签，并隔离训练/验证边界。
- 单种子42，batch_size=3000，100 epoch，学习率0.001，patience=10。
- 保留公共训练器按完整日期 Pearson IC 验证/提前停止的口径；不是以含正则项的训练 MSE
  选模型，也未把验证替换为 RankIC。
- 为保持既有 batch_size，超过3000只股票的日期分批训练，BN与正交惩罚按该日期的子批计算，
  不宣称严格等价于研报全截面一次训练。
- 训练配置2013–2025，样本外预测2016-01-01至2026-04-24，实际日期由公共日历调度决定。

## 数据路径与资源

派生半小时数据位于 `data/derived/stock/bar/30m/<year>/<date>.parquet`，为各模型共享。
provider 只保留有界内存中的单日数据（week最多150日、msnew最多30日），没有新增磁盘缓存。
训练默认复用公共 indexed 单日快照与窗口索引通路，临时数据位于各模型 D 盘 workspace
的 cache/scratch，按 bundle 生命周期释放；`cache_samples=false` 不代表禁用工作文件。
原 `tensor_storage="mmap"` 仍可显式选择，详见 [PERFORMANCE_WORKFLOW.md](PERFORMANCE_WORKFLOW.md)。
一个样本的特征 float32 体积：week 150×6×4=3600 字节，msnew 30×8×5×4=4800 字节，
未计键、标签、模型中间激活和文件开销。不会把全部图片先保存成重复的派生窗口库。

## 命令

以下命令在仓库根目录执行，本轮开发没有运行正式派生、训练、预测或回测。

已有日历下，首个训练样本为2013-01-04；week需从2012-05-29起的150日日线，
msnew需从2012-11-21起的30日半小时数据。修改训练起点后应重新按交易日回推历史。

```powershell
# 仅 msnew 需要生成半小时派生数据；date-batch-size 是并发日期数。
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar `
  --asset stock --source minute --bar-size 30 `
  --start-date 20121121 --end-date 20260424 --date-batch-size 20

$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = "$PWD\ml_alpha"

# 两个因子独立训练与预测，各输出一个因子列。
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\resnet_gru_week.toml
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\resnet_gru_msnew.toml

# 分阶段执行时，将 factor-run 分别替换为 factor-train / factor-predict。
```

标签库需事先具有覆盖训练/验证样本的 future_vwap_return_10d。
模型分别保存于 `data/model_workspace/resnet_gru_week/artifacts` 和
`data/model_workspace/resnet_gru_msnew/artifacts`。最终分数由公共写入器合并到
`data/factors/stock/daily/<year>/<date>.parquet`，对应同名因子列；64个内部因子不额外落盘。
配置与 registry 已登记；预测时由公共流程写入 metadata，也可用 `factor-metadata --config ...`
显式同步元数据。本轮不修改已有正式因子数据。
