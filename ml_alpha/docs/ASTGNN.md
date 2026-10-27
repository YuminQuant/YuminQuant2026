# 东方证券 ASTGNN：week / msnew 端到端因子

依据东方证券 2024-02-28《因子选股系列之一〇一：自适应时空图网络周频 alpha 模型》
第 4、6–7 页的输入说明、公式和网络图。配置为 `astgnn_week`、`astgnn_msnew`，
券商标签 `DFZQ`；两套网络分别训练，每个交易日各为每只合格股票输出一个因子。
它们是端到端因子，不占用 `mdl_*` 因子合成模型编号。

## 本次范围与研报未披露项

用户确认复用项目 week / msnew 数据、64 个内部因子等权输出、单种子 42、
10 日 VWAP 标签、年度重训、近 36 个月按 33/3 划分、每 5 个交易日采样。
不实现 day、Level-2、可选先验图临近损失，也不接第二阶段决策树或跨数据集组合。
研报将半小时图片输入称为 ms，本项目沿用已有数据接口名 msnew。

研报明确使用 GRU 隐状态作为自适应图层输入，但没有披露全部实现细节。本版直接令
节点嵌入 M = GRU 最后隐状态 H；M 由共享编码器训练产生，不是固定股票代码 embedding，
因而允许样本外出现新股票，也不要求固定节点数量。图层取一层、输出宽度 64、ReLU；
不额外拟合一套节点嵌入、不做余弦归一化、缩放点积或 Top-K 筛边。
这些明确记录的工程选择不声称是研报未公开的原始代码。

ResNet 卷积核 3、通道数 32，两层 64 维 GRU、dropout 0.1、学习率 0.001、
100 epoch、patience 10、梯度裁剪 5、正交权重 1.0，沿用现有 ResNet-GRU 参数并补充
梯度裁剪。研报未给出完整损失和这些超参；标准化 MSE + Gram 正则及输出 BN
采用现有 `resnet_gru_*` 的项目实现，不声称本报告规定了这些数值。

## 输入与预处理

完全复用 [RESNET_GRU.md](RESNET_GRU.md) 的 `ResNetPanelProvider`：

- week：截至 T 的 150 个交易日，30 张 6×5 图片，字段为
  open/high/low/close/vwap/turnover_rate_f。价格当日复权后统一除以 T 日复权收盘价；
  换手率保留源百分数。图片是滚动五日片段，不是自然周 K 线。
- msnew：截至 T 的 30 个交易日，每日 8 根完整半小时 K 线，30 张 5×8 图片，
  字段为 open/high/low/close/amount。价格除以各日 pre_close，金额除以该日流通市值，
  复用 Rust `derive-bar --bar-size 30` 的公共产物。
- 源缺失、历史不完整或非有限值的股票不构成样本；ST/BJ 筛选、股票池沿用现有规则。
  没有额外引入未来价格、未来股票池，也没有使用最新关系图回填历史。
- 对已经合格的输入节点，不因其未来收益标签缺失而删股票。公共
  `FeatureProvider.requires_full_cross_section` 和 `DatasetBundle.full_cross_section`
  传递此约束，`features.params.full_cross_section=true` 启用；旧模型默认行为不变。
  ASTGNN 训练/预测拒绝没有这一标记的 bundle，防止用普通股票子批冒充完整图。

## 网络与计算

```text
每股 30 张图片
  → 共享残差卷积编码
  → 两层 GRU，取最后隐状态 H（N×64）
  → A = softmax_row(ReLU(H Hᵀ))
  → Z = ReLU((H + A H) W)，W 无偏置
  → Linear(64,64) → BatchNorm(64)
  → 固定 1/64 等权求和 → BatchNorm(1)
  → 每股一个分数
```

A 含对角线，再额外加公式中的 I 项；不把自身边删掉。ReLU 会把负相似度置零，但
softmax 后这些位置仍通常有非零权重，因此这里实际是稠密图，并非稀疏邻接矩阵。
对所有节点共享 W；先算 H W 再计算 A(H W)，与 (A H)W 等价。

每次优化只处理一个完整交易日。缺少标签的节点参与 GRU、图交互和 BN，
只在监督 MSE、目标标准化及 Gram 惩罚中屏蔽；常量标签日期跳过。
损失为同日有效标签标准化后的 MSE，加
`orthogonal_lambda * ||BᵀB||_F / (N_labeled * K²)`，K=64。
验证按完整日期 Pearson IC 取日期均值，以负 IC 最小选择最佳模型。
BN 推理使用模型保存的统计量，不重新拟合样本外统计量。

训练、验证、预测都逐日构图；`predict_batch_size` 只控制同时加载的日期数，不混合日期。
股票编码 chunk 不决定图的节点范围。所有 query 块均读取完整当日 keys/values，
邻居张量不 detach，梯度可以穿过全部节点和自适应边权。

## 日期与公共防泄漏规则

- 配置训练范围 2013-01-01 至 2025-12-31，预测范围 2016-01-01 至 2026-04-24。
- 每个预测年使用之前三年，前 33 个月训练、后 3 个月验证，训练/验证各按现有窗口规则
  每 5 个交易日取样。预测每日输出。
- `future_vwap_return_10d` 对应 `lookahead_days=11`。公共 `build_windows` 先排除
  截至训练日收盘尚未到期的标签，再隔离训练/验证边界。原始 33/3 日历范围不代表所有
  末尾日期都进入训练。重训日 T 最晚可用的标签截面受 T-11 交易日边界约束。
- 数据构造所需的 150 / 30 日输入历史与标签成熟隔离分别处理，不能混为同一个 lookback。

## 资源与路径

公共 `CrossSectionBarAlphaModel` 管理完整日期优化、验证、预测；ASTGNN 自己管理网络
和损失。独立股票编码仍沿用 `batch_size=3000`，不提高原有上限；图 query 块为 512。
训练默认使用 activation checkpointing，反向时重新计算编码和注意力，降低激活显存占用，
代价是增加计算。编码阶段没有 BN，重算不重复更新 BN 运行统计。

图计算规模仍为 O(N²H)。float32、N=5000 时，完整 N×N 权重约 95.4 MiB，
512×N 的单块约 9.8 MiB；这是单个权重张量的量级，不是总显存上界，softmax、
反向梯度、编码器和优化器还会额外占用显存。没有实际运行，不提供性能提升或显存峰值承诺。
图、邻接矩阵和中间 64 因子不落盘，不新增图派生文件。

训练张量使用公共 indexed 单日快照与窗口索引工作流：

- `data/model_workspace/astgnn_week/cache/scratch/indexed-*`
- `data/model_workspace/astgnn_msnew/cache/scratch/indexed-*`

路径在本项目 D 盘内，随 bundle 生命周期释放，不使用 C 盘或系统默认临时目录。
`cache_samples=false` 禁用样本 Parquet 导出，不代表禁用上述临时单日快照。
每个样本 float32 特征净体积分别为 3600 / 4800 字节，按实际训练验证行数线性增长。
不把重叠输入窗口新增为长期派生数据集。

模型文件为各自 workspace 下 `artifacts/annual_<year>/model.pkl`（Torch checkpoint）；
诊断文件在同一窗口目录。最终因子合并写到
`data/factors/stock/daily/<year>/<date>_ml.parquet`，列名分别为
`astgnn_week`、`astgnn_msnew`。正式数据和 metadata 由未来 CLI 执行时生成。

## 手动执行命令

以下仅为后续使用，本轮没有执行派生、训练、预测或回测。

```powershell
# week 无新增派生步骤。msnew 复用已有完整半小时数据；缺少时先生成。
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-bar `
  --asset stock --source minute --bar-size 30 `
  --start-date 20121121 --end-date 20260424 --date-batch-size 120

$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = "$PWD\ml_alpha"

& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\astgnn_week.toml
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\astgnn_msnew.toml
# 分开执行时，用 factor-train / factor-predict 替换 factor-run。
```

首个训练样本若为现有日历的 2013-01-04，week 要求 2012-05-29 起的日线，
msnew 要求 2012-11-21 起的半小时数据。调整训练起点时应按交易日重算输入历史起点。
标签库须覆盖相应训练/验证样本。正式元数据可另用 `factor-metadata --config ...` 同步。
