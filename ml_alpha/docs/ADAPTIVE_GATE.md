# 单门自适应因子合成模型 mdl_000023

依据华泰证券 2023-12-01《自适应网络：从削足适履到量体裁衣》，第 8 页图 6、
第 9 页表 7、第 17–18 页单门参数。这是 `models/` 下的因子合成实验，
输入已有因子库，不是 `factors/` 下的端到端因子，不新增原始行情派生流程。

## 网络和损失

设原始因子数为 F（动态取已有合格列，不限定研报的 42/44 个）。
主干为 F→64→64→64→1，三个隐藏层均使用 LeakyReLU。
单门读取原始因子：F→64→BN→ReLU→2；第一类跳过，第二类通过。
门控制第二个 64→64 隐藏层及其激活函数；跳过时直接使用第一个隐藏层输出，
不是将跳层输出相加的残差网络。

训练使用 hard Gumbel-Softmax，前向为 0/1 决策，反向采用 straight-through 梯度。
对 logits 使用 Gumbel-Softmax，与先 softmax 再取对数进行 Gumbel 采样等价；
不能将 softmax 概率再次当作 logits。验证/预测改用确定性 argmax，不抽样噪声，
BN 使用训练运行统计，预测时仅为通过门的股票执行中间层。

每个交易日独立计算：

```
q_i = (descending_average_rank(y_i) - 1) / (N - 1)
w_i ∝ 2 ** (-q_i / rank_weight_half_life)
L_main = sum(w_i * (prediction_i - target_i)^2) / sum(w_i)
L_gate = (mean(g_i) - target_pass_rate)^2
L = L_main + gate_loss_weight * L_gate
```

较高收益对应较大权重；同收益同权重。权重按截面均值归一化，并在 CPU 上每窗口
预计算一次。`rank_weight_half_life=0.2` 表示每下降 20 个排名百分点权重减半，
最佳/最差约 32:1。研报只说“根据收益率衰减加权”，未披露公式，
本公式及半衰减参数为用户选择加权方向后的工程实现，不声称是研报原公式。
`loss="mse"` 可关闭排名权重。门控采用研报单门参数：通过率目标 0.4、损失系数 10；
0.4 是软约束目标，不是每个截面硬选 40% 股票。

使用普通排名的截面 z-score 标签；排序后再 z-score 不改变收益高低顺序，
因此权重可直接从训练标签计算。只在训练和验证中使用标签，预测不需要未来收益。

## 预处理和训练

公共 transform `mad5_zscore_rank_label`：特征按日期用 median ± 5×MAD 缩尾，
再 z-score，最后缺失填 0；不做行业/市值中性化。MAD=0 时缩尾区间收缩到中位数。
标签缺失保持缺失，交给公共数据流程剔除；标签采用截面普通排名再 z-score，
不使用高斯逆变换。所有计算只使用该日截面，训练/预测采用相同特征处理。

用户确认采用项目口径：全部可用正式因子、10 日 VWAP 标签、年度重训、
近 36 个月中前 33 个月训练和后 3 个月验证、5 交易日采样。
按现有因子合成配置的 2016 起点设置：首个名义窗口为 2016-01 至 2018-09 训练，
2018-10 至 2018-12 验证，2019 年预测；截止 2026-04-24。
实际标签截断复用公共层 T+11 到期和训练/验证隔离规则。
`__all__` 沿用公共筛选，剔除 deprecated/model_generated 因子。
股票池沿用 mkt_all、ST/BJ 过滤；不额外复现研报的市值/成交额前 60% 和上市天数筛选。

Adam lr=0.001、patience=20、单种子 42。研报的“batch size 32、每个交易日全体股票
视作 batch”本实现解释为逐日计算 BN/损失，累计 32 个日期的等权梯度再更新一次；
每次仅一个截面进入 GPU。剩余不足 32 日按实际日期数平均。
验证按完整日期计算加权 MSE 和门控惩罚，再对日期等权平均，以总验证损失早停。
验证/预测 GPU 前向按 3000 行切块，整套训练矩阵留在 CPU，不新增磁盘缓存。

研报未明确的工程参数：最多 100 轮、LeakyReLU slope=0.01、Gumbel 温度=1.0、
确定性预测门、排名半衰减=0.2。均在配置或本文中明确；非精确收益复现。

## 产物与 CLI

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
$env:PYTHONPATH = "$PWD\ml_alpha"
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000023.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000023.toml
```

权重、训练历史与门通过率诊断：`data/model_workspace/mdl_000023/artifacts`。
信号根目录：`data/models`，按现有模型输出规范写入 `mdl_000023` 列。
不会写入正式端到端因子库。权重保存因子列顺序，预测列不一致直接报错。
