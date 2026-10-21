# DFQ-FactorVAE / mdl_000013

参考东方证券 2024-05-14《因子选股系列之一〇三：DFQ-FactorVAE：融合变分自编码器
和概率动态因子模型的 alpha 预测方案》，架构依据第 7–11 页图 2、5、8、11，
参数参考第 13 页。这里实现用户确认的本地因子输入版本，归入模型层，
不创建正式端到端因子，不需要新增 Rust 派生数据。

## 输入与本地口径

- `models/mdl_000013.toml`，输入为 `data/factors/stock/daily` 的有效正式因子。
  自动发现复用公共层规则：排除 metadata 标记为 `model_generated`、`deprecated`
  的列及自身输出列；metadata 应与本地因子库保持同步。
- 每个样本使用包含当日在内的 **30 个连续交易日**，目标截面每 5 个交易日采样。
  不是只读取 30 个五日采样点。历史窗口内持续满足股票池及过滤条件的股票才进入截面。
- 输入截面按公共 `FactorHistory` 做 median ± 5 MAD 缩尾、zscore、缺失填 0；
  整日文件缺失报错，历史长度不足的日期不产生样本。
- 股票池 `mkt_all`，剔除 ST、BJ，不剔除涨跌停。
- 标签 `future_vwap_return_10d`，即复权 VWAP(t+11) / VWAP(t+1) - 1。
  标签按日 5 MAD 缩尾、zscore；缺失标签不填成收益 0，仅从后验组合收益和损失中排除。
  特征标准化、股票池、先验注意力不由标签是否缺失决定。
- 因子输入没有研报的 46/14 基础特征分组，采用用户确认的**单路 GRU**。
  不强行按列位置拆成双路，不宣称复现研报的私有特征集。
- 单种子 42、单个 10 日标签，不做原研报六标签或多种子集成。

## 网络与训练目标

```text
N stocks × 30 sessions × F factors
        LayerNorm(F) → Linear(F,32) → LeakyReLU → 1-layer GRU(32)
                          ↓ last hidden state e [N,32]
 training only: e + observed y → 100 virtual portfolios → posterior [48 means/stds]
 train/infer:   e → 48 global-attention heads             → prior [48 means/stds]
               e → alpha mean/std [N] and beta [N,48]
                          ↓ decoder
                  stock-return mean/std [N]
```

组合层沿**股票轴** softmax，每个组合的权重和为 1。100 个组合收益映射为 48 个
独立高斯后验因子的均值和标准差；标准差使用 Softplus 加数值下限。

先验使用每头独立的 key/value 线性层及可学习 query。按图 11 取
`max(cosine(query, key_i), 0)`，除以全截面正相似度之和，汇总 value。
没有正相似度的注意力头输出零向量，避免零除；不替换为 softmax。
48 个头的输出堆为 `[48,32]`，通过共享的 Linear/LeakyReLU 和均值、标准差头
分别映射为每个公共因子的分布。公共因子在日期截面内共享，不是每股 48 列 alpha 等权输出。

解码器遵循 `y_i = alpha_i + beta_i @ z`：

```text
mu_y[i]  = mu_alpha[i] + sum_k beta[i,k] * mu_z[k]
var_y[i] = sigma_alpha[i]^2 + sum_k beta[i,k]^2 * sigma_z[k]^2
```

训练使用后验分布解码，对股票收益做重参数化采样 `mu_y + sigma_y * epsilon`，
`epsilon ~ N(0,1)` 每次前向重采样。损失为股票平均重构 MSE 加
`kl_weight * KL(q_posterior || p_prior)`，对 48 个高斯维度**求和**。
默认 `kl_weight=1`，这是用户确认的工程口径，研报未给出该系数。
特征提取、编码器、预测器、解码器联合训练，不接 XGBoost 或 Optuna。

验证与预测只走 `e → prior → decoder`，返回 `mu_y`，不调用后验，不传标签，不采样。
输出是标准化标签空间中的相对收益分数，并非可直接解释为百分比的绝对收益。
概率分布标准差仍参与训练。这一确定性输出口径经用户确认，不将研报图 9 的实验结果
直接等同于本地实现；数值仍可能受 GPU 底层算子非确定性影响。

## 训练窗口、资源和产物

- 每年年末重训，回看 36 个月，前 33 个月训练、后 3 个月验证，每 5 个交易日采样。
  配置训练边界为 2016-01-01～2025-12-31，预测范围 2019-01-01～2026-04-24。
  公共 calendar/runtime 决定实际交易日和窗口，不在模型里复制日期规则。
- 公共标签到期规则和训练/验证隔离继续生效：训练日 T 的最后可用标签截面不晚于 T-11；
  采样可能让实际最后日期更早，缺少 30 日历史的起始样本也会跳过。
- 一个优化步骤是**一个交易日的完整截面**。`batch_size=3000` 仅限制每次 GRU 编码的
  股票数；通过梯度检查点控制编码激活显存，最后合并所有编码再计算组合和注意力。
  不能把股票块当成彼此独立的市场来估计公共因子。
- 200 轮上限，每轮最多随机选 100 个日期，按时间顺序读取；Adam lr=0.0003，
  无权重衰减，梯度范数裁剪 5。最近 5 轮权重平均用于验证和最优模型保存，
  验证结束后恢复优化器对应的实时权重，不改变继续训练的轨迹。
- 验证指标为每日 Pearson IC 的平均值，连续 20 轮没有超过 `min_delta=1e-8` 的提升早停。
  日志分别记录重构 MSE、KL、总损失、验证 IC 和耗时。
- `device=auto` 有 CUDA 时使用 GPU，否则使用 CPU，可显式改为 `cuda` 强制 GPU。
- 复用公共惰性序列读取和有界 RAM 缓存，默认预算 512 MiB；该预算只约束日度特征缓存，
  不包含股票索引、当前编码块、训练标签表、网络和优化器。不会落盘完整三年序列，
  也不会写 C 盘缓存。分块 CPU 输入会被检查点保留至当日反向结束，不能将缓存预算
  当作整个进程的内存上限。
- 权重：`data/model_workspace/mdl_000013/artifacts/<window_id>/model.pkl`，
  实际使用 torch checkpoint；同目录有 `model.manifest.json`、
  `loss_history.parquet`、`model_info.json`。checkpoint 保存列顺序、参数及结构版本。
- 分数：`data/models/<year>/<YYYYMMDD>.parquet` 中的 `mdl_000013` 列，沿用公共日度宽表
  与 `trade_date`、`ts_code` 键，其他模型列保留。

## CLI

在仓库根目录使用现有 Python 3.8.3 环境。以下命令是正式入口，本次接入未执行训练或预测。

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH

& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000013.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000013.toml

# 或一次完成训练和预测；不要在上面两条之后重复执行。
& $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000013.toml
```
