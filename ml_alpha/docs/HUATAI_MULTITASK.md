# Huatai AI 67 / 多任务因子合成

依据华泰研究 2023-05-06《AI 模型如何一箭多雕：多任务学习》（人工智能系列之 67）
第 4–5 页损失公式、第 7 页图表 8–9、第 8 页图表 10。
输入是已构建的单日因子表，归入模型层；券商 tag 为 `HTZQ`。

| Config | Loss weighting | Output |
| --- | --- | --- |
| `models/mdl_000033.toml` | Uncertainty weighting (UW) | mean(10-day head, 20-day head) |
| `models/mdl_000034.toml` | Dynamic weight average (DWA) | mean(10-day head, 20-day head) |

两份配置共用 `MultiTaskAlphaModel`，分别训练独立权重，不互相融合，不新增正式端到端因子。
没有重复实现报告的单任务 MLP 对照组，也不实现概述章节中的 MMoE、cross-stitch 等扩展。

## Network / 网络

```text
T-date N stocks x F local formal factors
  -> Linear(F,256) -> LeakyReLU
  -> Linear(256,256) -> LeakyReLU          shared layers
  -> two independent task heads:
       Linear(256,256) -> LeakyReLU -> Linear(256,1)
       10-day rank return             20-day rank return
  -> arithmetic mean of both scalar predictions
```

第三个隐藏层及输出层各任务独立，前两层共享。LeakyReLU negative_slope=0.01，
没有 BatchNorm、Dropout、序列编码或输出激活。每支股票独立前向，任务间通过共享参数交互。
最终分数无固定数值范围，用于排序，不解释为实际收益百分比。
UW/DWA 是训练损失的任务权重，不用于最终双头预测加权；预测严格按算术均值合成。

## Input and targets / 输入与目标

用户确认输入换为本地全部有效正式因子，自动排除 deprecated 和 model-generated 列。
`factor_frame` 只用样本日 T 的截面，不增加历史序列输入或 Rust 派生数据。
股票池为 `mkt_all`，沿用项目 ST/北交所过滤，不增加研报的季度流动性/市值前60%筛选。
新配置训练起点为 2016 年，输入晚于公共 ST 筛选起点。

公共层先对当日股票池的每个输入因子执行 5 倍 MAD 缩尾、截面 z-score、缺失填0，
不做中性化；之后剔除任一任务标签缺失或非有限的股票。配置使用
`mad5_zscore_features` 保留原始收益标签，模型在两个任务共同有效的股票中分别
计算收益排名并 z-score。少于两只股票或任一任务收益恒定的日期跳过并计入诊断。
输入预处理不依赖未来标签可用性；标签排名/权重只参与训练与验证，不用于预测。

两个标签共用 `data/label/stock/daily`：

```toml
[label]
id = "future_vwap_return_10d"
lookahead_days = 11

[[label.auxiliary]]
id = "future_vwap_return_20d"
lookahead_days = 21
```

本地标签元数据已登记这两项，实际训练仍需要所选区间每个标签的数据文件。
报告使用 close(t+11)/close(t+1)-1 和 close(t+21)/close(t+1)-1；项目替换为
对应复权 VWAP 收益。主标签列仍保留旧接口；`label_columns` 按主标签、辅助标签顺序传入模型。

每任务日期截面的样本权重按原始收益降序排名计算：

```text
p_i = (descending_average_rank_i - 1) / (N - 1)
w_i = 2 ** (-(p_i - min(p)) / 0.5)
w_i = w_i / mean(w)
L_k = mean_i [w_i,k * (prediction_i,k - zscore(rank(return_i,k))) ** 2]
```

半衰期 0.5 表示下降50个百分点时权重减半，权重在每日期每任务独立归一化。
并列收益用平均排名。复用已有收益排名权重实现，不改变其他模型的半衰期。
报告未披露完整衰减公式，0.5 为用户确认的本地工程口径。

## UW and DWA / 任务加权

UW 用可学习参数 `s_k=log(sigma_k^2)`，初值0（sigma=1），最小化：

```text
L_UW = sum_k [0.5 * exp(-s_k) * L_k + 0.5 * s_k]
```

此式对应报告 `L_k/(2*sigma_k^2) + log(sigma_k)`。方差参数纳入 Adam 和 checkpoint，
没有将它设为固定超参，也没有用总损失代替各任务损失。
UW 总目标可能为负，因为含 log sigma 项；单任务 WMSE 仍非负，日志分别记录。

DWA 在第 e 轮使用前两轮训练任务损失（尚未乘任务权重）的日期等权均值：

```text
r_k(e) = train_L_k(e-1) / max(train_L_k(e-2), 1e-12)
lambda_k(e) = K * softmax_k(r(e) / temperature)
L_DWA = sum_k lambda_k(e) * L_k
```

K=2，temperature=2；前两轮 lambda=[1,1]。数值上先减最大 logit 再指数化。
历史单位为 epoch，不是股票块或 optimizer step。不使用验证损失更新 DWA。
这些初始化、温度和零分母保护为显式工程设定。

## Training, safety and resources / 训练与资源

年度重训，过去36个月按前33个月训练、后3个月验证，5交易日采样，单种子42。
配置训练范围 20160101–20251231，预测范围 20190101–20260424。
实际窗口由交易日历和公共成熟规则确定。最长20日标签在t+21到期，因此 T 日训练
最后可用截面不晚于 **T−21**，训练/验证交界也按最长21日清除标签重叠。
公共层自动计算所有声明标签及已有辅助监督的最大期限，不靠模型内部手写日期偏移。

Adam lr=0.0001，weight_decay=0，最多200轮，早停20轮。
验证分数为两个任务 WMSE 的等权均值，再按日期等权平均，不使用可学习UW权重或
动态DWA权重作为早停评分，避免任务权重变化使评分不可比。保存最优验证评分权重，
不做额外参数平均、调参或多种子集成。

每交易日作为完整优化 batch。GPU股票计算块上限3000，所有块梯度累积后才更新一次。
每块平方误差除以完整日期股票数；样本权重已归一到均值1。
UW正则按块股票占比计入，整日合计恰好一次，不随分块数量放大。
网络无跨股票统计或随机层，因此分块保持完整日期目标，仅有浮点累积顺序差异。
CUDA 显式要求可用 GPU，不自动回退 CPU。

公共读取保留按日期进度、行数和耗时日志。输入表和float32矩阵驻留CPU，
GPU只接收当前股票块；两份模型应顺序训练。`cache_samples=false`，无磁盘预处理缓存，
不向 C 盘写缓存，不预先物化跨日期窗口矩阵。大规模因子表及其float32副本仍占CPU内存，
股票计算块大小不是进程总内存上限。

## Public API compatibility / 公共接口兼容

- `LabelConfig.auxiliary` 为可选辅助标签声明，共用主标签 root；拒绝重复ID和非法期限。
- 普通表、旧序列和bar数据路径统一读取、预处理多标签，完整向量缺失过滤在特征处理后执行。
  要求完整截面的图模型继续保留未标记节点，由图模型自行掩码监督。
- `ModelContext.label_column` 保留主标签，新增 `label_columns` 表示有序全部任务。
- 预测只读特征，不依赖标签文件；配置中标签列表仅用于模型/manifest身份校验。
- 多标签声明进入模型指纹；未声明辅助标签时不增加空字段，原单标签指纹保持兼容。

## Artifacts and CLI / 产物与命令

权重：`data/model_workspace/<model_id>/artifacts/<window_id>/model.pkl`，包含两个输出头、
共享层、UW方差参数（如适用）、特征/标签顺序及诊断。
同目录公共manifest覆盖两项标签和期限；`loss_history.parquet` 按任务记录训练/验证
WMSE和任务权重，另记录总训练损失、早停评分和耗时。
预测合并写入 `data/models/<YYYY>/<YYYYMMDD>.parquet` 的 `mdl_000033` / `mdl_000034` 列。

仓库根目录运行（本次未执行）：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000033.toml
& $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000034.toml
# Separate: model-train / model-predict with the same --config.
```

服务器准备好数据及PyTorch CUDA环境后：

```bash
cd /home/wym/YuminQuant/YuminQuant2026
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000033.toml
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000034.toml
```

本次仅静态检查，不运行测试、训练、预测或派生数据生成。
