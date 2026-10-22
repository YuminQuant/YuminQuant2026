# Ordinal NN / 有序回归因子合成

`mdl_000029` 对应华泰证券《人工智能60：有序回归》（2022-10-11）。
它是使用既有因子的 **alpha 合成模型**，不属于直接输入原始量价的端到端因子。
用户最终选择：NN 基模型、样本加权 sw、10 档 logistic 有序回归、单种子 42；
不实现三个目标集成，也不额外加入 GAT、类别加权 cw 或组合优化器。

## Network and loss / 网络与损失

按报告 p10 图13 的 NN 结构：

```text
F factors -> Linear(F,64) -> Sigmoid -> BatchNorm
          -> Linear(64,64) -> Sigmoid -> BatchNorm
          -> Linear(64,64) -> LeakyReLU -> Linear(64,1)
```

最终输出一个连续分数 `f(x)`，越高表示预测排名越高；不是十个类别概率，
也不是以百分比表示的收益预测。没有多网络、多种子或输出集成。

输入复用公共 `mad5_zscore_rank_label`：每日期每因子 MAD5 截尾、z-score、缺失填 0；
收益标签先截面平均秩排序，再做 z-score。不做额外行业/市值中性化。
设某天合格股票标签为 y，K=10，阈值为当日 y 的 10%、20%、…、90% 分位数：

```text
c[k] = quantile(y, k/K), k=1,...,K-1
s[i,k] = +1 if y[i] >= c[k], else -1
L[i] = sum_k softplus(-s[i,k] * (f(x[i]) - c[k]))
p[i] = (descending_average_rank(y[i]) - 1) / (N - 1)
w[i] = 2 ** (-p[i] / tau)
loss = sum_i w[i] * L[i] / sum_i w[i]
```

这遵循报告 p6 的多阈值 logistic 求和与 p9 的样本半衰加权。
`tau=0.5` 沿用本文引用的《人工智能55：图神经网络选股的进阶之路》p11 的
WMSE 基线半衰期。收益最高者权重 1、中位附近权重 0.5、最低者权重 0.25。
实现先减去最优百分位再求指数，仅改变所有权重的公共倍数，归一化损失不变。
相同收益使用相同平均秩，不按股票代码打破平局；分位数线性插值，等于阈值时归入
上侧（报告 p6）。有效标签少于两只或完全相同的日期跳过并记录计数。

阈值与权重按完整日期截面一次构建，仅在本窗口 CPU 内存中保留，跨 epoch 复用。
验证使用验证日期自身的标签构造验证损失，模型处于 eval 模式，不更新参数或
BatchNorm 统计；训练与验证日期必须完全分开。**预测不读取标签、不计算阈值**，
直接输出神经网络的标量分数。

## Training and resources / 训练与资源

- 本地全部有效正式因子，自动排除 model-generated/deprecated 输入，股票池 `mkt_all`，
  剔除 ST 与北交所，沿用公共层筛选。
- 10 日 VWAP 标签；年度重训，回看 36 个月，前 33 月训练、后 3 月验证，5 日采样。
  配置训练日期 2016-01-01 至 2025-12-31，预测 2019-01-01 至 2026-04-24；
  实际窗口由数据可用性决定。
- 公共窗口 `lookahead_days=11` 处理到期及训练/验证隔离，不在模型里复制切分逻辑。
  T 日收盘训练时，训练与验证可用截面均不得晚于 T−11。
- 按报告 p12，每个完整日期截面作为一个训练 batch，Adam，lr=0.0001，
  验证损失连续 20 轮无改善早停并恢复最佳权重。每天一次优化器更新，各日期损失等权
  汇总用于 epoch 日志。每年从 seed42 重新初始化，不沿用上个年度的权重。
- 最多 100 epochs、LeakyReLU 斜率 0.01、无权重衰减是显式工程默认值，
  不是报告披露的最优值。报告 p10 网络没有 Dropout，本实现不加 Dropout。
- `device="cuda"` 要求可用 PyTorch CUDA，未配置时直接报错，不自动回退 CPU。
  训练矩阵仅构建一次并保留 CPU，GPU 每次只接收一个日期，模型宽度 64；
  不构建 N×N 图，不分配全窗口 GPU 特征矩阵。
- `model.params.predict_batch_size=3000` 为验证/预测股票分块；eval 使用已拟合的
  BatchNorm 运行统计，股票分块不改变截面训练口径。现有其他模型 batch_size 未修改。
  `materialize.predict_batch_size=20` 仍表示公共层每批加载的预测日期数。
- 全窗口 CPU 矩阵元素约占 `4 * (N_train + N_valid) * F` 字节，此外还有输入
  DataFrame、标签、权重、行索引及转换临时副本；不是全流程流式读取。
  不新增磁盘缓存，`cache_samples=false`，不写 C 盘缓存。

报告原输入为 42 个因子、股票池带流动性/市值筛选，训练 6 年＋验证 2 年、半年更新。
本地采用项目因子库、股票池与上述年度窗口，保留其 NN 与 sw 有序损失结构；
不宣称复现报告的收益结果。数据直接使用已有因子库，无需新增 Rust 派生数据。

## Artifacts and CLI / 产物与命令

- `data/model_workspace/mdl_000029/artifacts/<window_id>/model.pkl`：PyTorch checkpoint，
  包含权重、BatchNorm 统计、因子顺序、参数与诊断；由本模型 `load` 恢复。
- 同目录 `model.manifest.json`：公共窗口和配置/特征指纹。
- `loss_history.parquet`、`model_info.json`：逐轮损失、最佳 epoch、跳过日期及实际参数。
- `data/models/<YYYY>/<YYYYMMDD>.parquet`：`trade_date`、`ts_code` 键下的
  `mdl_000029` float32 分数列，使用公共写入器合并。

仓库根目录运行；需已有输入因子、标签和 PyTorch CUDA 环境：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000029.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000029.toml
# Combined: model-run --config ml_alpha\models\mdl_000029.toml
```

服务器在已配置好数据与 PyTorch CUDA 的环境中使用同一份配置：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000029.toml
```

本次仅接入代码与配置，不安装依赖、不执行测试或训练；实际运行与性能尚未验证。
