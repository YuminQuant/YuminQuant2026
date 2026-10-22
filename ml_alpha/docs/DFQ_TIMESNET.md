# DFQ-TimesNet / 本地因子序列版

`mdl_000030` 基于东方证券《因子选股系列之一一八：DFQ-TimesNet，捕捉量价特征
周期规律，提升股票收益预测效果》（2026-04-16）第 9–17 页的网络与训练说明。
本版本由用户选择使用全部有效正式因子的历史序列，归入 **alpha 因子合成模型**，
tag 为 `DFZQ`；没有向端到端因子目录新增同名配置。

## Architecture / 网络

```text
N stocks x 60 sessions x F factors
  -> TokenEmbedding: Conv1d(F, 128, kernel=3, zero padding=1)
  -> one TimesBlock:
       period=5:  [N, 128, 12, 5]
       period=60: [N, 128, 1, 60]
       each period -> Inception(128->128) -> GELU -> Inception(128->128)
       unfold each period back to [N, 60, 128]
       arithmetic mean of the two periods + input residual
  -> take the last observed time step
  -> Linear(128, 1) -> one score per stock
```

每个 Inception 分别运行 1×1、3×3、5×5 二维卷积，再将三个输出等权平均。
两个周期共享同一 TimesBlock 内的卷积参数；前后两层 Inception 各有独立参数。
不同 TimesBlock（若增加层数）也各自初始化。卷积核用 Kaiming 初始化。
通道始终表示隐藏特征；二维行表示跨周期位置，列表示周期内位置。

按报告最终方案采用固定 [5,60] 周期、周期等权融合、残差、最后时间步读出；
不使用 FFT 动态选周期、FFT 权重、日历嵌入、位置编码、时序均值池化或额外 GRU。
报告 dropout=0，当前模型不引入 Dropout、BatchNorm、LayerNorm 或时间归一化。
60 日输入正好整除两个周期；通用路径在末尾补零到周期整数倍，卷积后裁掉补齐部分。
TokenEmbedding 的零边界、无 bias 以及卷积权重跨周期共享为显式实现口径；
没有把观察窗口的首尾行情当成相邻交易日进行循环填充。
卷积可以读取窗口内相邻位置，但输入窗口最晚只到样本日 T，不含 T+1 数据。

## Input and target / 输入与标签

- 使用 `factor_sequence`，有效正式因子自动发现；沿用公共 deprecated/model-generated
  排除规则。输入是连续 60 个交易日，不是每 5 日采样后的 60 个截面。
  5 日采样只决定训练样本的结束日期。
- 每个历史日期先按当日 `mkt_all`/ST/北交所筛选，然后分别对每个因子做截面
  z-score（ddof=0），再 clip 到 [-3,3]，缺失或非法值填 0。零方差列输出 0。
  不做 MAD 截尾、不按股票的时间轴归一化，不用训练窗口总体均值标准差。
- 公共 `FactorHistory` 新增可选 `history_transform="zscore_clip3"`；其他配置未指定时
  仍走原 MAD5/zscore 路径，旧默认数据指纹保持不变。预处理选择写入数据身份，
  更换预处理会触发 manifest 不匹配，不能混用旧模型。
- 历史截面预处理先于标签合并，未来标签缺失不影响输入归一化。
  股票需在全部 60 日历史中存在且通过相应日期筛选，沿用公共序列可用性规则；
  历史缺失日期报错，不把其他日期冒充完整序列。
- 标签使用项目 `future_vwap_return_10d`，在每个训练/验证日期的有效标签股票中
  做普通收益 z-score，不排序、不 clip。缺失标签不参与损失，常数标签日期跳过。
- 预测仅使用因子序列及权重，返回无固定数值范围的 float32 标量选股分数。
  不读取未来标签，不再次截面归一化，也不将分数解释为收益百分比。

报告原输入为 60 项基础日行情、分钟派生及 Level-2 派生特征，未披露完整清单；
原标签为 20 日收益，采用固定训练/验证与种子 1000。本地替换为正式因子库、
10 日 VWAP 和以下滚动方案。实现保留网络主干，不宣称完整复现报告数据或绩效。
报告表中 `num_latent=60` 对应其输入设定，本版本输入维度由实际 F 推断，不强行
截取 60 个本地因子。无需新增 Rust 派生数据或预先生成窗口矩阵。

## Training and resources / 训练与资源

年度重训，近 36 个月按前 33 月训练、后 3 月验证，5 日采样，单种子 42。
训练日期配置 2016-01-01 至 2025-12-31，预测日期 2019-01-01 至 2026-04-24。
历史特征需要覆盖首个样本日前 59 个交易日，实际可训练窗口取决于数据库覆盖。
公共层 `lookahead_days=11` 保证训练与验证标签在 T 日收盘已到期，末截面不晚于
T−11，并执行训练/验证隔离；模型不另行实现滚动窗口。

报告未明确写出损失函数和优化器，本地经用户确认采用标准化收益 MSE 与 Adam。
lr=0.00009、最多 200 epochs、每轮随机选最多 100 个训练日期（无放回，按日期顺序
处理以复用重叠历史）、验证平均日 Pearson IC 连续 20 轮无提升早停。
无额外权重衰减；min_delta=1e-8 是数值比较容差。

每个日期作为一次完整优化更新。股票分块不超过 3000，每块平方误差求和除以
当日全部有效标签股票数，然后立即反向传播；完整日期结束再 optimizer.step。
由于网络无跨股票统计、Dropout 或注意力，此梯度累积保持完整日期 MSE 目标，
最后不足 3000 的块不会被错误地等权放大。其他模型 batch_size 不变。

每轮结束将最近最多 5 轮原始参数求平均，用平均参数验证并选择最优 checkpoint；
验证后恢复当前原始参数继续优化，不把平均参数与旧优化器状态混用。最终保存最佳
验证 IC 对应的平均参数。它是单模型参数平滑，不是多种子/多模型预测集成。
每天先完整收集验证预测，再计算该日 IC，最后日期等权平均；不计算分块 IC 再平均。

`device="cuda"` 要求 PyTorch CUDA，不自动回退 CPU。CPU 历史缓存由公共序列类
控制在 512 MiB 数据缓存预算，超过预算的历史按需重读；不写任何磁盘预处理缓存。
此预算不是进程总内存上限，DataFrame、代码/位置索引、窗口加载临时数组和模型状态
另占内存。股票分块输入元素约占 `4 * B * 60 * F` 字节，GPU 还需卷积激活、梯度、
优化器工作区。因子维度大于报告 60 维，不能保证其约 4 GiB 显存数字适用于本地。
实现按周期和卷积核累加输出，避免额外 stack 张量；不保存全窗口 N×60×F 矩阵。
`materialize.predict_batch_size=20` 是预测日期批数，模型 batch_size=3000 是股票块数。

## Artifacts and CLI / 产物与命令

- `data/model_workspace/mdl_000030/artifacts/<window_id>/model.pkl`：PyTorch checkpoint，
  含特征列顺序、参数、最佳平均权重与诊断。
- 同目录 `model.manifest.json`、`model_info.json`、`loss_history.parquet`：公共身份校验、
  模型说明、每轮训练 MSE/验证 IC/早停及平滑记录。
- `data/models/<YYYY>/<YYYYMMDD>.parquet`：公共写入器按 `trade_date`、`ts_code`
  合并 `mdl_000030` 分数列，不在正式因子库写中间列。

在仓库根目录运行（本次没有执行以下命令）：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000030.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000030.toml
# Combined: model-run --config ml_alpha\models\mdl_000030.toml
```

服务器准备好数据库与 PyTorch CUDA 环境后：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000030.toml
```

仅静态代码检查，不运行测试、训练、预测或派生数据生产，不写 C 盘缓存。
