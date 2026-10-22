# DFQ Neural ODE / 本地因子序列版

`mdl_000032` 参考东方证券 2025-05-27《因子选股系列之一一六：Neural ODE，
时序动力系统重构下深度学习因子挖掘模型》第 6–9 页，归入 alpha 因子合成模型，
券商 tag 为 `DFZQ`。用户确认首版采用 Neural ODE 和普通 MLP 收益头；
不包括扩散项、泊松跳跃项、ABCM 风险/alpha 双头及与基准模型的分数合成。
报告第 11 页 Model1/Model2 为与基准模型合成后的结果，不是本模型的绩效依据。

## Architecture / 网络

```text
N stocks x 30 historical sessions x F formal factors
  -> two-layer GRU, hidden=128, last hidden state
  -> separate Linear heads: mu and logvar, latent=64
  -> train: z0 = mu + exp(logvar/2) * epsilon, epsilon ~ N(0,I)
     validate/predict: z0 = mu
  -> dz/dt = MLP(concat(z,t)), shared over all time steps
       Linear(65,128) -> Tanh -> Linear(128,128) -> Tanh -> Linear(128,64)
  -> RK4 on 30 observation times from 0 to 1
       Linear(64,F): reconstructed mean at each observed date
       Linear(64,F): reconstructed log variance, for training likelihood
  -> terminal reconstructed mean -> Linear(F,128) -> Tanh -> Linear(128,1)
  -> one scalar score per stock
```

时间网格 `t_i=i/(L-1)`，`L=30`。相邻观测之间默认一个 RK4 步长，
`ode_substeps` 可增加每段的等分步数；不是对自适应误差容限的设置。
所有阶段使用相同动力函数参数，输入包含当前状态和积分时间。
初值是整个已观测窗口经 GRU 编码得到的潜变量，不是首日原始因子向量；
ODE 在潜空间演化，由观测头映射回 F 维。GRU 按时间正序处理。
收益头读取最后一个重构观测，不直接读取 GRU 隐状态或原始末日输入。

整个积分区间对应已观测的历史 30 日，没有向未来 10 日推进或读取未来因子。
这是历史序列的联合重构，早期重构点允许使用同一窗口较晚的已观测信息；
它不是逐历史时点的在线预测。最终收益分数的输入最晚为样本日当日。

采用离散 RK4 计算图自动微分，不使用报告讨论的连续伴随法。
纯 PyTorch 实现，无需 `torchdiffeq`、`torchsde` 或新增数值求解依赖。
验证/预测使用 `mu`，不采样噪声，不计算重构方差/损失，不调用未来标签；
这是一条确定性均值初值轨迹，不是非线性模型的 Monte Carlo 期望收益。
浮点计算仍可能存在设备/库层面的微小差异，不承诺跨设备逐位一致。

## Objective and report alignment / 损失与研报差异

```text
nll = mean_{time,factor} 0.5 * [log(2*pi) + logvar_x
                               + (x - mean_x)^2 * exp(-logvar_x)]
kl  = 0.5 * sum_{latent} [mu^2 + exp(logvar_z) - 1 - logvar_z]
loss = mean_{stocks} [(score - zscore(return))^2 + 1.0*nll + 0.001*kl]
```

重构是对角高斯似然，包含预测 log variance 项；后验 KL 对标准正态 `N(0,I)`。
NLL 在时间和输入因子维求均值，KL 在潜变量维求和，然后两者均按股票求均值。
这样增大因子数量不会线性放大重构项；更改潜变量维度会改变 KL 规模。
训练每个样本每次前向采一次初值噪声，ODE 轨迹本身无布朗噪声。
仅有效标签股票参与这三个训练损失，标签缺失行不用于无监督额外训练。

报告第 9 页将 log likelihood 以正号写入总损失，且显示的高斯公式未包含完整的
log variance 项。本实现按最小化目标采用完整高斯**负**对数似然，避免直接按
显示公式实现时出现优化方向相反或无限放大方差的问题。
稳定性边界为后验 log variance 截到 [-10,10]，观测 log variance 截到 [-8,8]；
该边界适用于 log variance，不是收益预测范围。

GRU 具体结构、隐变量维度、时间网格、RK4、观测映射、Tanh、方差边界、优化器
及权重都是本地工程设定，研报未完整披露。已确认重构权重 1、KL 权重 0.001。
原研报收益标签有中性化，本版本按用户确认只做日期截面标准化，不额外中性化。
保留变分编码、动力系统解码、历史重构、KL 和收益监督这一组合结构，
不宣称复现报告全部数据、参数或组合结果。

## Data, training and resources / 数据与资源

- 全部有效正式因子，公共自动发现排除 deprecated 和 model-generated 因子。
  采用 `factor_sequence` 与 `history_transform="zscore_clip3"`：逐历史日期、
  按当日股票池和过滤条件做截面 z-score，然后 clip 到 [-3,3]、缺失填 0。
  股票需有连续完整 30 日历史。输入处理先于未来标签合并。
- 标签为 `future_vwap_return_10d`，每个样本日期有效标签股票做收益 z-score，
  不排序；常数标签日期跳过。预测返回无固定范围的标量分数，不是收益百分比。
- 年度重训，过去 36 个月按前 33 月训练、后 3 月验证，每 5 交易日抽取样本日；
  样本内部仍包含连续 30 日。配置训练范围 20160101–20251231，预测范围
  20190101–20260424；实际可用窗口由数据覆盖与公共窗口构造决定。
- 10 日 VWAP 使用 t+1 至 t+11，公共 `lookahead_days=11` 控制 T 日训练的
  最晚截面不晚于 T−11，并清除训练/验证交界标签重叠。不另写模型日期切分，
  不需要 FactorGCL 的未来特征辅助监督配置。首个样本需额外 29 交易日历史。
- Adam，lr=0.0001，最多 200 epochs，每轮无放回随机抽最多 100 个日期、按日期
  顺序执行；每日期完整梯度累积后更新一次。batch_size=3000 不变。
  股票块所有损失均除以完整日期有效股票数，末尾小块不会被等权放大。
- seed=42 单种子；最近最多 5 轮原始参数平均，用平均参数计算验证平均日
  Pearson IC，验证后恢复原始训练参数。连续 20 轮无提升早停，min_delta=1e-8；
  最终保存最佳验证 IC 对应的平均参数，无额外调参或多种子集成。
- `checkpoint_ode=true` 在反向传播重算各段 RK4 和重构似然，减少保存的激活；
  代价是额外计算。逐观测累计重构损失，不持有完整重构张量。
  关闭 checkpoint 会增加显存；增加 ode_substeps 会增加计算量。
- CUDA 显式要求可用 GPU，不自动回退 CPU。CPU 历史数据缓存预算 512 MiB，
  按需组装股票块，不写磁盘预处理缓存或 C 盘文件。输入块约为 `4*B*30*F` 字节，
  GRU、激活、索引、优化器等另占内存，512 MiB 不是进程总内存上限。
  每个前向默认 29×4 次动力函数调用，不能承诺比普通 GRU 更快。

## Artifacts and CLI / 产物与命令

- `data/model_workspace/mdl_000032/artifacts/<window_id>/model.pkl`：PyTorch checkpoint，
  包含参数、特征顺序、最佳权重及诊断记录。
- 同目录 `model.manifest.json`、`model_info.json`、`loss_history.parquet`：公共指纹，
  结构与求解器口径，每轮总损失、收益 MSE、重构 NLL、KL、验证 IC 和早停记录。
- `data/models/<YYYY>/<YYYYMMDD>.parquet`：公共写入器按日期、股票键合并
  `mdl_000032` 分数列。无需新增 Rust 派生数据。

仓库根目录运行（本次未执行训练或预测）：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000032.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000032.toml
# Combined: model-run --config ml_alpha\models\mdl_000032.toml
```

xz336 准备好数据与 PyTorch CUDA 环境后：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000032.toml
```

本次仅静态检查，不运行测试、训练、预测或派生数据生成。
