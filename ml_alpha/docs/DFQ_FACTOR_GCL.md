# DFQ-FactorGCL / 本地因子库版

`mdl_000031` 对应东方证券《因子选股系列之一一七：DFQ-FactorGCL：基于超图卷积
神经网络和时间残差对比学习的股票收益预测模型》。实现参照报告 pp19–23 超图、
pp29–32 对比损失、pp33–35 完整 FactorGCL 结构与 pp38–39 数据/参数。
本地输入采用有效正式因子，归入 alpha 因子合成模型，券商 tag 为 `DFZQ`。

## Network / 模型

```text
Historical factors [t-29,...,t] -> historical GRU -> X0
  prior = HyperGCN(X0, H_prior)
  X1 = X0 - prior
  H_hidden = sigmoid(X1 @ learned_prototypes.T)
  hidden = HyperGCN(X1, H_hidden)
  alpha_past = LeakyReLU(Linear(X1 - hidden))
  score = Linear(concat(prior, hidden, alpha_past))

TRAINING ONLY:
Future factors [t+1,...,t+10] -> independent future GRU -> X0_future
  -> same prior/hidden/alpha modules, using HISTORICAL H_prior and H_hidden
  -> alpha_future -> temporal contrastive supervision
```

两个 GRU 独立，均为 2 层、128 隐藏维，使用最后层末时刻隐藏状态。
预定义/隐藏超图和个体 alpha 模块在两条路径间共享网络参数；两类超图矩阵由历史
路径决定，未来路径不得重新估计隐藏暴露。隐藏原型为 48×128 的可学习参数。
三个预测分量对应独立线性权重，合并为一个 Linear(384,1)，输出每只股票一个分数。
没有额外 HIST 概念注意力、TopK 邻居、GAT 或多种子集成。

超图传播使用报告的对称归一化：

```text
HyperGCN(X,H) = LeakyReLU(Dv^(-1/2) H De^(-1) H.T Dv^(-1/2) X Theta)
```

超边权重固定为单位阵；节点度与超边度按关联权重求和。预定义 0/1 图用稀疏索引
进行股票→概念→股票传播，隐藏图使用 N×48 矩阵；均不构造 N×N 传播矩阵。
无预定义概念的股票该分量为 0，继续通过隐藏及个体分支预测。

## Inputs and concepts / 数据

- 本地正式因子的过去 30 个连续交易日；5 日采样只控制样本结束日期。
- 每个历史日期按当日股票池/筛选做截面 z-score，clip[-3,3]，缺失因子值填 0。
  输入归一化先于未来标签合并，不能用标签可用性决定图节点或截面统计。
- 采用 `concept_factor_sequence`，保留所有具有完整历史的当日合格股票，包括标签缺失
  股票；标签只决定 MSE 掩码。股票池 `mkt_all`，剔除 ST、北交所。
- 收益标签 `future_vwap_return_10d`，按日期将有效标签取平均秩百分位，再 z-score；
  不是 TimesNet 的原收益 z-score。标签少于两只或常数日期不训练。
- 预定义概念复用 `data/derived/stock/hist_concepts` 的主营业务 I 分类。按样本日
  `available_date` 读取 0/1 关联，不用市值加权，不引入其他股票之间的相似度修正。
  继续沿用既有派生口径：年报次年 5 月 1 日、半年报当年 9 月 1 日生效；不使用
  未来主营业务快照回填历史。公共读取器校验概念日期及去重。

报告的 60 项基础日线、分钟及 Level-2 特征未完整披露；本地不假称复现这些输入。
本地 10 日标签、年度滚动及种子 42 替代报告固定训练、20 日标签/多期限实验和
种子 1000。不额外生成风险因子解释图或指数增强优化器。

## Contrastive objective / 对比损失

两条路径共用 Linear(128,128)→LeakyReLU→Linear(128,128) 投影头 p。
按报告 pp31–32，时间项与截面项采用不同相似度公式：

```text
s_ij = cosine(p(alpha_past_i), p(alpha_future_j))
L_temporal = mean_i [logsumexp_j(s_ij^2 / tau) - sign(s_ii)*s_ii^2 / tau]
c_ij = cosine(p(alpha_past_i), p(alpha_past_j))
L_cross = mean_i [logsumexp_j(c_ij / tau) - c_ii / tau]
L = mean_labeled_stocks((score - standardized_rank_return)^2)
    + gamma * (L_temporal + L_cross)
```

tau=0.25，gamma=1。分母包含正样本本身；时间项分子保留正样本方向，分母使用
平方余弦；截面项遵循报告展示的普通余弦公式，不擅自替换为平方。
报告用求和记号描述截面汇总，本地经用户确认三个目标各自按有效股票数取均值，
避免股票数量改变 gamma 的含义。截面对比覆盖完整历史图，包含无收益标签节点。

未来序列不完整的股票仍保留在历史图和有效 MSE 中，只不参与时间对比。
未来分支取历史两张关联矩阵中未来完整股票的对应行，不重新估计关联值；
其度归一化在该子图重新计算。时间正负样本均来自同一批有完整未来序列的股票。
少于两只时该日时间项为 0，仍做 MSE 和截面对比，并记录无配对日期数。
缺失整日输入文件报错，不补造数据。日志记录每轮配对数、未来缺失股票数和无配对日期数。

## Information boundaries / 防泄漏

训练 t 样本的未来辅助输入为 t+1 至 t+10，严格正序且排除 t。
配置显式声明 `features.params.supervision_lookahead_days=10`；公共窗口统一使用
`max(label.lookahead_days, supervision_lookahead_days)`，本配置为 max(11,10)=11。
T 日收盘训练时训练/验证样本末日不晚于 T−11，并按同一最大跨度隔离训练与验证。
若以后延长未来输入，必须同时提高声明；否则模型在训练开始前报错。

公共 bundle 传递 `information_cutoff` 与辅助跨度。模型在创建未来读取器前再检查
每个训练样本的未来末日不晚于训练截止日、严格早于首个验证日期；没有截止日或
跨度声明的直接调用拒绝训练。辅助跨度及政策写入公共 manifest。
已有模型没有声明时默认跨度 0，旧窗口和旧默认 manifest 保持原样。

验证仅用历史分支预测并与验证标签计算 IC，不读取未来序列、不优化对比损失。
预测仅用截至预测日的因子和概念，完全不读取标签、不创建未来读取器。
未来编码器保存在 checkpoint 中以完整记录训练模型，但预测不会调用它。

## Training and resource usage / 训练和资源

- 年度重训，过去 36 个月按前 33 月训练、后 3 月验证，5 日采样、单种子 42。
  配置训练范围 2016-01-01 至 2025-12-31，预测 2019-01-01 至 2026-04-24。
  训练首个样本需额外具备前 29 个交易日特征，实际窗口受数据覆盖限制。
- Adam，lr=0.00009，最多 200 epochs；每轮最多随机抽取 100 个日期，再按日期
  顺序处理。一个完整日期做一次优化器更新。
- 最近最多 5 轮原始参数平均用于验证；按平均日 Pearson IC 早停 20 轮并恢复最佳
  平均权重。验证后恢复原参数继续优化，保持优化器状态一致。
- GRU 股票编码块上限 3000，保持现有内存偏好；全日隐藏状态再参加完整超图传播。
  GRU 激活 checkpointing 在反向阶段重算，原始输入留在 CPU。
- 对比损失 query 块上限 512，每块仍对全体 key 归一化；checkpointing 避免同时
  保存完整 N×N 相似度激活，计算量仍为 O(N²d)，不能视作线性复杂度。
- 历史日快照 RAM 缓存 384 MiB，未来快照独立预算 128 MiB，仅训练期间存在并在
  finally 中释放；无磁盘缓存、无 C 盘缓存。总进程内存还包括 DataFrame、历史/未来
  CPU 输入块、图与索引、5 份参数状态，512 MiB 不是总内存上限。
- `device="cuda"` 要求 PyTorch CUDA，不自动回退 CPU。无 Dropout、无权重衰减，
  LeakyReLU=0.01、投影头宽度128、GRU 默认初始化是明确工程口径，未声称均为
  研报公开的最优超参。输入 F 可能显著多于研报60维，实际性能未测量。

## Artifacts and commands / 产物与命令

- `data/model_workspace/mdl_000031/artifacts/<window_id>/model.pkl`：两个编码器、共享
  超图/投影模块、原型、预测头、列顺序、参数和诊断。
- 同目录 `model.manifest.json`、`model_info.json`、`loss_history.parquet`：窗口身份、
  最晚未来读取日期、最佳 IC、各损失和缺失配对统计。
- `data/models/<YYYY>/<YYYYMMDD>.parquet`：按 `trade_date`/`ts_code` 合并
  `mdl_000031` float32 分数列，分数越大预测收益排名越高，不是收益百分比。

本次检查时，本地和 xz336 均尚未发现概念派生库的 `manifest.json`。训练前需先准备已有 Rust
接口对应的数据，或者同步兼容的派生库；本次没有启动生成：

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-hist-concepts `
  --asset stock --start-date 20160101 --end-date 20260424 --date-batch-size 120

$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000031.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000031.toml
# Combined: model-run --config ml_alpha\models\mdl_000031.toml
```

服务器准备好依赖和数据后：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000031.toml
```

本轮仅静态检查，不执行测试、训练、预测或全量生成；不提交测试脚本/产物。
