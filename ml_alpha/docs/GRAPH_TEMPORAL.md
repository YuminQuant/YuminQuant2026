# mdl_000011：异构图 GNN 与时序 GRU 因子合成

依据东方证券《基于异构图神经网络的股票关联因子挖掘》
（2024-01-02，因子选股系列之九十九）实现完整双分支。
输入是已有正式因子，不是原始 K 线，因此归入 `models/`，输出模型信号。
券商标签为 `DFZQ`。

## 数据可行性

2026-10-20 对当前仓库的检查：

| 输入 | 当前情况 | 使用口径 |
| --- | --- | --- |
| 正式日频因子 | 422 个有效因子；2503 个日文件，20160104–20260424 | 全部可用正式因子；排除 `model_generated`、`deprecated` 和模型自身输出 |
| 中信行业历史 | `data/index_data/member_ci/ci_members.parquet`，8712 行 | 一级行业 `l1_code`；复用公共 ClassificationMap 的 `in_date <= T <= out_date` 口径，空退出日表示仍有效，重叠时取最新入选记录 |
| 基金共同持仓 | `data/fund_data/portfolio` 当前没有年度 Parquet | 需要先恢复原始持仓；现有下载器和下面的字段协议可直接接入 |
| 分析师报告 | `data/chn_stock_data/analyst_report/{year}.parquet`，2010–2026 年共 17 个年度文件 | `report_date` 过去六个自然月内的共同覆盖 |

按数据库公告日期使用信息，不使用 `create_time` 或供应商入库时间。
基金字段为 `ann_date`；分析师接口对应字段叫 `report_date`；
行业接口没有 `ann_date`，使用历史入选/退出日期。

当前代码可以接入完整模型，但在基金原始数据恢复、关系图派生之前，
不能启动正式训练。缺少关系图、年度原始分区或基金可见持仓会明确报错，
不会自动退化成两种关系。

### 基金输入协议

现有 `FundPortfolioDownloader` 使用 `fund_portfolio` 接口，
按 **公告年份** 写入 `data/fund_data/portfolio/{ann_year}.parquet`：

| 字段 | 含义 |
| --- | --- |
| `ts_code` | 基金代码，不是股票代码 |
| `ann_date` | 公告日，YYYYMMDD |
| `end_date` | 财报期末，YYYYMMDD |
| `symbol` | 股票代码 |
| `mkv` | 持有市值，用于选择前十大重仓 |
| `amount` | 持股数 |
| `stk_mkv_ratio` | 占股票市值比例 |
| `stk_float_ratio` | 占流通股比例 |

日期为 Int32，其余数值为 Float64；下载器使用 ZSTD。
派生器只投影前五个所需字段；不需要额外修改下载器或发明新的原始产物格式。
同一基金、报告期、公告日应包含该次披露的完整持仓记录。
`portfolio/revisions/` 是下载器的修订归档，不作为图构建输入。

对 T 日，每只基金先取 `ann_date <= T` 的最新报告期，再取该报告期最新可见公告；
按 `mkv` 降序选前十，等值按股票代码稳定排序。晚公告的旧报告期不会覆盖较新报告期。
沿用最新已公告持仓直到新披露替换，不额外设置固定到期日。
缺少可靠的基金份额合并标识，因此按基金代码计数，不通过当前基金基本信息倒推历史类别。
这意味着不同份额可能重复贡献权重，停止披露的基金可能保留旧持仓；
后续如补充可靠历史基金身份/终止数据，应升级关系数据版本并重新派生。

### 分析师覆盖

窗口为 `(T - 6个自然月, T]`。作者字段按中英文逗号、顿号、分号等拆分，
以“机构 + 单个分析师姓名”识别覆盖组。
同股票、同机构、同分析师在窗口内的多份报告、多个预测年份只贡献一次覆盖。
不会把一个多人署名字符串误当成一个分析师。

## 派生数据与稀疏图

Rust CLI `derive-stock-relations` 写入：

```text
data/derived/stock/relations/
  manifest.json
  <year>/<YYYYMMDD>.parquet
```

每日表只有四列：`trade_date, relation, group_id, ts_code`。
`relation` 为 `industry / fund / analyst`；三种关系各自解释 group_id。
使用公共派生 Parquet 写入器的 Snappy 压缩。
这是可复用关系数据，不是预处理训练样本缓存。
源数据补充后，需要重新派生受影响的日期范围；Python 不隐式重建关系数据。

Python 构建稀疏股票—分组矩阵 B，每个成员关系值为 1：

```text
A = B Bᵀ - diag(B 1)
D = diag(A 1)
GCN 邻居消息 = D^(-1/2) A D^(-1/2) H
```

同行业股票之间权重为 1；共同基金或共同分析师越多，原始边权越大。
通过两次稀疏矩阵乘法完成聚合，不落盘股票对，也不创建 N×N 的稠密邻接矩阵。
去掉所有自环；孤立股票的邻居消息为零，自身信息由残差保留。

`--date-batch-size` 控制一次日期任务批次的大小，即最多同时持有/写出多少日关系表，
不是输入预读天数。默认 20；内存紧张时可调小。源数据按年度读取，基金状态按公告推进。

## 两阶段模型

```text
T 日正式因子 ── 3 层异构 GNN ── 32 个弱因子 ── XGBoost Ranker ── 截面 z-score ─┐
                                                                               ├─ 相加 → mdl_000011
T-39..T 因子 ── 2 层 GRU ───── 32 个弱因子 ── XGBoost Ranker ── 截面 z-score ─┘
```

- GNN：输入投影到 32 维，每层分别对三类关系做 GCN、ReLU、LayerNorm；
  邻居消息等权平均、Dropout 后，与自身信息按 0.5/0.5 相加，再做 LayerNorm。
- GRU：40 个连续交易日，2 层，隐藏维度 32，层间 Dropout 0.5；
  最后一个时间步进入因子单元。
- 因子单元：隐藏层 32、ReLU、Dropout、32 维弱因子输出和 BatchNorm，
  固定等权合成并做输出 BatchNorm。训练损失为截面标准化收益的 MSE
  加弱因子间非对角相关系数 RMS 惩罚（系数 0.05）。
- 两路神经网络独立训练，单种子 42，最多 100 epoch；
  各自按验证集平均日 Rank IC 选择最佳权重，连续 20 epoch 无提升则停止。
- 冻结各路最佳编码器，提取训练/验证日期的弱因子；再各自训练一个
  `rank:pairwise` XGBoost Ranker。每个日期是一个排序组，不能混成一个跨日期排序任务。
- 每路独立执行 **20 次 Optuna TPE**，复用已经构建好的训练/验证 DMatrix；
  不会在每个 trial 重训 GNN/GRU。每个 trial 连续 20 棵树无验证 Rank IC 提升则停止。
  使用验证集选出的最佳迭代树数预测；没有按五次 trial 停止的额外规则。
- 最终输出 `zscore(GNNRanker) + zscore(GRURanker)`，每股一个分数。
  常数截面的 z-score 定义为零。

Ranker 搜索范围：

| 参数 | 范围 |
| --- | --- |
| `n_estimators` | 300–800，步长 100 |
| `max_depth` | 3–8，整数 |
| `learning_rate` | 0.005–0.05，对数采样 |
| `min_child_weight` | 1–20，对数采样 |
| `subsample` | 0.7–1.0 |
| `colsample_bytree` | 0.7–1.0 |
| `reg_lambda` | 0.1–10，对数采样 |

研报未逐项披露 GNN 隐藏宽度、因子单元细节、惩罚权重和完整搜索范围，
上述取值是项目工程配置，不是声称完全还原原作者代码。
研报的 63 个量价因子、5+1 年训练验证、20 日中性化收益和五模型集成，
已按确认口径替换为当前正式因子库、33+3 月、10 日 VWAP 收益、单种子。

## 训练数据与信息边界

配置 `models/mdl_000011.toml`：

- 每年重训，近 36 个自然月，前 33 月训练、后 3 月验证，每 5 个交易日采样。
- 训练数据范围从 2016 年开始；计划从 2019 年起每日样本外预测。
- 标签 `future_vwap_return_10d`，`lookahead_days=11`；
  沿用公共 `build_windows` 的标签到期和训练/验证隔离。
  T 日收盘重训只能使用标签在 T 日及以前到期的截面；
  10 日 VWAP 标签最后允许截面为 T-11，实际还受 5 日采样影响。
- 时序分支每个样本使用连续 40 个交易日，**不是** 40 个五日采样点。
  当前因子库从 20160104 开始，因此前 39 个交易日不能组成完整输入；
  不伪造 2015 年的因子，也不补零整日缺失历史。
- 每个历史日独立执行股票池/ST/北交所过滤、MAD5 去极值、截面 z-score、
  缺失因子值填 0。最终股票必须在整个 40 日窗口具有合格输入；
  停牌等原因导致整日股票行缺失时，不拼凑不连续序列。
- 特征归一化和图节点选择不读取标签；没有有效标签的股票仍留在图中传播信息，
  仅在 MSE、验证指标和第二阶段排序损失中排除。
- 预测使用同样的完整合格截面，不接受部分股票子集作为完整图；
  预测不读取未来标签。训练与预测必须匹配因子列顺序、关系版本和配置指纹。

第二阶段使用冻结编码器在第一阶段训练集上的弱因子，不是 OOF stacking。
验证集用于两个阶段的早停和选参；样本外预测日期不参与这些选择。

## 存储与资源

- Python 只保留有上限的内存缓存，不写预处理 Pickle、序列磁盘缓存或 C 盘缓存。
  当前日因子缓存预算为 512 MiB；它只限制日因子缓存，不代表整个进程内存上限。
- 数据集主表只存日期、股票和标签，不展开全部 N×40×F 训练序列。
  序列按股票批次临时组装，默认 `batch_size=3000`。
- 422 个因子、3000 股、40 日的单批 Float32 输入约 193 MiB；
  CUDA 激活、梯度、参数和临时数组另占空间。GNN 每次必须处理完整日期图，
  但中间节点表示只有 32 维，不按股票截断邻居。
- 编码器冻结后，Ranker 只读 32 维弱因子；两路依次训练。
  GNN/GRU 编码器在对应 XGBoost 阶段移回 CPU，释放其 CUDA 空间。
- 关系存储随“股票—分组成员数”增长，不随全部股票对数量增长。
  基金数据尚未恢复，不能给出可信的全历史磁盘总量；无需预留固定 40 GiB 缓存。
- `device="auto"` 和 `ranker_device="auto"` 在 PyTorch CUDA 可用时启用 GPU；
  CPU 时分别使用 CPU 网络和 XGBoost `hist`。旧版 XGBoost 使用 `gpu_hist`，
  2.x 以上使用 `hist + device=cuda`。运行失败会报错，不静默从 GPU 重试 CPU。
- 不新增 PyG/DGL 依赖；使用现有 PyTorch、NumPy/Pandas、XGBoost、Optuna 环境。

## 正式命令

在仓库根目录执行。以下为正式数据恢复/派生/训练命令，开发时不自动执行。

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
$env:PYTHONPATH = "$PWD\ml_alpha"
$env:PATH = "D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;" + $env:PATH

# 恢复持仓。2015 年覆盖首个 2016 年截面需要的此前已公告报告。
& $py -B -u scripts\init_fund_data.py `
  --datasets portfolio --start-year 2015 --end-date 20260424

# 行业与分析师使用现有数据；派生全部三个关系。
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-stock-relations `
  --asset stock --start-date 20160104 --end-date 20260424 `
  --date-batch-size 20

& $py -B -u -m yq_ml_alpha model-train `
  --config ml_alpha\models\mdl_000011.toml

& $py -B -u -m yq_ml_alpha model-predict `
  --config ml_alpha\models\mdl_000011.toml

# 如需一次完成训练和预测，用此命令替代上面两个命令。
# & $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000011.toml
```

训练前还需确保对应日期的 `data/label/stock/daily` 中已有
`future_vwap_return_10d`，以及历史 `trade_filter` 文件完整。

产物：

- 模型：`data/model_workspace/mdl_000011/artifacts/<window_id>/model.pkl`，
  内容是 PyTorch checkpoint，包含两个编码器权重和两个 XGBoost Booster 原始字节。
- 同目录 `model.manifest.json`、`loss_history.parquet`、`model_info.json`：
  配置指纹、每轮损失/验证指标，以及每路 20 次 trial 的参数、成绩和最佳迭代数。
- 预测：`data/models/<year>/<YYYYMMDD>.parquet`，
  宽表键为 `trade_date, ts_code`，新增信号列为 `mdl_000011`；
  使用项目公共输出写入器保留其他模型列。
