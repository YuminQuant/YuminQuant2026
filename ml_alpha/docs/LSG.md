# LSG：分数阶记忆谱与相空间状态门控

参考西南证券2026-10-14《量价特征时序平稳化与几何建模的联合升维选股模型——分数阶微分记忆谱与相空间重构》。正式因子为 `lsg_alpha`，配置 `factors/lsg_alpha.toml`，券商tag为 `XNZQ`。

本项目版本直接联合训练完整LSG网络，不依次训练FDS、FSDE和RAB，也不将这些模型的预测列串接起来。

## 输入与日期

使用7项市场数据和本地9项CNE6一级风格暴露，共16维，不使用EMA、DIF、DEA、MACD：

| 来源 | 字段 |
|---|---|
| 日行情 | open、high、low、close、VWAP、volume |
| 日基础数据 | turnover_rate_f，保留原始百分数 |
| CNE6一级 | DIVIDEND_YIELD、GROWTH、LIQUIDITY、MOMENTUM、QUALITY、SENTIMENT、SIZE、VALUE、VOLATILITY |

默认读取路径由配置的 `data_root` 和 `features.root` 解析：

- 行情：`data/chn_stock_data/daily/pv/YYYY/YYYYMMDD.parquet`。
- 自由流通换手率：`data/chn_stock_data/daily/basic/YYYY/YYYYMMDD.parquet`。
- 复权因子：`data/chn_stock_data/daily/adj_factor/YYYY/YYYYMMDD.parquet`。
- 风格暴露：`data/barra/stock/daily/CNE6/YYYY/YYYYMMDD.parquet`。

VWAP按 `amount * 10 / vol` 转换为元，volume按 `vol * 100` 转换为股；已观测的零成交量日VWAP置0。价格乘对应日期的 `adj_factor`。先按日期/股票连接，使用项目股票池并剔除ST、北交所；需要完整52个交易日的有效行情、换手和复权历史，不前向补齐。

确认采用的归一化按完整样本日股票截面进行，标签不参与：

1. 五项价格除以样本末日复权收盘价。
2. 成交量、换手率分别除以该股票52日均值；全窗口为0时保持0。
3. 各历史时间位置、各特征分别按股票截面做z-score（总体标准差）；Barra不做末值比值，缺失暴露在z-score后填0。

截面归一化在股票编码分块之前完成，改变 `batch_size` 不改变归一化股票集合。未取得收益标签的股票仍参与输入归一化，仅在损失中屏蔽。

原研报使用10个CNE5风格项、包含技术指标；这里的16维输入和分类型归一化属于用户确认的项目版本。因子含基本面风格信息，使用 `fundamental` tag，避免被仅量价的模型配置误选。

## 完整网络

输入为 `[stocks, 52, 16]`，所有数据不晚于预测日收盘。

1. **FDS记忆谱**：对阶数 `[0, 0.25, 0.5, 0.75, 1]` 构造因果GL权重矩阵。每个特征学习高斯核注意力的中心与宽度，两条分支共享这组参数。谱融合与卷积可利用线性性结合计算，避免显式保留五套完整股票输入。
2. **时序路径**：仅对最后40日独立构造40日谱，不把额外12日并入其GL算子；融合后线性投影到32维，加固定正弦位置编码，进入两层四头Transformer，最后取末步。
3. **几何路径**：优先研报正文的“双谱”描述，对完整52日构造另一套谱并融合，再以 `tau=3、m=5` 做Takens嵌入，得到 `[stocks, 40, 16, 5]`。共享Conv1d（1→32、核3、无填充、ReLU）沿五个延迟坐标读形，对卷积位置均值汇聚；每个特征使用同一个单层32维GRU沿40日聚合，拼接所有特征后投影到32维。
4. **RAB偏置**：对每项特征计算40个延迟向量的两两欧氏距离，再跨16项特征取均值；自距离置0，按每只股票距离矩阵的均值归一化。每层注意力在softmax前加入 `-lambda * D`，lambda逐层独立可学习，并投影为非负。注意力仅连接该股票截至样本日的40个已观测状态，不连接其他股票或未来数据。
5. **LSG门控**：取距离矩阵最后一行，排除当前状态自身，在其余39个历史状态中取最近5个距离的均值 `r_rel`；按研报公式 `g = 1 + softplus(w_raw) * softplus(r_rel)` 放大几何向量。
6. **输出**：拼接两个32维向量，经过64→32→1的ReLU预测头，每支股票输出一个分数。

研报关于softplus的文字说明有笔误：softplus不保证大于等于1，`w_raw=0` 也不会让门控精确等于1。实现保留公式，按确认的 `w_raw=-5` 初始化，使初始放大系数接近1；门控不是Lyapunov指数斜率估计。

Transformer宽度、层数、头数、FFN宽128、Dropout 0.1、CNN位置均值汇聚、GRU宽度、融合头、距离归一化、初始化等属于明确配置的工程口径，并非研报完整披露参数。所有可调维度、学习率和训练资源参数来自TOML。

## 训练与内存

- 年度重训，每个窗口回看60个月，前57个月训练、后3个月验证；5日固定采样，每个epoch遍历全部训练日期。
- 训练从2011年起，2016年起逐年预测至2026-04-24；10日VWAP标签，公共层按11个交易日到期边界清洗训练/验证样本。
- 单种子42，Adam学习率5e-5，最多100轮，验证IC连续10轮未改善早停，保存最优权重。
- 每个样本日只计算一次完整截面 `1 - Pearson IC`，只更新一次参数。`batch_size=5000` 限制股票编码块，分块预测拼接后再计算完整截面损失，不把各块IC当成独立目标。梯度检查点用于控制显存。
- 复用公共 `DailySequenceProvider` 和 `DailyTensorSequence`：日表连接结果以float32在RAM中按预算保留，跨epoch复用；窗口依赖的时序/截面归一化在每个样本日执行。当前日表缓存1 GiB、在途预读128 MiB，本地串行、HDFS可并发30；这些预算不包括模型和当前编码激活。
- 不写磁盘样本缓存、不新增派生数据文件。`factor-materialize` 对该输入协议明确报错，避免误以为键/标签表就是完整训练张量。
- 每轮日志包含训练/验证IC、耗时、缓存加载/淘汰增量和Parquet打开次数。此次实现不启动训练或全量数据生成。

## 正式入口与产物

在项目根目录使用已有Python环境：

```powershell
& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\lsg_alpha.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\lsg_alpha.toml
# 或合并训练与预测：
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\lsg_alpha.toml
```

权重和manifest在 `data/model_workspace/lsg_alpha/artifacts/annual_YYYY/`；权重文件名为 `model.pkl`，内容为PyTorch checkpoint，包含参数、列顺序、state_dict及诊断。损失历史和模型信息沿用公共诊断目录规则。

预测列为 `lsg_alpha`，公共写出层按日期锁合并到 `data/factors/stock/daily/YYYY/YYYYMMDD_ml.parquet`，支持重复 `--config` 的联合预测。回测沿用正式因子入口和 `data/backtest/stock/daily/lsg_alpha/` 路径，不需要新增回测协议。
