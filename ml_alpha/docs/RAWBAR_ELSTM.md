# eLSTM量价端到端因子

参考东北证券2024-09-08《人工智能系列之四：指数门控在LSTM中的应用，eLSTM》。正式因子 `rawbar_elstm`，券商tag `DBZQ`，配置 `factors/rawbar_elstm.toml`。每个交易日为每只符合条件的股票输出一个分数；独立于既有因子合成模型eLSTM＋RankNet。

## 输入

输入为截至样本日的连续60个交易日，OHLC、VWAP、成交量六维。直接读取现有日频行情和复权因子，无需新增派生数据。OHLC、VWAP乘当日adj_factor；成交量转换为股，VWAP按amount×10/vol计算，沿用项目零成交量VWAP=0的规则。

每个股票、每个特征各自用该60日窗口的均值和总体标准差（ddof=0）做时序z-score。不做截面标准化、训练面板缩尾或跨股票归一化；不使用标签决定输入统计量。任一特征零方差或60日行情不完整即剔除整个股票样本，不填补历史。股票池为mkt_all，剔除ST和北交所。

公共 `DailyTensorSequence` 按日缓存复权后的六维截面，仅驻留RAM，预算512 MiB，跨epoch复用。窗口均值/标准差使用float64 Welford递推，只保留当前样本日统计量；不保存所有重叠窗口张量。HDFS历史最多30个读取线程、128 MiB预读，本地串行，不写磁盘缓存。`rawbar_sequence` 是独立输入协议，不改变原有 `rawbar` 因子的归一化。

## 网络与目标

`[股票数, 60, 6] → 单层64维eLSTM末步隐藏状态 → LayerNorm → Dropout(0.3) → Linear(64,1)`。

与普通LSTM不同，输入门和遗忘门使用指数函数，候选记忆为tanh，输出门为sigmoid；引入归一化状态n。共享eLSTM单元执行：

```text
m = max(a_f + m_prev, a_i).detach()
i = exp(a_i - m)
f = exp(a_f + m_prev - m)
c = f * c_prev + i * tanh(a_g)
n = f * n_prev + i
h = sigmoid(a_o) * c / (n + eps)
```

稳定器m不参与梯度；c/n之后没有额外tanh。初始化h/c/n/m为0，所有线性权重Xavier uniform、bias为0。共享单元从既有RankNet模型提取，保留原导入名和state_dict键，不改变旧权重兼容性。

损失采用用户确认的MSE。标签使用项目10日VWAP收益，均值和标准差只从训练集全部有效标签拟合，保存到权重并用于验证；预测无需标签，输出标准化收益分数，不进行样本外收益标准化。验证MSE按有效股票样本数加权。

研报使用t+2至t+10 VWAP相对全A等权超额收益，并分别对训练/验证/测试标签标准化；本项目使用已有10日VWAP标签及训练集拟合参数，二者不是逐字复现。研报batch256、200轮/20轮早停也按项目口径替换。

## 训练和输出

- 年度重训，近60个月：前57个月训练，后3个月验证。2011起训练，2016起逐年预测至2026-04-24。
- 固定每5个交易日取训练/验证截面，每个epoch遍历所有已选日期，不再抽取日期子集；股票小批次顺序可打乱。
- 10日VWAP、lookahead_days=11，继续使用公共标签到期清洗和训练/验证隔离，不由本模型另行推断可用标签日期。
- Adam，lr=1e-4，weight_decay=0；batch_size=5000，单种子42，最多100轮，验证MSE连续10轮未改善早停。GPU可用时自动使用GPU。
- 日志记录epoch、训练/验证MSE、最优MSE、早停计数、训练/验证日期数、每轮耗时及公共RAM缓存命中/加载/淘汰情况。

权重保存到 `data/model_workspace/rawbar_elstm/artifacts/annual_YYYY/model.pkl`（版本化PyTorch checkpoint），包含网络、训练标签统计量及诊断信息；同目录由公共流程写manifest、loss_history.parquet、model_info.json。预测写入 `data/factors/stock/daily/YYYY/YYYYMMDD_ml.parquet` 的 `rawbar_elstm` 列，使用现有日期锁和原子合并写出，可与其他模型重复 `--config` 联合预测。

## CLI

在仓库根目录使用现有环境，无需先派生：

```powershell
& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\rawbar_elstm.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\rawbar_elstm.toml
# 或训练并预测：
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\rawbar_elstm.toml
```

本次只接入代码和配置，未启动训练或Slurm任务。
