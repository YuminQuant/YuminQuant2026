# LogsigRNN_week：多变量高频路径的低频序列因子

参考东北证券《基于Logsig-RNN的高频数据低频化选股因子——机器学习系列之三》，文件名日期2023-09-08，正文发布日期2023-09-07。正式因子 `logsigrnn_week`，券商tag `DBZQ`，配置 `factors/logsigrnn_week.toml`。只实现该LSTM分支，不加入Sig_week的Lasso分支或二者合成。

## 输入和路径

每个股票样本使用截至当日的21个交易日，每日48根完整5分钟bar。Rust从现有1分钟行情聚合5分钟OHLC、成交量，要求每根包含5个不同的有效分钟；每日OHLC乘当日adj_factor。缺失关键数据、缺失复权、缺少bar或历史不完整的股票不进入对应样本。

在整个21日、1008根bar上构造六维序列：

```text
u = i / 1007, i=0,...,1007
p = log(adjusted_close)
r = (close - open) / (high - low)
upper = (high - max(open,close)) / std_21d(upper)
lower = (min(open,close) - low) / std_21d(lower)
cv = cumsum(volume) / sum_21d(volume)
```

影线长度使用复权价格，标准差为总体标准差ddof=0，不减均值。按用户确认，H=L时r=0；影线标准差=0时整个对应通道置0；21日总成交量=0则剔除样本。

对所有六个坐标取lead，对价格另取lag，得到顺序固定的七维路径 `[u_lead,p_lead,r_lead,upper_lead,lower_lead,cv_lead,p_lag]`。每次原始bar跳变先更新六个lead坐标，再更新lag价格，不增加basepoint或时间外的额外通道。先变换完整序列再按日切分：第一段含当日48个点，后20段共享前一日末点，各含49个原始端点，以保留跨日跳变。每段计算4阶截断log-signature。

标准Lyndon括号基按次数、字典序排列；各阶维数7、21、112、588，合计728。与signatory使用的不同线性基表示不保证逐列一致，但本实现保存完整同阶Lie元素，训练/预测固定使用同一基和列顺序。基实现使用张量指数、Chen乘积、截断张量对数及Lyndon三角消元，不是原始signature，也不是抽取728个任意张量项。

## 为什么提前派生

每个epoch重复从分钟数据计算21段log-signature会反复承担分钟IO及高阶路径积分，因此新增Rust `derive-logsigrnn`。原有 `derive-logsig` 为20日成交量、二维lead-lag、10阶、226维，不能直接复用，也不修改其输出。

新数据存放：`data/derived/stock/logsigrnn_week/YYYY/YYYYMMDD.parquet`，包含股票/日期键、definition_version=1、日内段intra_0001...0728、带前一日末点的bridge_0001...0728，以及上/下影线均值、M2、成交量和五项float64统计量。路径积分在float64计算，归一化后的日系数以float32、Snappy保存。

日派生暂用本日影线标准差及成交量和作数值尺度，零尺度替为1。Python以21日统计量合并得到整窗影线标准差和成交量和，再根据每个Lyndon词包含各通道的次数缩放系数。因为Lie基多重齐次，这与先按整个21日窗口归一化再求log-signature等价。时间尺度始终固定1/1007；累计成交量的起始常数和绝对时间偏移对不带basepoint的signature无影响。第一日使用intra，其余20日使用bridge，窗口外的前一日不会进入第一段。

无需把每个样本日的21×728张量重复存盘。每日一行的浮点主体约5864字节，5000只股票时约28 MiB/日、250交易日约6.8 GiB（未压缩，不含键和Parquet开销；实际Snappy体积依覆盖/数据变化）。这是数据规模估算，没有执行全量生成或磁盘预算变更。

日截面缓存在 RAM 中按需填充，默认无字节上限。每个训练交易日组装一次完整截面，训练三遍后释放该日输入张量。5000 股的 21×728 输入张量约 292 MiB，每个 1000 股 mini-batch 的输入约 58 MiB，另有日缓存、归一化暂存和 GPU 激活占用。

`--date-batch-size`延续Rust派生语义：把待输出日期划分为批次，每批日期任务由threads线程执行；不是训练采样间隔，也不是预先保留该数量天的原始分钟表。每个日期读取本日和上一交易日；年分片会自动读取年前一天来构建bridge，不依赖上一年度任务先完成。缺失上一日原始分钟文件时可以生成本日intra，bridge为空；只有需要该连接的21日样本会被剔除。

## 网络和训练

`[stocks,21,728] → 输入BatchNorm(728) → 两层LSTM(hidden=32,层间Dropout=0.5) → 末步隐藏状态 → BatchNorm(32) → Linear(32,1)`。

训练时输入 BN 按 mini-batch 内股票和 21 个时间位置计算统计量，输出 BN 使用该批股票的末步隐藏状态。预测使用保存的 BN 运行统计量。

每个训练窗口每隔 5 个交易日固定采样一次，按日期顺序遍历合格截面。每个截面在有有效标签的股票上连续训练 3 遍，每遍以种子 42 控制股票打乱顺序并按 1000 股分批。每个有效 mini-batch 以 −Pearson IC 为损失，执行一次 Adam 更新，学习率为 0.001。全部截面训练完成后评估独立验证集的日均 IC，保存最终权重。

- 项目统一年度滚动5年，57个月训练＋3个月验证。
- 2011起训练，2016起逐年预测，预测截至2026-04-24；原论文的周频含义是回测调仓频率，本库仍可每日输出。
- 项目10日VWAP标签，lookahead_days=11；公共层按标签到期边界及训练/验证分界清洗。输入归一化只使用样本日及以前的行情。
- 日志按训练截面记录进度、训练 IC、Adam 更新次数、耗时和缓存统计，结束时记录验证 IC。

## CLI和产物

在仓库根目录执行。以下第一步先从交易日历计算覆盖首次训练输入所需的20个历史交易日，生成期间也会读取每个日期的前一日；需确保这些原始分钟/复权数据存在。该日期推导只读日历，不生成数据。

```powershell
$start = & $py -B -c "from yq_ml_alpha.calendar import TradingCalendar; c=TradingCalendar.load('data'); d=c.between(20110101,20110131)[0]; print(c.dates[max(0,c.dates.index(d)-20)])"
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-logsigrnn `
  --asset stock --start-date $start --end-date 20260424 `
  --date-batch-size 120 --threads 4

& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\logsigrnn_week.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\logsigrnn_week.toml
# 或派生完成后训练并预测：
& $py -B -u -m yq_ml_alpha factor-run --config ml_alpha\factors\logsigrnn_week.toml
```

权重在 `data/model_workspace/logsigrnn_week/artifacts/annual_YYYY/model.pkl`，含版本、参数、LSTM/BN/线性权重和诊断。同目录保存 manifest、loss_history.parquet 和 model_info.json。预测因子写入 `data/factors/stock/daily/YYYY/YYYYMMDD_ml.parquet` 的 `logsigrnn_week` 列。

本轮不启动正式派生、训练、Slurm或正式预测，不改动运行任务。
