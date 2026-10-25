# Huatai AI69 / 股票间Transformer

依据2023-08-22《Attention is indeed all you need》第7–10页，尤其图表4及图表7。
本次按用户选择只实现股票间Transformer，编号`mdl_000038`，tag为`HTZQ`。
输入为正式因子截面，归入因子合成alpha模型。

## Comparison / 与现有Transformer的区别

| 项目 | 现有`rawbar_transformer` | 新`mdl_000038` |
|---|---|---|
| 每个token | 一只股票历史中的一个交易日 | 当天的一只股票 |
| 输入 | 每股30日、6个量价通道 | 当天完整合格股票截面、全部有效正式因子 |
| 注意力 | 同一股票的历史时间点之间 | 同一天所有股票之间 |
| 时间／位置编码 | Time2Vec相对时间 | 无；股票排列不是经济特征 |
| 编码器宽度 | 输入投影到64维 | 保持因子宽度F；仅补零使其可被头数整除 |
| 输出 | 单个IC目标分数、正式因子 | 10／20日双头均值、模型分数 |

注意力轴从时间变成股票，属于结构上的增量，而非更换历史窗口或学习率。
原DFZQ量价Transformer不替换；研报普通时序Transformer未重复添加。
研报的任务间注意力也未添加，现有`mdl_000033/34`保持原网络与checkpoint格式。

## Network / 网络结构

```text
T-date N stocks x F factors
  -> append zero feature if F is odd; D = ceil(F / 2) * 2
  -> Encoder 1: two-head full-stock scaled-dot-product attention
       -> residual + LayerNorm
       -> Linear(D,64) -> ReLU -> Dropout(0.1)
       -> Linear(64,D) -> Dropout(0.1)
       -> residual + LayerNorm
  -> Encoder 2: identical structure, independently initialized parameters
  -> two independent heads:
       Linear(D,256) -> LeakyReLU(0.01) -> Dropout(0.1) -> Linear(256,1)
  -> mean(10-day prediction, 20-day prediction)
```

每个注意力层均有标准可学习Q/K/V及输出投影，无额外输入升维层、位置编码、Time2Vec、
行业或分析师关系掩码。所有当日股票可相互关注，包括自身；不跨交易日做注意力。
补零仅解决因子数与2头的整除问题，原因子顺序和数值保留，补零数记录在model_info。
预测时股票重排只应使输出对应重排，不赋予股票代码顺序含义。

## Inputs and labels / 输入与标签

复用公共`factor_sequence`，固定`sequence_length=1`；这里的1表示只取当前日因子，
股票token数为当日合格股票数N，不是把Transformer序列长度设成1。
无需新增Rust派生数据。输入为本地全部有效正式因子，排除deprecated及模型生成列。
股票池沿用mkt_all及ST/北交所过滤，不新增报告的季度市值／流动性前60%筛选。

provider在关联未来标签之前确定完整当日股票池，逐因子MAD5缩尾、截面z-score及缺失填0，
无中性化。使用512MiB的现有进程内因子日表预算，不写预处理磁盘缓存或C盘缓存。
这不是整个进程的内存上限，训练张量、注意力和模型另占资源。

双标签为`future_vwap_return_10d`和`future_vwap_return_20d`，分别声明成熟期限11和21。
公共层按最大21个交易日执行T日训练的标签成熟约束及训练／验证隔离；
T日训练的最后截面不晚于T−21，5日采样可能让实际末截面更早。
预测不读取未来标签，也不根据标签缺失删除注意力节点。

监督股票取两个标签共同有效的当日子集，收益分别平均排名后z-score；
每项任务样本权重按收益排名指数衰减，半衰期0.5、当日均值归一到1。
**缺少任一未来标签的股票仍参与完整截面的注意力，只从监督损失中排除。**
共同有效股票少于2只或任一任务收益恒定的日期不用于监督，诊断记录跳过次数。

## Workflow and resources / 训练与资源

年度重训，近36个月按前33个月训练、后3个月验证，5交易日采样，单种子42。
配置训练20160101–20251231、预测20190101–20260424。Adam lr=1e-4，最多200轮，
验证日期等权、任务等权的加权MSE连续20轮无提升早停，保存最佳验证权重。

训练采用DWA：前两轮两任务等权，此后用此前两轮各任务原始加权MSE的比值，
temperature=2，经softmax得到任务系数；复用现有多任务损失helper。
验证选模不乘动态任务系数，预测始终为两个头的算术均值。
DWA、收益权重半衰期、单种子、本地VWAP及年度33/3方案为用户确认的项目口径，
报告没有明确这两个注意力模型的任务损失采用UW还是DWA。

每个交易日完整截面进行一次优化更新，不把5000支股票切成互不通信的小截面。
`batch_size=3000`控制输入、QKV投影、FFN与输出头的股票块上限；
`attention_chunk_size=512`只切查询Q，每块始终使用全日K/V和全股票softmax。
训练时checkpoint重算查询块注意力，保留Dropout随机状态，控制峰值显存。
必须先算完第一层全部股票的输出，再进入第二层，不能各股票块独立走完两层。

按日期单独执行，避免原报告一次padding32个日期的显存开销，也无需padding mask；
该批次安排是工程调整，完整当日关系仍保留。训练Dropout随机数路径随分块可能变化，
不承诺不同块大小训练结果逐位一致；预测关闭Dropout。
完整截面注意力计算量仍近似N²，查询分块主要降低峰值内存，不将复杂度改成线性。
配置显式`device="cuda"`，沿用公共设备选择器，缺可用GPU时失败，不静默回退CPU。

## CLI and outputs / 命令与产物

已有正式因子库及两个标签到位后，从仓库根目录执行；本轮未执行以下命令：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')

& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000038.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000038.toml
# Combined: model-run --config ml_alpha\models\mdl_000038.toml
```

xz336使用相同配置及已有CUDA PyTorch环境：

```bash
cd /home/wym/YuminQuant/YuminQuant2026
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000038.toml
```

- `data/model_workspace/mdl_000038/artifacts/<window_id>/model.pkl`：独立类型的PyTorch checkpoint。
- 同目录manifest、model_info.json、loss_history.parquet：配置／输入身份、因子维度及补零数、
  每轮任务损失、DWA系数、最佳轮次和耗时。
- `data/models/<year>/<date>.parquet`：公共写入器合并`mdl_000038`分数列。

本轮仅实现代码、配置与静态检查，不运行测试、数据生成、训练或预测。
