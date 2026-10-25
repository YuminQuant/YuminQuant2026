# Huatai AI 58 / 分析师共同覆盖 GAT

`mdl_000035` 参考华泰研究 2022-07-07《人工智能58：分析师共同覆盖因子和图神经网络》
第6页关系构建、第24–26页GAT结构与公式。输入是正式因子，归入因子合成模型，
券商tag为 `HTZQ`；最终每只合格股票输出一个模型分数。

另有独立正式因子 `analyst_coverage_gat_40d`，从原始行情派生报告的三项40日输入，
复用同一GAT，写入正式因子库；详见 [三输入正式因子](ANALYST_COVERAGE_GAT_40D.md)。
`mdl_000035` 继续使用全部有效正式因子，不替换其输入或模型编号。

本轮实现报告的单层残差GAT，不重复添加线性对照组，也不新增前文的确定性
共同覆盖动量、反转、换手率和波动率因子。这些因子属于独立的公式因子生产范围。

## Architecture / 网络

```text
T-date covered stocks x F formal factors
  -> W: Linear(F,5, bias=False)
  -> one additive GAT head, over all co-covered neighbors plus self:
       e_ij = LeakyReLU(a_source . W*x_i + a_target . W*x_j)
       alpha_ij = softmax_j(e_ij), j in neighbors(i)
       message_i = LeakyReLU(sum_j alpha_ij * W*x_j)
  + skip_i = Linear(F,5, bias=True)(x_i)
  -> Linear(5,1) -> one scalar score
```

两组注意力向量各5维，GAT只有一层、一个头；独立残差含偏置。
输入3因子时参数量为 `3*5 + 2*5 + (3*5+5) + (5+1) = 51`，与图表53一致。
本地输入维度F由正式因子库推断，参数量为 `10*F+21`，不再固定51。
注意力打分和消息激活的LeakyReLU斜率均为0.2；该斜率与Xavier初始化为工程设定。
没有额外MLP隐藏层、BatchNorm、LayerNorm、Dropout、GRU、多头或第二阶段树模型。

共同分析师数量只决定是否有边，不作为边权或注意力先验；多个分析师覆盖同一股票对，
连接仍为1。报告前文统计因子用到的 `log(1+共同分析师数)` 不用于此GAT。
softmax沿每个查询股票的全部邻居归一化，允许非对称注意力。
自环始终存在；有覆盖但其他关联股票都被过滤时，节点仍可用自身消息及残差预测。
输出没有固定数值范围，是选股分数，不是收益百分比；不额外执行行业/市值中性化。

## Coverage data / 覆盖数据

原始年度文件：`data/chn_stock_data/analyst_report/<year>.parquet`。
只读取 `ts_code, report_date, org_name, author_name`，无需基金持仓、行业或报告正文。
复用原三关系派生器的分析师事件解析，作者按中英文逗号、顿号、分号、斜杠和竖线拆分，
以“机构＋单个分析师姓名”标识。缺日期/机构/有效姓名的记录跳过；覆盖成员在窗口内去重，
多篇报告和多个盈利预测年份不会重复产生覆盖成员。按数据库report_date判断可见性，
不额外推断供应商入库时间；原始数据修订后应重新派生并更新模型data_version。

每季度最后一个交易日Q收盘，使用 `(Q−6个自然月, Q]` 的报告构建覆盖表，
此后沿用至下个季度快照。样本日T选择最近的 `Q<=T`，Q当天收盘即可使用新图。
因此两次更新之间的新报告不会立即进入图，上市或开始覆盖的股票可能等到下次更新才有值。

季度末用完整交易日历中“下一交易日进入下一季度”确认，不把命令区间末日当季度末。
Rust与Python使用同一判定。Python根据日历定位**应有**快照，缺文件会明确报错，
不会悄悄退回更旧季度。日历末尾若缺下一季度日期，就不能确认该季度结束。

新派生入口 `derive-analyst-coverage` 输出：

```text
data/derived/stock/analyst_coverage/
  manifest.json
  <year>/<quarter_last_trading_date>.parquet
```

表结构为 `trade_date, relation, group_id, ts_code`；trade_date为季度锚点，relation固定
`analyst`，其余为分组和股票代码。只存成员关系，不落盘全部股票两两连接。
使用公共派生Parquet写入器的Snappy压缩，空覆盖写完整schema的空快照。
所需原始年度文件缺失时失败，不将缺失源数据冒充无覆盖。

manifest为 `analyst_coverage_v1`，记录6月窗口、quarterly_end、last_open_session、
report_date_lte_snapshot及institution_split_author。首次写快照前原子保存身份；
同目录身份不一致，或目录非空却无manifest时拒绝混写。延长日期范围不会修改稳定身份。
`--window-months` 默认6，新模型配置也要求6；更改窗口需清理/迁移旧派生目录并重建。

命令自动补首个请求交易日前最近一次季度快照，并生成区间内完成的季度快照。
例如从2016年初请求时需要上一年季末快照及此前半年报告，不要求手动生成每日重复表。
`--date-batch-size` 表示一个输出任务批次最多包含多少个**季度快照**，不是预读多少日输入。
原三关系 `derive-stock-relations` 的默认行为、日频滑窗和 `stock_relations_v1` 保持不变。

## Inputs, target and windows / 输入、标签与窗口

用户确认使用本地全部有效正式因子，排除deprecated和model-generated列，单日输入。
新provider `analyst_factor_sequence` 固定sequence_length=1，复用公共有界历史读取器，
没有额外的因子时序窗口。股票池为mkt_all中有当期可见覆盖且通过ST/北交所过滤的股票。
先确定覆盖股票和当日筛选，再逐因子MAD5缩尾、截面z-score、缺失填0，不做中性化。
`FactorHistory._filter_frame` 在归一化前实现可选股票池细化，其他模型默认行为不变。

训练/预测保留同样的完整图截面，缺失未来标签的股票仍参与消息传递，只从损失排除。
不允许把任意股票子集当成完整图预测。未覆盖股票不输出有效分数。
标签使用 `future_vwap_return_10d`，在当日有效标签股票中做普通收益z-score，
不用收益排名或指数样本权重。有效标签少于2个或收益恒定的日期跳过。

年度重训，过去36个月按前33个月训练、后3个月验证，每5交易日采样，单种子42。
配置训练20160101–20251231，预测20190101–20260424；实际日期由交易日历决定。
10日VWAP标签从t+1至t+11，公共 `lookahead_days=11` 保证T日重训末截面不晚于T−11，
并执行训练/验证边界隔离。预测不读取标签或样本日之后的报告。

报告原版只用40日反转/换手/波动率3个因子、20日收益、约半年滚动及5种子集成；
这些已按用户确认替换为项目口径。网络结构保留报告的小型GAT，输入因子数量扩大。
报告没有披露全部输入预处理细节，本地MAD5/zscore和标签zscore已明确为工程口径。

## Training and resources / 训练与资源

每个完整日期图进行一次MSE优化更新，Adam lr=0.0005，最多100轮，验证日期等权
平均MSE连续10轮无提升早停；保存最佳验证MSE权重。无Optuna、多种子或参数平均。
每轮打乱季度块顺序，块内日期正序处理，复用当前季度成员表，不改变样本范围或采样频率。

输入投影股票块最大3000；注意力查询块512，每块仍面对全图所有节点和二值邻居。
训练用checkpoint重算查询块注意力及掩码，控制峰值显存，不保存N×N全图矩阵。
CPU只缓存最近季度成员表，因子日快照预算512MiB，不写磁盘预处理缓存或C盘缓存。
CPU输入、字符串、分组索引和GPU激活另占空间，该预算不是整个进程的内存上限。
图很稠密时，计算量仍可接近N²；分块降低峰值内存，不承诺线性耗时。
`device="cuda"` 显式要求GPU，不静默回退CPU，不引入PyG/DGL或SciPy新依赖。

## Artifacts and CLI / 产物与命令

- `data/model_workspace/mdl_000035/artifacts/<window_id>/model.pkl`：PyTorch checkpoint。
- 同目录manifest、model_info.json、loss_history.parquet：输入/覆盖身份、网络参数、
  每轮MSE、验证评分、早停和耗时。
- `data/models/<year>/<YYYYMMDD>.parquet`：公共写入器合并 `mdl_000035` 分数列。

仓库根目录运行，开发时未执行以下生成、训练或预测命令：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')

cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-analyst-coverage `
  --asset stock --start-date 20160104 --end-date 20260424 `
  --window-months 6 --date-batch-size 20

& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000035.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000035.toml
# Combined: model-run --config ml_alpha\models\mdl_000035.toml
```

xz336 数据与PyTorch CUDA环境准备好后：

```bash
cd /home/wym/YuminQuant/YuminQuant2026
cargo run --release --manifest-path factor_engine/Cargo.toml -- derive-analyst-coverage \
  --asset stock --start-date 20160104 --end-date 20260424 --window-months 6 --date-batch-size 20
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000035.toml
```

本次只做静态代码审查、Python语法及配置检查、Rust格式检查；未编译Rust，
未运行测试、派生生成、训练或预测。源分析师数据可用不等于新季度派生文件已经生成，
正式训练前须先执行上述独立派生命令。
