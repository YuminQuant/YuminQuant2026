# Huatai AI61 / 共同覆盖关联事件模型

参考2022-10-26《人工智能61：深挖分析师共同覆盖中的关联因子》第17–22页。
本轮仅加入事件模型，不增加前半部分的直接／间接关联动量和反转公式因子。

| 配置 | 输入 | 输出 |
|---|---|---|
| `mdl_000036` | 自身28类衰减事件次数 | `P(最高收益组) − P(最低收益组)` |
| `mdl_000037` | 自身28类＋Top3直接关联公司28类之和，共56维 | 同上 |

两个配置共用 `EventXGBoostAlphaModel`，属于事件驱动alpha模型，tag为`HTZQ`，
不是GAT，也不重复添加新的通用因子回归器。预测输出范围为[-1,1]。

**当前缺少完整事件源，代码接入不代表数据已具备。** 本地业绩预告不是完整事件库，
分析师报告也不是机构调研记录；不会从这些表拼出代理类别，或把尚未接入的类别填零。

## Event source contract / 待接入数据协议

默认原始入口为 `data/chn_stock_data/stock_events/`，仅接入规范化Parquet，不绑定某一家
供应商接口。后续Wind、其他数据源或自建适配器负责把源事件转换成以下格式。

```text
data/chn_stock_data/stock_events/
  manifest.json
  <available_year>.parquet
```

年度表列：

| 字段 | 类型 | 含义 |
|---|---|---|
| `event_id` | UTF-8 | 稳定事件标识；重复导出不得增加次数 |
| `ts_code` | UTF-8 | 项目股票代码，例如000001.SZ |
| `available_date` | Int32 YYYYMMDD | 首次可用于收盘后计算的交易日，不是活动发生日 |
| `event_type` | UTF-8 | 下表28个固定英文key之一 |
| `source` | UTF-8 | 来源及其接口／表标识 |

适配器应按可见公告日期映射交易日；非交易日公告映射下一交易日，有明确收盘后时间戳的
公告同样延至下一交易日。不推断供应商历史入库时间。未来信息不得通过回填活动日期提前。
同一真实事件的重传、预测年度重复行或重复参与人不能变成多个事件。
去重键为`(source,event_id)`，同键的日期、股票、类别冲突会报错；
跨股票或跨机构类别的记录应生成不同event_id，不能直接复用活动编号。
同一真实事件来自不同供应商时，适配器须先统一选源／去重，不能仅因source不同重复计数。
调研按“活动＋股票＋机构类别”形成事件，同类别多个机构参与同一活动计一次；
同一活动涉及不同机构类别，可在各类别各计一次。不同真实事件即使同日同股也分别计数。

`manifest.json`必须声明 `version="stock_events_source_v1"`、有序`event_types`、
`coverage_start`、`coverage_end`、`complete=true`、
`availability="explicit_first_available_trading_day"`、`universe="all_a_shares"`，
以及非空`source_revision`。日期为YYYYMMDD整数。完整性声明必须覆盖全A股、全部28类，
而不是“这次恰好拿到哪些记录”；不能用过滤后的事件样本宣布完整。
在完整区间内，某类别可以没有任何事件记录，不要求28类都实际发生。
请求涉及的每个年度文件都必须存在，无事件年度也应提供保留上述schema的空表。
示例manifest字段中的`event_types`为下表全部英文key，按表顺序排列。

| # | `event_type` | 事件 |
|---|---|---|
| 1 | `earnings_profit` | 业绩预盈 |
| 2 | `equity_incentive` | 股权激励 |
| 3 | `major_contract` | 签订重大合同 |
| 4 | `fiscal_subsidy` | 财政补贴 |
| 5 | `shareholder_increase` | 股东增持股票 |
| 6 | `litigation_win` | 诉讼胜诉 |
| 7 | `negative_rumor_clarification` | 澄清利空传闻 |
| 8 | `dishonest_execution_removed` | 移出失信执行 |
| 9 | `issuer_rating_upgrade` | 主体评级上调 |
| 10 | `earnings_loss_warning` | 业绩预警亏损 |
| 11 | `enforcement_subject` | 被执行人 |
| 12 | `shareholder_reduction_plan` | 股东拟减持股票 |
| 13 | `equity_freeze` | 股权冻结 |
| 14 | `regulatory_rectification` | 监管机构整改 |
| 15 | `litigation_defendant` | 诉讼被告方 |
| 16 | `dishonest_execution` | 失信执行 |
| 17 | `issuer_rating_downgrade` | 主体评级下调 |
| 18 | `exchange_inquiry` | 交易所问询函 |
| 19 | `abnormal_auditor_change` | 会计师事务所异常变更 |
| 20 | `seasoned_offering_termination` | 增发方案终止 |
| 21 | `survey_securities_asset_management` | 证券公司资管调研 |
| 22 | `survey_securities_proprietary` | 证券公司自营调研 |
| 23 | `survey_fund` | 基金公司调研 |
| 24 | `survey_insurance` | 保险公司调研 |
| 25 | `survey_investment` | 投资公司调研 |
| 26 | `survey_foreign` | 外资机构调研 |
| 27 | `survey_securities` | 证券公司调研 |
| 28 | `survey_other` | 其他调研 |

## Derivation and relationships / 派生与关联

在交易日T，类别k的输入为：

```text
event[i,k,T] = sum(2 ** (-(pos(T)-pos(e.available_date))/20))
              for e.stock=i, e.type=k, T-3自然月 < e.available_date <= T
```

`pos`为同一完整交易日历中的位置；周末不增加衰减天数。当天事件权重1，20个交易日前
事件权重0.5；自然月左边界严格排除。三个月是事件回看窗口，不是模型训练窗口。
不做MAD、z-score、中性化或负面事件手动变号，事件方向由分类器学习。

Rust `derive-stock-events`按年度读规范事件，自动读取首个输出日前三个月的事件历史，
按输出日期批次写到 `data/derived/stock/stock_events/<year>/<date>.parquet`。
日表包含 `trade_date, ts_code, is_daily_pv` 和有序的28个`event_<event_type>`计数列。
每日必须包含当天行情股票，以及窗口内所有仍有事件的股票；后者即使当天无行情也保留，
用于关联事件统计。Python只把`is_daily_pv=true`且自身28项至少一项大于0的股票作为目标，
再执行现有ST/北交所过滤。**两个模型目标池相同**，不因为关联公司有事件而扩大目标池。

日表外的邻居只有在已确认完整事件覆盖、且派生身份明确保证包含所有活跃事件股票时，
才可贡献零；当天无行情的邻居有事件时仍必须保留其真实计数。缺日表、缺类别或完整声明
不足均报错，不进行“缺文件就当无事件”的回退。

关联模型复用 `derive-analyst-coverage`：每季度最后交易日Q，使用`(Q−6自然月,Q]`报告，
按“机构＋拆分后姓名”识别分析师。T使用最近已完成的`Q<=T`快照，季度内冻结。
按共享分析师人数降序、股票代码升序选至多3个直接邻居，排除自身。
从完整覆盖成员选邻居，不根据当日事件、标签、ST或目标股票池重新排名或补位。
若不足3个则使用实际邻居数；自身未被覆盖时关联28维为0，但仍可使用自身事件。
关联特征为这些邻居28项逐项求和，不平均、不乘关联强度，也不把3个向量拼成84维。

源历史修订必须更新`source_revision`并重建一致派生数据、重训模型。
完整覆盖日期可以只扩展；范围不进入稳定模型指纹，单纯追加未来数据无需使旧模型失效。
窗口、衰减、类别顺序、源修订、可用日政策及关联政策均进入身份校验。
派生目录使用写锁、原子manifest与压缩Parquet；不允许在C盘写派生产物。
Python只保留有界日表／当前季度关系索引，不写预处理磁盘缓存。

## Target and fitting / 标签与训练

年度重训，近36个月按前33个月训练、后3个月验证，每5个交易日采样，单种子42。
配置训练范围20160101–20251231，预测范围20190101–20260424；实际窗口由交易日历决定。
公共成熟边界沿用`future_vwap_return_10d`的`lookahead_days=11`，T日训练末截面不晚于T−11，
训练／验证边界也执行隔离。预测不读取未来标签或未来季度关系。

每个日期在当前模型目标池的有效标签股票内，计算中点百分位
`p=(average_rank(return)-0.5)/N`，以`floor(3*p)`分为低／中／高三个组。
同收益保持同组，因此并列收益时人数不一定恰好三等分。少于3个标签的日期不用作监督；
每个训练窗口必须实际包含全部3类，验证池必须有可用目标。

普通GPU XGBoost三分类，`multi:softprob`，固定600轮、深度3、学习率0.03，
subsample/colsample均0.8、min_child_weight=1、L2=1。不调参、不验证早停，
不将验证数据合并回训练。三分类每轮3棵树，600轮合计1800棵单棵树。
复用训练／验证DMatrix，固定轮数结束后各计算一次mlogloss，避免无选模用途的逐轮验证。
训练每50轮可打印进度。要求CUDA，缺GPU或GPU编译支持时失败，不静默回退CPU。

研报原版是20日超额收益、过去一年年度重训；未披露精确类阈值、衰减半衰期、完整树参数
或最终概率映射。上述10日VWAP、36个月33/3、20日半衰期、三等分和P(高)−P(低)
均是用户确认的项目口径，不应写成研报原参数。3个月事件回看、28／56维及Top3结构来自研报。

## CLI and outputs / 命令与产物

先完成上述完整事件源接入，再从仓库根目录执行；以下命令未在开发中运行：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')

cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-stock-events `
  --asset stock --start-date 20160104 --end-date 20260424 --date-batch-size 120

# mdl_000037 additionally requires quarterly analyst coverage.
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-analyst-coverage `
  --asset stock --start-date 20160104 --end-date 20260424 `
  --window-months 6 --date-batch-size 20

& $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000036.toml
& $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000037.toml
# Separate steps: model-train / model-predict with the same --config.
```

`--date-batch-size`表示一次输出任务批次的日期数，不是预读多少天；事件历史仍自动回看。
首个输出为20160104时，事件源完整性及交易日历必须至少覆盖其三个月回看。
xz336同样从`/home/wym/YuminQuant/YuminQuant2026`运行，将路径分隔符换为`/`即可；
需自行准备完整事件数据、XGBoost CUDA环境及10日VWAP标签，不会自动下载事件源。

- 权重：`data/model_workspace/mdl_000036/artifacts/<window_id>/model.pkl`，
  `mdl_000037`使用自己的同级目录；保存原生XGBoost Booster的模型对象。
- 同目录manifest记录配置／输入身份，`model_info.json`记录类顺序、特征、参数及最终损失，
  `loss_history.parquet`只记录固定600轮结束时的一条训练／验证损失，而非伪造逐轮曲线。
- 预测：公共写入器合并到`data/models/<year>/<date>.parquet`，列为`mdl_000036`／`mdl_000037`。

本轮只做代码和配置接入及静态检查，不运行测试、Rust构建、数据生成、训练或预测。
