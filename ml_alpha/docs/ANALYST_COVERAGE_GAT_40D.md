# Analyst-coverage GAT / 40日三输入正式因子

`analyst_coverage_gat_40d` 以华泰人工智能58报告第24–25页的三输入GAT为基础，
作为独立正式因子管理，券商tag为 `HTZQ`。
原 `mdl_000035` 使用本地完整正式因子库，本配置只使用直接从行情计算的三项40日特征。
两者共用 `AnalystCoverageGATAlphaModel`、季度覆盖图、完整截面训练与预测逻辑，
权重和输出目录各自独立。

```text
raw daily close + adj_factor + turnover_rate_f
  -> Rust derive-analyst-gat-inputs: three 40-day features
  -> current analyst-covered eligible cross-section, MAD5/zscore
  -> one-head GAT(F=3, hidden=5) + independent linear skip + FC
  -> analyst_coverage_gat_40d, one score per stock
```

此入口从行情派生固定统计特征后学习合成，不额外训练原始K线编码器。
三项中间特征写入derived，不向正式因子库增加三个基础列，也不依赖已有模型生成列。

## Confirmed feature definitions / 已确认输入口径

令 `C_t = close_t * adj_factor_t`，r_t为复权简单日收益率。

| Column | Definition |
| --- | --- |
| `return_40d` | `C_T / C_(T-40) - 1`；保留累计收益原始正负，反转方向由模型学习 |
| `turnover_mean_40d` | 最近40个交易日 `turnover_rate_f` 的算术均值，保留原始百分数单位 |
| `volatility_40d` | `r_(T-39)..r_T` 的总体标准差，`r_t=C_t/C_(t-1)-1`，ddof=0，不年化 |

完整输入需要41个连续交易日的有效正复权收盘价，以及末40个交易日的有效非负换手率。
最老锚点日T−40的换手率不参与均值，也不要求有效；该日收盘价参与收益和波动率计算。
换手率为0有效，大于100%的值保留原值。复权日收益和累计收益都不额外取负或截断。
标准差使用双精度稳定计算；这三项公式、自由流通换手率字段及ddof均为用户确认口径，
报告未披露的部分不声称是原作者参数。

源数据路径：

- `data/chn_stock_data/daily/pv/<year>/<date>.parquet`：close。
- `data/chn_stock_data/daily/adj_factor/<year>/<date>.parquet`：adj_factor。
- `data/chn_stock_data/daily/basic/<year>/<date>.parquet`：turnover_rate_f。

计算按交易日历推进，不按单只股票的观测行数压缩时间。单股票行情、复权因子或换手
缺失时，该日保留无效占位，不前向填补；窗口不完整时派生三列为null。
Python严格读取并剔除任一输入非有限的行，不将缺少40日历史的股票填0后纳入图。
全日原始文件或关键字段缺失则报错，避免将数据故障当作正常无效样本。

## Derived data / 派生数据

新入口 `derive-analyst-gat-inputs`：

```text
data/derived/stock/analyst_gat_inputs/
  manifest.json
  <year>/<YYYYMMDD>.parquet
```

每日表为 `trade_date, ts_code, return_40d, turnover_mean_40d, volatility_40d`，
三列数值为可空Float64，使用公共Snappy压缩和原子Parquet写入。
只写请求范围内的每日输入，自动从首个目标交易日前40日预读必要历史，
滚动状态跨批次保留；无需用户把start-date手动提前40日。
`--date-batch-size` 控制一次输出日期任务批次，默认20；不改变40日窗口。
未完成窗口仍写完整schema的null行，避免回填或不连续样本。

稳定身份为 `analyst_gat_inputs_v1`，记录40日窗口、复权公式、换手字段、波动率分母
和完整性规则；不含请求日期范围，可延长日期。
使用目录旁锁和原子manifest身份写入，拒绝混合不同口径。源数据修订后需重建对应派生
范围及其后40日受影响窗口，并更新训练配置的data_version；manifest不哈希整库内容。
同一套派生输入可以由其他模型复用，不是训练专用预处理缓存。
`factor-materialize` 对该配置明确拒绝写调试样本缓存，并提示使用上述Rust派生入口。

季度分析师图继续读取 `data/derived/stock/analyst_coverage`，每季度最后一个交易日
Q收盘使用 `(Q−6自然月,Q]` 的报告建图，样本日用最近 `Q<=T`。
作者按机构＋拆分姓名去重，二值边加自环；缺应有季度快照时报错，不用旧季替代。
图的来源、分块注意力及报告差异见 [GAT公共说明](ANALYST_COVERAGE_GAT.md)。

## Training and output / 训练与输出

用户确认日期：训练20130101–20251231，预测20160101–20260424。
年度重训，过去36个月前33个月训练、后3个月验证，5交易日采样，单种子42。
首个名义窗口为2013–2015年训练/验证、2016年预测；公共标签成熟和隔离规则继续生效。
标签为10日VWAP，T日重训最后可用特征截面不晚于T−11；实际最后采样日还受5日抽样影响。

股票池为通过公共ST/北交所筛选、三项输入完整并有当期分析师覆盖的股票。
先确定合格输入股票，再做每列MAD5缩尾、截面z-score；仅损失排除缺标签节点。
保留公共层2016年前不执行ST剔除的现有规则，不在本因子中另写例外。
标签采用当日有效股票收益z-score，损失MSE。单头单层5维GAT、独立线性残差及FC
正好51个参数；Adam lr=0.0005、最多100轮、验证MSE早停10轮。
输入块3000、注意力查询块512，查询仍面对完整图；CUDA显式要求可用GPU。
`cache_samples=false`，有界CPU因子日缓存512MiB，无C盘缓存或磁盘预处理缓存。

模型权重和诊断在：
`data/model_workspace/analyst_coverage_gat_40d/artifacts/<window_id>/`。
最终因子列合并写入：
`data/factors/stock/daily/<year>/<YYYYMMDD>_ml.parquet` 的 `analyst_coverage_gat_40d`。
未覆盖或历史不足的股票没有有效分数。
metadata由公共factor入口写入，包含e2e/model_generated及HTZQ等标签；
不会将中间三列或权重写入 `data/models`。

## CLI

仓库根目录运行（开发期间未执行）：

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')

# Both commands include all input dates required for 2013 onward training.
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-analyst-gat-inputs `
  --asset stock --start-date 20130101 --end-date 20260424 --date-batch-size 120

cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-analyst-coverage `
  --asset stock --start-date 20130101 --end-date 20260424 --window-months 6 --date-batch-size 20

& $py -B -u -m yq_ml_alpha factor-train --config ml_alpha\factors\analyst_coverage_gat_40d.toml
& $py -B -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\analyst_coverage_gat_40d.toml
# Combined: factor-run --config ml_alpha\factors\analyst_coverage_gat_40d.toml
```

首次行情派生会自动读取2012年的40日预热；季度图入口自动补2012年末快照，
该快照还需此前半年报告。已有从2016年开始的季度图，需要补齐2013–2015年。
重建更早季度数据不改变稳定关系身份或原 `mdl_000035` 的输入定义。

服务器环境准备好后，将路径分隔符换为 `/`，使用同样的Rust及factor命令。
本次只进行静态检查，不编译Rust、不运行测试、派生生成、训练或预测。
