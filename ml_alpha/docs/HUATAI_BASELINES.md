# Huatai baseline models / 华泰基础模型

The report families are adapted to the local formal factor library. Existing
architectures are reused; classification-only objectives retain the reports'
return-tail targets. This is not a replication of the reports' private 70-factor
library, monthly datasets or reported backtest results.

本次从 `mdl_000006` 插入基础模型。已有 OLS、Lasso、Ridge、ElasticNet、PCA+OLS、
随机森林、MLP、RNN、GRU、LSTM、XGBoost 均复用代码，不另写同构实现。
可做收益回归的模型使用回归；逻辑回归、hinge/modified-Huber 分类目标和
GaussianNB/LDA/QDA 保留分类接口。LDA/QDA 是判别分析，并非朴素贝叶斯。

## Catalog / 编号

| ID | Model / 模型 | Objective and output / 目标与输出 |
| --- | --- | --- |
| `mdl_000001`–`mdl_000005` | OLS / Lasso / Ridge / ElasticNet / PCA+OLS | 原配置不变 |
| `mdl_000006` | Logistic regression | 尾部二分类，正类概率 |
| `mdl_000007` | SGD linear SVM, hinge loss | 尾部二分类，正类方向的 decision margin |
| `mdl_000008` | SGD modified Huber | 尾部二分类，正类概率 |
| `mdl_000009` | RBF SVR | 10 日 VWAP 收益回归 |
| `mdl_000010` | Gaussian Naive Bayes | 尾部二分类，正类概率 |
| `mdl_000011` | LDA | 尾部二分类，正类概率 |
| `mdl_000012` | QDA | 尾部二分类，正类概率 |
| `mdl_000013` | Random forest | 10 日 VWAP 收益回归，复用 `RandomForestAlphaModel` |
| `mdl_000014` | AdaBoost regression trees | 10 日 VWAP 收益回归 |
| `mdl_000015` | Classic gradient boosting trees | 10 日 VWAP 收益回归；不是 Histogram GBDT |
| `mdl_000016` | MLP, F → 40 → 10 → 1 | Tanh 隐藏层，线性回归头，MSE |
| `mdl_000017` | RNN, 2 × 100 | 连续 5 个交易日因子序列，MSE |
| `mdl_000018` | GRU, 2 × 100 | 连续 5 个交易日因子序列，MSE |
| `mdl_000019`–`mdl_000026` | Existing models / 旧模型 | 只迁移编号，见下表 |

Classifier scores rank stocks in the same daily signal table as regressors.
They are not estimates of returns. There is no automatic class threshold during
prediction. `SGDLinearSVMAlphaModel` exposes a margin rather than pretending it
is a calibrated probability. Kernel selection for SVR is configurable through
`kernel`, `degree`, `gamma` and `coef0`, not separate copies of the implementation.

## Training contract / 训练口径

Only the newly inserted `000006`–`000018` configs adopt the following scheme:

- 输入：全部有效正式日因子，复用因子自动发现、排除模型生成及 deprecated 列的规则。
- 股票池：`mkt_all`，剔除 ST 和北交所，不因涨跌停删除样本。
- 特征：每个日期截面做中位数 ± 5 MAD 截尾、z-score，缺失填 0。
  `mad5_zscore_features` **不改变收益标签**；没有新增磁盘预处理缓存。
- 标签：`future_vwap_return_10d`，复权 VWAP(t+11) / VWAP(t+1) − 1。
- 窗口：训练日历范围 2016-01-01 至 2025-12-31；每年末重训，回看 36 个月，
  前 33 个月训练、后 3 个月验证，两个区间分别每 5 个交易日采样。
- 公共层剔除跨越验证边界或训练截止日的标签，`lookahead_days=11`。
  T 日收盘训练时，最新可用截面不晚于 T−11；有效采样日可能更早。
- 配置的预测范围为 2019-01-01 至 2026-04-24，每日输出；实际窗口仍取决于
  数据、交易日历和完整训练窗口的可用性。
- 单种子 42；不做 Optuna 搜索。神经网络最多 100 个 epoch、验证 MSE 连续
  10 轮不改善早停，batch_size=1000；传统估计器每个窗口拟合一次训练分区，
  验证分区仅供诊断，不混入最终拟合。

分类标签由公共 `data/targets.py:cross_section_tail_labels` 在模型拟合时生成：
每个采样日独立对可用收益排序，前 30% 为 1，后 30% 为 0，中间不进入分类训练。
验证集按自己的日期独立生成标签。并列收益使用平均名次的中点百分位，避免按
股票代码强行打破并列，因此恰好落在边界的同收益股票可能使类别数量偏离 30%。
缺失收益不生成类别。预测入口既不读取未来收益，也不使用尾部筛选，覆盖所有合格股票。

RNN/GRU 复用现有序列加载器：以采样日为终点取连续 5 个交易日，而非取 5 个
间隔 5 日的截面。报告原版是 5 个月，这里沿用项目日频因子序列。输入/输出维度
随本地因子数量变化；两层之间及最后一层输出各使用 0.2 Dropout，RMSprop lr=0.001。
MLP 使用 Adam lr=0.001、Dropout=0.2。MLP 的训练/验证及两种网络的预测都分批
前向；完整输入保留在 CPU。旧 MLP 默认仍为 ReLU，旧序列模型默认仍为 Adam、
output_dropout=0，checkpoint 的有参数层名称保持不变。

Traditional sklearn estimators run on CPU; MLP/RNN/GRU use `device="auto"`.
SVR uses the exact kernel implementation, which can be slow on large training
pools. `cache_size=512` limits its **in-memory** kernel cache in MiB, not a disk
directory. No automatic sampling, kernel approximation or GPU conversion is
introduced. Random forest uses eight CPU workers; no server-specific paths are
hard-coded into models.

## Report alignment / 研报对应与工程默认值

| Source / 原报告 | Reused or added / 接入范围 |
| --- | --- |
| 华泰人工智能之2《人工智能选股之广义线性模型》，2017-06-22，p12–14、p29 | 原有四种线性回归；补 Logistic、SGD hinge、SGD modified Huber |
| 之3《人工智能选股之支持向量机模型》，2017-08-04，p12–15、p25 | 回归优先，使用报告讨论的 RBF SVR；其他核通过参数选择 |
| 之4《人工智能选股之朴素贝叶斯模型》，2017-08-17，p10–13 | GaussianNB 及报告比较的 LDA/QDA；均保留分类 |
| 之5《人工智能选股之随机森林模型》，2017-08-31，p11、p14–15 | 复用随机森林，分类改为收益回归 |
| 之6《人工智能选股之Boosting模型》，2017-09-11，p14、p17 | 补 AdaBoost 和传统 GBDT；已有 XGBoost 顺延后复用 |
| 之7《人工智能选股之Python实战》，2017-09-19 | 实现示例，不另设重复模型 |
| 之8《人工智能选股之全连接神经网络》，2017-11-23，p16–18 | 复用 MLP，保留 40/10 隐层与 tanh，将三分类头改为标量回归头 |
| 之9《人工智能选股之循环神经网络模型》，2017-11-24，p13–16 | 复用 RNN/GRU/LSTM，补 RNN/GRU 配置，已有 LSTM 只顺延 |

参数不应理解为已在本地调优：Logistic 的 C=0.001 是采用 sklearn 惩罚系数约定
对报告惩罚强度的配置；SGD 的 alpha 分别为 0.1/1.0。SVR 的 C=1、gamma=0.01、
epsilon=0.001 是本地默认值，其中 epsilon 对应收益率小数单位的 0.1 个百分点。
LDA 使用自动收缩协方差、QDA 使用 reg_param=0.1，是对高维本地输入的工程设置。
RF 保留 500 棵树、叶子至少 10 个样本、分裂至少 50 个样本，但候选特征数用
sqrt(F) 适配本地特征数；AdaBoost 100 棵深度 3 树、lr=0.05；GBDT 300 棵深度 3
树、lr=0.03、subsample=0.8。这些固定默认值不是研报最优参数或已验证的收益结论。

各篇报告中的专有因子清单、行业均值填缺、行业市值中性化、月频样本和交叉验证
未被暗中加入本地公共流程。可选 `pca_components` 由传统模型适配器支持，仅在训练
样本上拟合并随模型保存，默认不开启。分类验证输出 AUC，回归输出 MSE；组合收益
和 IC 仍应使用项目独立回测入口。

## Migration / 旧编号迁移

| Old ID | New ID | Model |
| --- | --- | --- |
| `mdl_000006` | `mdl_000019` | LSTM |
| `mdl_000007` | `mdl_000020` | LightGBM + Optuna |
| `mdl_000008` | `mdl_000021` | XGBoost + Optuna |
| `mdl_000009` | `mdl_000022` | XGBoost fixed 600 trees |
| `mdl_000010` | `mdl_000023` | Single-gate adaptive MLP |
| `mdl_000011` | `mdl_000024` | GNN + GRU + two XGBoost rankers |
| `mdl_000012` | `mdl_000025` | HIST |
| `mdl_000013` | `mdl_000026` | FactorVAE |

The migrated models keep their existing hyperparameters and training schemes.
Model IDs, artifact directories, registry entries, production strategy signal
references and documentation move together. There are no aliases from the old
IDs, because those IDs now mean different models. Existing artifacts must not
be renamed and loaded under the new configs: regenerate weights/manifests and
predictions for the new identity. Keep any externally deployed legacy data
separate before using the inserted configs.

本次检查 `xz336:/home/wym/YuminQuant2026` 未发现 data 目录、外部数据目录软链接或
config.toml。已将本地旧 `mdl_000009` workspace、仅含该模型分数的 `data/models`
和旧 `000008/000009` 回测目录移到回收站。原始数据、因子、标签与端到端产物保留。
这不是训练时自动清理数据的逻辑。

## Commands and output / 命令与产物

Run from the repository root. For example, train/predict the new MLP locally:

```powershell
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
$env:PYTHONPATH = (Join-Path (Get-Location) 'ml_alpha')
& $py -B -u -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000016.toml
& $py -B -u -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000016.toml
# Combined entry point:
# & $py -B -u -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000016.toml
```

服务器使用相同相对路径配置。在已经准备数据及兼容依赖的 Python 环境中：

```bash
cd /home/wym/YuminQuant2026
export PYTHONPATH="$PWD/ml_alpha${PYTHONPATH:+:$PYTHONPATH}"
python -B -u -m yq_ml_alpha model-run --config ml_alpha/models/mdl_000016.toml
```

检查时服务器 Miniforge base 为 Python 3.13，尚未安装 sklearn/torch，仓库尚无数据。
本次仅同步代码，不安装依赖或启动训练。上面的服务器命令必须在完成环境和数据
准备后执行，不能据此视为服务器已通过训练验证。

- 权重、manifest、诊断：`data/model_workspace/<model_id>/artifacts/<window_id>/`。
- 预测：`data/models/<YYYY>/<YYYYMMDD>.parquet`，键为 `trade_date`、`ts_code`，
  每个模型占一列 `mdl_XXXXXX`，float32。分类概率、margin 和预测收益共用此格式。
- `cache_samples=false`；所有配置路径均相对项目，未新增 C 盘缓存操作。
- 回测单独运行，输出仍放 `data/backtest/stock/daily`。多个预测进程不要并发覆盖
  同一批每日文件，使用现有顺序预测流程。
