# yq_ml_alpha / Python ML Alpha

`ml_alpha` 是 YuminQuant 的 Python 机器学习 alpha 包。它读取正式因子库或外部 alpha root，按配置切分训练/验证/预测窗口，训练模型，并把预测分数写成标准日频 alpha parquet。

`ml_alpha` is the Python ML alpha package. It reads formal factors or external alpha roots, builds train/valid/predict windows from TOML configs, trains models, and writes predictions as standard daily alpha parquet files.

## 快速开始 / Quick Start

不安装包时，从 `ml_alpha` 目录运行：

Run from the `ml_alpha` directory when you do not want to install the package:

```powershell
cd ml_alpha
python -m yq_ml_alpha run --config configs\mdl_000001.toml
python -m yq_ml_alpha run --config configs\mdl_000006.toml
python -m yq_ml_alpha run --config configs\monthly_xgb_36.toml
python -m yq_ml_alpha run --config configs\monthly_mlp_36.toml
python -m yq_ml_alpha run --config configs\monthly_elstm_ranknet_36.toml
```

Python 3.8.3 GPU environment on this machine:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000006.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000007.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000008.toml
```

`D:\Users\Devin\anaconda383\python.exe` is Python 3.8.3 with PyTorch, CUDA,
pandas, and pyarrow available. Running from `ml_alpha` avoids changing
`PYTHONPATH` or global environment variables.


如果想从仓库根目录直接运行，可以临时设置 `PYTHONPATH`：

If running from the repository root, set `PYTHONPATH` temporarily:

```powershell
$env:PYTHONPATH = "D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha"
python -m yq_ml_alpha run --config ml_alpha\configs\monthly_mlp_36.toml
```

可用子命令 / Commands:

```powershell
python -m yq_ml_alpha run --config configs\monthly_mlp_36.toml
python -m yq_ml_alpha train --config configs\monthly_mlp_36.toml
python -m yq_ml_alpha predict --config configs\monthly_mlp_36.toml
python -m yq_ml_alpha materialize --config configs\monthly_mlp_36.toml
```

`run` = train + predict + write alpha. `train` only saves model artifacts. `predict` uses existing artifacts. `materialize` only builds sample cache when configured.

## 输出与回测 / Output And Backtest

Alpha 输出 / Alpha output:

```text
data/models/{year}/{trade_date}.parquet
columns: trade_date, ts_code, alpha_id
```

同一天多个 alpha 会写入同一个 daily parquet。Parquet 不能真正原地追加列，因此 writer 会读取旧文件、合并/覆盖当前 alpha 列，再重写文件。

Multiple alphas for the same date are stored in one daily parquet. Parquet cannot append a column in place, so the writer reads, merges, and rewrites the file.

回测 / Backtest:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --groups 10 --rebalance 20
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_monthly_alpha --factor-root data\models --factor-fill ffill
```

低频 alpha 只在月末或周末有截面时，使用 `--factor-fill ffill` 让回测用最近一次 alpha 日频结算。

Use `--factor-fill ffill` for low-frequency alpha snapshots.

## 配置结构 / TOML Config

所有示例配置都在：

All configs live under:

```text
ml_alpha/configs/*.toml
```

Numbered production model configs use `mdl_******` ids. The registry file
`ml_alpha/model_registry.toml` records the model id, output alpha id, config
path, model class, description, feature source, preprocessing, and tags.

典型结构 / Typical shape:

```toml
run_id = "monthly_mlp_36"
alpha_id = "ml_alpha_mlp"
data_root = "data"
output_root = "data/models"

[dates]
train = [20110101, 20260424]
valid = []
predict = [20110101, 20260424]

[sample]
train_frequency = "monthly_end"
predict_frequency = "daily"

[train_scheme]
type = "rolling"              # static | expanding | rolling
refit_frequency = "monthly_end"
train_sample_count = 36
validation_sample_count = 1

[label]
id = "future_vwap_return_20d"

[filters]
exclude_limit = false
exclude_st = true
exclude_bj = true

[preprocess]
cross_section_transform = "rank_gauss"
feature_fill_value = 0.0

[features]
type = "factor_frame"         # factor_frame | raw_panel | bar_panel
root = "data/factors/stock/daily"
columns = "__all__"

[model]
name = "mlp"
class = "yq_ml_alpha.models.mlp_model.MLPAlphaModel"
artifact_dir = "data/model_workspace/monthly_mlp_36/artifacts"
```

常用采样频率 / Sampling frequencies:

```text
daily
weekly
monthly_end
quarterly
5
20
every_5_days
```

`valid = []` 表示不使用固定验证区间。滚动训练里的 `validation_sample_count = 1` 仍会用训练窗口之后第一个样本截面做动态验证。

`valid = []` means no fixed validation period. In rolling mode, `validation_sample_count = 1` can still create a dynamic validation sample after the training window.

### 滚动训练续跑 / Resuming Rolling Training

如果一次 rolling 训练中途暂停，通常**不要**为了续跑而把 `train` 改成暂停日期附近。`train` 是训练样本池范围，程序真正读取哪些训练截面由每个 refit window 的 `train_dates` 决定。对于 `train_sample_count = 36`、`validation_sample_count = 1` 这类配置，每个窗口只会从 `train` 样本池中取 refit 日期之前最近的 36 个训练截面和 1 个验证截面。

续跑时应主要调整 `predict` 区间，让它从“下一段预测所需的 refit anchor”开始。例如已经预测完 `20231229`，下一段需要预测 2024 年 1 月，月频 refit 下建议：

```toml
[dates]
train = [20110101, 20260424]   # 保留足够历史样本池
valid = []
predict = [20231229, 20260424] # 20231229 是下一段预测的 refit anchor
```

当前实现中，`refit_date` 本身不会被该窗口重新预测；窗口预测的是 `refit_date` 之后到下一个 refit date 之间的交易日。因此上面的配置会从 `20240102` 开始写后续 alpha，同时保留足够历史样本用于 rolling 训练。

When resuming an interrupted rolling run, usually do **not** shrink `train` to the interrupted date. `train` is the sample pool. The actual training data is selected per refit window from `window.train_dates`. With `train_sample_count = 36` and `validation_sample_count = 1`, each window uses only the latest 36 training snapshots plus one validation snapshot before the refit date.

To resume, adjust `predict` to start from the refit anchor that owns the next unfinished prediction segment. If predictions are complete through `20231229` and the next segment is January 2024, keep `train` broad and set `predict = [20231229, 20260424]`. The refit date itself is not predicted again; predictions start after it.

## 预处理 / Preprocessing

当前推荐截面变换是：

Recommended cross-sectional transform:

```toml
[preprocess]
cross_section_transform = "rank_gauss"
feature_fill_value = 0.0
```

`rank_gauss` 做：

`rank_gauss` applies:

```text
rank -> (rank - 0.5) / n -> inverse normal CDF -> cross-section zscore
```

feature 缺失值在变换后填 `feature_fill_value`，label 缺失不会填充，训练时缺 label 的样本会被剔除。

Feature missing values are filled after transform. Label missing values are not filled and are dropped for training.

可用 transform 在 `yq_ml_alpha/features/transforms.py` 注册。新增 transform 时只需要注册一次，然后 TOML 中写注册名。

Transforms are registered in `yq_ml_alpha/features/transforms.py`. Add a transform once and reference its registered name in TOML.

## 已有模型 / Built-In Models

```text
LinearRegressionAlphaModel                  mdl_000001.toml
XGBoostAlphaModel                           monthly_xgb_36.toml
XGBoostOptunaAlphaModel                     monthly_xgb_optuna_36.toml
LightGBMOptunaAlphaModel                    monthly_lgbm_optuna_36.toml
LassoAlphaModel                             mdl_000002.toml
RidgeAlphaModel                             mdl_000003.toml
ElasticNetAlphaModel                        mdl_000004.toml
PCAOLSAlphaModel                            mdl_000005.toml
BarGRUAlphaModel                            mdl_000006.toml
MultiBarGRUAlphaModel                       mdl_000007.toml
ResidualMultiBarGRUAlphaModel               mdl_000008.toml
RandomForestAlphaModel                      monthly_rf_36.toml
MLPAlphaModel                               monthly_mlp_36.toml
RNNAlphaModel                               monthly_rnn_36.toml
GRUAlphaModel                               monthly_gru_36.toml
CNNAlphaModel                               monthly_cnn_36.toml
eLSTMRankNetAlphaModel                      monthly_elstm_ranknet_36.toml
ICSignEqualWeightAlphaModel                 monthly_ic_sign_equal_weight.toml
MeanFeatureAlphaModel                       mean_combo_smoke.toml
```

深度模型依赖 PyTorch；XGBoost/LightGBM/Optuna 是可选依赖。

Deep models require PyTorch. XGBoost, LightGBM, and Optuna are optional dependencies.

## Sequence 模型输入 / Sequence Model Input

`RNN/LSTM/GRU/eLSTM` 使用 `DatasetBuilder.load_sequence()` 读取过去 `sequence_length` 个样本截面。若 `sequence_length = 6`，训练样本日期是月末，则每个样本包含最近 6 个月末截面的 feature。

`RNN/LSTM/GRU/eLSTM` use `DatasetBuilder.load_sequence()` to load the last `sequence_length` sample dates. With `sequence_length = 6` and monthly samples, each row contains features from the last six month-end snapshots.

进入模型前的形状：

Input shape before model:

```text
flat DataFrame feature matrix: [N, sequence_length * F]
torch tensor:                  [N, sequence_length, F]
```

如果 feature 数不能整除 `sequence_length`，会在右侧补 0 后 reshape。

If feature count is not divisible by `sequence_length`, zeros are padded on the right before reshaping.

## Bar Panel End-to-End GRU / 通用 Bar Panel 端到端模型

### 设计 / Design

`bar_panel` 是端到端量价模型的通用行情输入接口。它负责把原始行情 bar
合成为固定长度的 tensor 特征，但不会把合成后的 bar 作为独立文件落盘。

`bar_panel` is the reusable market-bar input interface for end-to-end models. It
builds fixed-length tensor features from raw bars, but it does not persist the
synthesized bars as standalone files.

分钟源数据流程 / Minute source flow:

```text
读取某一天 1m parquet / read one daily 1m parquet
  -> 合成目标 bar，例如 16 根 15m bar / aggregate into requested bars
  -> 只在进程内 LRU cache 保留合成后的日度 bar / keep aggregated day in cache
  -> 释放原始 1m DataFrame / release raw 1m DataFrame
  -> 读取下一天 / read the next day
```

日频源数据流程 / Daily source flow:

```text
按 trade_date 读取 daily pv parquet / read daily pv parquet by trade_date
  -> 如果 bar_size > 1，合成非重叠 N 日 bar / aggregate non-overlapping N-day bars
  -> 在进程内 cache 保留合成后的 daily panel / keep aggregated daily panel in cache
```

`cache_samples = true` 只会缓存最终训练/预测样本，方便 debug；它不是合成
bar 的持久化缓存。

`cache_samples = true` only caches final train/predict samples for debugging. It
is not a persistent cache of synthesized source bars.

### Single Bar Panel: `mdl_000006`

`mdl_000006` 是单频率 15 分钟 GRU 模型。核心配置如下：

`mdl_000006` is the single-frequency 15-minute GRU model. Core config:

```toml
[features]
type = "bar_panel"
root = "data/stock_data/minute"
columns = ["open", "high", "low", "close", "vwap", "volume"]

[features.params]
source_frequency = "minute"
bar_size = 15
lookback_sessions = 20
time_series_scale = "mean"
strict = true
```

模型输入 / Model input:

```text
[N, 320, 6] = [stocks, 20 trading days * 16 bars, open/high/low/close/vwap/volume]
```

预处理顺序 / Preprocessing order:

1. 对每只股票、每个特征，把 320 步时序值除以自身时序均值；
2. 对每个 `trade_date`，将每个 `time_step x feature` 列做截面 z-score；
3. 对 `future_vwap_return_20d` label 做截面 z-score。

English: divide each stock-feature time series by its own mean, then apply
cross-sectional z-score to each `time_step x feature` column and to the label.

### Multi Bar Panel: `mdl_000007` / `mdl_000008`

`multi_bar_panel` 用来组合多个 `bar_panel`。当前生产配置使用一个日频分支
和一个 15 分钟分支：

`multi_bar_panel` composes multiple `bar_panel` providers. Current production
configs use one daily branch and one 15-minute branch:

```toml
[features]
type = "multi_bar_panel"

[features.panels.daily]
root = "data/stock_data/daily/pv"
source_frequency = "daily"
bar_size = 1
lookback_sessions = 40
time_series_scale = "last"
columns = ["open", "high", "low", "close", "vwap", "volume"]

[features.panels.minute]
root = "data/stock_data/minute"
source_frequency = "minute"
bar_size = 15
lookback_sessions = 20
time_series_scale = "mean"
columns = ["open", "high", "low", "close", "vwap", "volume"]
```

输出列会带分支前缀，例如 `daily__open__t000` 和 `minute__close__t319`。
模型侧按前缀恢复 tensor：

English: output columns are prefixed by branch, such as `daily__open__t000`
and `minute__close__t319`; the model restores tensors by prefix.

```text
daily branch:  [N, 40, 6]
minute branch: [N, 320, 6]
```

`mdl_000007` 是普通多频率混合 GRU：日频分支和分钟分支分别输出 30 维表示，
经过 BatchNorm 后 concat，再通过 FC 映射为一个 score。

`mdl_000007` is the normal multi-frequency GRU. The daily and minute branches
produce 30-dimensional representations, then BatchNorm + concat + FC maps them
to one score.

`mdl_000008` 是参数冻结 + 残差预测版本：

`mdl_000008` is the frozen-parameter residual version:

```text
stage 1: train daily branch only -> y_hat_1
stage 2: freeze daily branch, train minute branch -> y_hat_2
final:   y_hat = y_hat_1 + y_hat_2
```

两个阶段都使用 date-wise negative IC loss。`loss_history.parquet` 会包含
`stage = stage1_daily / stage2_residual`。

Both stages use date-wise negative IC loss. `loss_history.parquet` includes
`stage = stage1_daily / stage2_residual`.

### 训练命令 / Training Commands

默认 Python 环境 / Default Python environment:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

python -m yq_ml_alpha run --config configs\mdl_000006.toml
python -m yq_ml_alpha run --config configs\mdl_000007.toml
python -m yq_ml_alpha run --config configs\mdl_000008.toml
```

Python 3.8.3 GPU 环境 / Python 3.8.3 GPU environment:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha

& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000006.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000007.toml
& D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config configs\mdl_000008.toml
```

### 回测命令 / Backtest Commands

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant

cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors mdl_000006 --factor-root data\models --groups 10 --rebalance 20 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors mdl_000007 --factor-root data\models --groups 10 --rebalance 20 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors mdl_000008 --factor-root data\models --groups 10 --rebalance 20 --date-batch-size 120
```

## Diagnostics / Loss 输出

MLP、RNN、LSTM、GRU、eLSTM RankNet 支持 diagnostics。TOML 中打开：

MLP, RNN, LSTM, GRU, eLSTM RankNet, and BarGRU support diagnostics:

```toml
[diagnostics]
enabled = true
print_epoch = true
write_loss_history = true
write_model_info = true
write_window_summary = true
```

窗口级输出 / Per-window outputs:

```text
data/model_workspace/{run_id}/artifacts/{window_id}/loss_history.parquet
data/model_workspace/{run_id}/artifacts/{window_id}/model_info.json
```

全 run 汇总 / Run-level summaries:

```text
data/model_workspace/{run_id}/diagnostics/loss_history.parquet
data/model_workspace/{run_id}/diagnostics/window_summary.parquet
```

Regularized linear models also write diagnostics when enabled. For
`mdl_000002` / `mdl_000003` / `mdl_000004`, the aggregate
`window_summary.parquet` includes `best_alpha`, `best_l1_ratio` when
applicable, `best_score`, and `best_params_json`.

`mdl_000005` writes PCA diagnostics to the same summary file, including
`n_original_features`, `n_components`, `explained_variance_ratio_sum`, and
`explained_variance_ratio_json`.

`mdl_000006` writes GRU diagnostics to the same paths. `loss_history.parquet`
contains date-wise negative IC loss by epoch, and `model_info.json` records the
15-minute panel shape, train/valid row counts, device, and best epoch.

`loss_history.parquet` 记录每个 epoch 的 `train_loss`、`valid_loss`、`best_loss`、`stale_epochs`、`elapsed_seconds` 等。`model_info.json` 记录样本量、设备、模型参数、best epoch 和 best loss。

`loss_history.parquet` records per-epoch loss. `model_info.json` records data sizes, device, model params, best epoch, and best loss.

## 调参 / Tuning

调参属于模型内部逻辑，不做框架级公共 objective/loss。TOML 中通过 `[model.search]` 和 `[model.search.space]` 暴露搜索空间。

Tuning is model-owned. There is no shared framework-level objective or loss. Search spaces are configured through `[model.search]` and `[model.search.space]`.

示例 / Example:

```toml
[model.search]
enabled = true
method = "random"
n_iter = 40
scoring = "neg_mean_squared_error"

[model.search.space]
alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10]
solver_selection = ["cyclic", "random"]
```

关闭调参时：

Disable tuning:

```toml
[model.search]
enabled = false
```

## 新增模型 / Add A Model

新增文件：

Create a model file:

```text
ml_alpha/yq_ml_alpha/models/my_model.py
```

实现接口 / Implement:

```python
from yq_ml_alpha.models.base import AlphaModel, ModelContext

class MyAlphaModel(AlphaModel):
    def fit(self, train_data, valid_data, context: ModelContext) -> None:
        ...

    def predict(self, data, context: ModelContext):
        ...

    def save(self, path):
        ...

    @classmethod
    def load(cls, path):
        ...
```

TOML 中指向类路径：

Reference the class path in TOML:

```toml
[model]
name = "my_model"
class = "yq_ml_alpha.models.my_model.MyAlphaModel"
artifact_dir = "data/model_workspace/my_run/artifacts"

[model.params]
learning_rate = 0.03
```

训练管线会动态 import 该类。模型自己的 loss、调参、early stopping 和 artifact 结构由模型内部决定。

The training pipeline dynamically imports the class. Loss, tuning, early stopping, and artifacts are model-owned.

## IC Sign Equal Weight 模型

`ICSignEqualWeightAlphaModel` 读取 Rust 回测输出的 RankIC 序列，使用 `sign(mean(rank_ic))` 调整每个 feature 方向，再对有效 feature 等权平均：

`ICSignEqualWeightAlphaModel` reads RankIC history from Rust backtest outputs, orients each feature by `sign(mean(rank_ic))`, then averages valid features:

```toml
[model]
class = "yq_ml_alpha.models.ic_sign_model.ICSignEqualWeightAlphaModel"

[model.params]
ic_root = "data/backtest/stock/daily/ic"
ic_metric = "rank_ic"
```

缺 IC 文件、RankIC 全空或均值为 0 的 feature 会被剔除。

Features with missing or invalid IC files are dropped.

## 维护提示 / Maintenance Notes

- 当前配置都在 `ml_alpha/configs/*.toml`。
- `data/models` 是正式 ML alpha 输出根目录。
- `data/model_workspace/{run_id}` 是模型 artifact、diagnostics 和可选 cache 目录。
- `rank_gauss` 是当前推荐预处理 transform。
- 输出 alpha 不写入 Rust factor metadata，回测时用 `--factor-root data\models --factors alpha_id`。

- Current configs live under `ml_alpha/configs/*.toml`.
- `data/models` is the formal ML alpha output root.
- `data/model_workspace/{run_id}` stores artifacts, diagnostics, and optional cache.
- `rank_gauss` is the recommended transform.
- ML alpha is not written into Rust factor metadata; use `--factor-root data\models`.
