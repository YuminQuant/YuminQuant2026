# ml_alpha

`ml_alpha` separates model experiments from formal end-to-end factors.

## Developer Handoff / 开发交接

See [Development Workflow (中文 / English)](docs/DEVELOPMENT_README.md) before
adding or changing a model, feature provider, derived dataset, or factor config.
It covers directory ownership, runnable commands, TOML configuration, model
interfaces, performance conventions, leakage controls, and the handoff checklist.

新增或修改模型、特征、派生数据和因子配置前，请先阅读上述双语开发文档。
本文保留现有功能的详细说明；开发流程与交接清单集中在该文档中。

## Configuration Layers

- `models/*.toml`: model or experiment configs. These write signals to `data/models`.
- `factors/*.toml`: formal single-factor configs. The TOML file name, `factor_id`, output column, and factor metadata identifier must all use the same semantic snake_case factor name.

Current formal factor configs:

- `factors/resnet_gru_week.toml` / `factors/resnet_gru_msnew.toml`（见 [ResNet-GRU 数据与命令](docs/RESNET_GRU.md)）

- `factors/bar_gru_15m.toml`
- `factors/multi_bar_gru_daily_15m.toml`
- `factors/residual_multi_bar_gru.toml`
- `factors/gru_patch1.toml`
- `factors/gru_patch2.toml`
- `factors/rawbar_agru.toml`
- `factors/rawbar_lstm.toml`
- `factors/rawbar_multi_gru.toml`（64 个内部因子等权输出，见 [说明](docs/MULTIVARIATE_GRU.md)）
- `factors/logsig_alpha_v.toml`

`mdl_*` is reserved for model-layer configs. `e2e_fct_*` is no longer a valid formal factor identifier.

## CLI

`models/mdl_000010.toml` adds a single-gate adaptive-depth factor combination
model using the existing daily factor library. It is a model-layer signal,
not an end-to-end factor. See [adaptive network configuration and CLI](docs/ADAPTIVE_GATE.md).

### GPU Optuna models

`models/mdl_000007.toml` is LightGBM + Optuna; `models/mdl_000008.toml`
is XGBoost + Optuna. Both use all formal daily factors, cross-sectional
`rank_gauss`, training and validation samples every 5 trading days, a 5-day future VWAP return label,
and at most 20 TPE trials minimizing validation MSE. Both search 300-800 trees in
steps of 100. LightGBM fits once (`static`); XGBoost refits quarterly.
Each model implements trial-level early stopping in its own module: stop after
5 consecutive completed trials without at least 0.01% relative MSE reduction
from the last meaningful best value. Small reductions accumulate against that
reference. The first completed trial establishes the reference; no warmup
exemption is applied. Optuna still selects the absolute best trial for final
refitting. Configure `early_stopping_patience` and `min_relative_improvement`
under `[model.search]`. This does not stop individual boosting fits early.
LightGBM tunes using 2016–2021 training and 2022 validation, then
`model.search.refit_on_validation = true` refits the best parameters on the
combined training and validation samples. Daily prediction covers 2023-01-01
through 2026-04-24. The existing runtime removes samples whose labels cross
split boundaries (6 trading days for this label); these purged samples are
also absent from the final refit. No random validation split is used here.

XGBoost (`mdl_000008`) uses `rolling`, `quarterly_end`, `train_lookback="24m"`,
`validation_ratio=0.2` and `validation_unit="rows"`. Public runtime settings
`start_from_features=true` and `prediction_activation="after_label_maturity"`
start the first complete 24-calendar-month pool in the first month containing
usable factor data within the configured training bounds (currently January
2016, first nonempty factor date 2016-01-04). The first pool is 2016-2017.
Every quarter, sample the trailing 24 months every 5 trading days, including
only sampled dates whose labels have matured by quarter-end close. Load this pool
once, apply stock filters and remove missing labels, then sort by trade_date
and ts_code. The first floor(N * 0.8) rows train the candidate models and the
remaining rows validate them. Do not split the date list 80/20. A trading date
may appear on both sides of this row boundary; no additional train/validation
date purge is applied in row mode. The out-of-sample label maturity protection
below remains active. Final refitting uses both row partitions.

公共层 `validation_unit="rows"` 支持模型和因子训练入口，默认 `"dates"` 保留其他配置的旧语义。
24 个月是训练与验证共用的总窗口，不是训练 24 个月后再追加验证期。
行数划分发生在过滤及标签缺失行剔除后，不随机打乱；整数边界使验证行数约为 20%。
这种按行划分不保证训练与验证之间的时间隔离；同日股票或重叠标签区间可能跨两侧。
样本外预测仍独立遵守标签到期边界。manifest 会记录行划分规则及实际行数、边界股票。

`mdl_000009` uses the ordinary `XGBoostAlphaModel` with the same 24-month,
10-day sampling and `future_vwap_return_10d` labels, and refits at each month-end (`monthly_end`). It fixes 600 trees, learning rate 0.001,
depth 6, min_child_weight 10, row/column subsampling 0.8, L1 0.1 and L2 10.
There is no Optuna search. `model.params.fit_on_validation=true` combines the
row-based train/validation partitions before one final fit, preserving 008's
final-fit pooling approach without a preliminary fit or per-round evaluation.
The 20% partition is therefore not a held-out performance estimate for 009.
Existing ordinary XGBoost configs retain their behavior when this flag is absent.

All model and factor training uses an end-of-day information cutoff T. Both
training and validation rows must have labels fully observed by T close.
For N-day VWAP returns, the last eligible feature date is T-(N+1) trading
sessions: T-11 for 009 and T-6 for 008. Sampling may end earlier. Each window
samples its trailing calendar-month range, then removes immature dates.
The legacy `after_label_maturity` setting now fits only mature labels at the
scheduled cutoff and activates the model on the next trading session. The
preceding model owns dates before activation; initial warmup dates stay null.
This public rule also applies to static and train-only windows. Old model
manifests must be retrained after this policy change.

Generate the 10-day adjusted VWAP label from the repository root:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-run `
  --asset stock --frequency daily --start-date 20160101 --end-date 20260424 `
  --labels future_vwap_return_10d
```

The label is adjusted VWAP(t+11) / adjusted VWAP(t+1) - 1. Dates without
available future prices remain missing; the CLI does not start model training.

This policy works in both model and factor pipelines; it currently supports
monthly/quarterly rolling or expanding ratio-split windows. Unsupported
combinations are rejected. Other configs retain the existing `purge` policy.
Saved manifests and window logs record training cutoff and activation dates.

The shared tabular `DatasetBuilder.load` prints flushed per-date start/done
logs for labeled data. Model and factor pipelines pass `split=train` or
`split=valid`; row mode uses `split=train_valid_pool` and then reports the actual
train/validation row counts and date spans. Logs include date progress, usable rows after filters and
missing-label removal, cumulative usable rows, per-date seconds and total
elapsed seconds. Empty dates report zero rows; completion includes concatenation
and optional cache-writing time. Existing sequence/bar-panel logging is unchanged.

Use the existing Python 3.8.3 environment. Its DLL directories must be on PATH:

```powershell
Set-Location D:\yuminwu_workspace\Internship\YuminQuant
$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH

& $py -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000008.toml
& $py -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000008.toml

& $py -m yq_ml_alpha model-train --config ml_alpha\models\mdl_000007.toml
& $py -m yq_ml_alpha model-predict --config ml_alpha\models\mdl_000007.toml
```

`model-run` combines training and prediction. XGBoost uses `gpu_hist`,
`gpu_id=0`, `gpu_predictor`, matching the installed XGBoost 0.90 API.
LightGBM explicitly requires `device_type=gpu` (OpenCL); if unavailable,
skip its training, never change it to CPU. As inspected on 2026-10-17,
Optuna 4.5.0 and LightGBM 3.3.5 are installed in Python 3.8.3.
LightGBM's OpenCL GPU trainer identified the NVIDIA GeForce RTX 3060 Laptop
GPU and completed fitting and prediction. XGBoost GPU fitting and prediction
also passed. LightGBM 4.6.0 was incompatible with this environment's existing
scikit-learn 0.23.1/joblib 0.16.0; retain the verified LightGBM 3.3.5 version.
The installation bypass used process-local `NO_PROXY`, leaving the Windows
system proxy unchanged. Pip's final environment inspection reported the
pre-existing invalid pyodbc version `4.0.0-unsupported`; the installed Optuna,
SQLAlchemy and Alembic imports and study execution were separately verified.

Outputs (paths relative to repository root). Window IDs are `static` for
LightGBM and `cutoff_<training_cutoff>_active_<activation_date>` for XGBoost:

- `data/models/<YYYY>/<YYYYMMDD>.parquet`: daily wide table keyed by
  `trade_date` (int32), `ts_code`; each model adds its own float32 column
  `mdl_000007` / `mdl_000008`. Other model columns are retained. Run the
  two prediction jobs sequentially because they share daily files.
- `data/model_workspace/<model_id>/artifacts/<window_id>/model.pkl`: pickled
  model wrapper, fitted estimator, `best_params_`, `best_value_` and
  `study_summary_` (best validation MSE, searched parameters, trial count).
- `data/model_workspace/<model_id>/artifacts/<window_id>/model.manifest.json`:
  configuration, feature and training-window identity used for loading checks.
- `cache_samples=false`: no sample-cache output by default. Optuna studies
  run in memory; no SQLite study database or full trial-history file is saved.

The numbered model catalog now comprises OLS, Lasso, Ridge, ElasticNet,
PCA+OLS, LSTM, LightGBM+Optuna and XGBoost+Optuna (`mdl_000001`–`mdl_000008`).
Additional implementations remain available: untuned LightGBM/XGBoost,
RandomForest, MLP, RNN, GRU, CNN, eLSTM+RankNet and generic sklearn adapters.
IC-sign equal weighting and feature-mean baselines are not learned ML models.
BarGRU, MultiBarGRU, ResidualMultiBarGRU and LogsigOrthogonalMLP are currently
configured in the formal factor layer. Removing the six legacy monthly
example configs does not remove their model implementations.

Use explicit model/factor commands only. Run these from the repository root
with the existing Python 3.8.3 environment; do not execute every example blindly:

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
& $py -m yq_ml_alpha model-run --config ml_alpha\models\mdl_000001.toml
& $py -m yq_ml_alpha factor-run --config ml_alpha\factors\bar_gru_15m.toml
# Produce logsignature derived features separately before training.
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-logsig `
  --asset stock --start-date 20110101 --end-date 20260424 --threads 2
& $py -m yq_ml_alpha factor-run --config ml_alpha\factors\logsig_alpha_v.toml
& $py -m yq_ml_alpha factor-metadata
& $py -m yq_ml_alpha factor-metadata-all
```

Available commands:

- `model-run`, `model-train`, `model-predict`, `model-materialize`
- `factor-run`, `factor-train`, `factor-predict`, `factor-materialize`, `factor-metadata`, `factor-metadata-all`

The old generic `run/train/predict/materialize` CLI commands have been removed.

## Bar Sequence Tensor Provider / Bar 序列 Tensor Provider

English:

`bar_panel` and `multi_bar_panel` now use a tensor data path for end-to-end sequence factors such as `bar_gru_15m`, `multi_bar_gru_daily_15m`, and `residual_multi_bar_gru`.

Before this change, each target date was materialized as a very wide pandas frame:

- read each source bar session into a per-day wide frame;
- rename each lookback day into `feature__t000`, `feature__t001`, ... columns;
- merge all lookback days into one `N x (T * F)` frame;
- concatenate train/valid/predict dates into a larger wide frame;
- each GRU batch selected those wide columns, cast them to `float32`, converted to NumPy, and reshaped to `[N, T, F]`.

Now the bar providers keep only row metadata in pandas and build tensors directly:

- `DatasetBundle.frame` contains row keys and labels only, such as `trade_date`, `ts_code`, and the label column;
- `DatasetBundle.tensors["bar"]` stores a single-bar panel as `[N, total_steps, input_size]`;
- `DatasetBundle.tensors["daily"]` and `DatasetBundle.tensors["minute"]` store multi-bar branches separately;
- cross-sectional preprocessing uses `float64` for calculations and stores final model tensors as `float32`;
- sequence models consume the bundle through `fit_bundle()` and `predict_bundle()` without repeatedly rebuilding tensors from a wide DataFrame.

The cache setting for sequence bar factors is now:

```toml
max_cache_sessions = "auto"
```

`auto` is computed from the target-date stride and lookback length:

```text
cache_sessions = max(0, lookback_sessions - min_target_stride_sessions)
```

For example, a 20-session lookback sampled every 5 trading days caches 15 sessions; daily prediction caches 19 sessions.

中文：

`bar_panel` 和 `multi_bar_panel` 现在为端到端序列因子使用 tensor 数据通路，例如 `bar_gru_15m`、`multi_bar_gru_daily_15m`、`residual_multi_bar_gru`。

修改前，每个目标日会被物化成一张很宽的 pandas 表：

- 读取每个 source bar session，先转成单日宽表；
- 把 lookback 中每一天重命名为 `feature__t000`、`feature__t001` 等列；
- 把所有 lookback 日期 merge 成一张 `N x (T * F)` 宽表；
- 再把 train/valid/predict 多个目标日 concat 成更大的宽表；
- GRU 每个 batch 再从宽表取列、转 `float32`、转 NumPy，并 reshape 成 `[N, T, F]`。

修改后，bar provider 只把行级信息留在 pandas 中，序列特征直接构造成 tensor：

- `DatasetBundle.frame` 只保存 `trade_date`、`ts_code`、label 等行级 metadata；
- 单 bar panel 放在 `DatasetBundle.tensors["bar"]`，形状为 `[N, total_steps, input_size]`；
- 混频模型的 daily/minute 分支分别放在 `DatasetBundle.tensors["daily"]` 和 `DatasetBundle.tensors["minute"]`；
- 截面预处理计算时使用 `float64`，最终交给模型的 tensor 存为 `float32`；
- 序列模型通过 `fit_bundle()` / `predict_bundle()` 直接消费 tensor，不再每个 batch 反复从宽表重建 tensor。

序列 bar 因子的缓存配置现在使用：

```toml
max_cache_sessions = "auto"
```

`auto` 按目标日采样间隔和 lookback 自动计算：

```text
cache_sessions = max(0, lookback_sessions - min_target_stride_sessions)
```

例如，20 日 lookback、每 5 个交易日采样时缓存 15 个 session；日频预测时缓存 19 个 session。

Benefits / 好处：

- Lower memory use: the long-lived training object no longer stores thousands of wide feature columns in pandas.
- Less repeated conversion: GRU batches no longer repeatedly execute wide-frame `astype("float32").to_numpy().reshape(...)`.
- Cleaner multi-frequency layout: daily and minute branches stay as separate tensors instead of being merged into one giant frame.
- More predictable cache: `"auto"` follows actual reuse instead of retaining an arbitrary fixed number such as 120 sessions.
- No impact on tabular factors: `logsig_alpha_v`, raw panel, factor frame, and monthly tabular models still use their existing data paths.

- 更低内存占用：训练对象不再长期持有几千列 pandas 宽表。
- 更少重复转换：GRU batch 不再反复执行宽表到 `float32` NumPy tensor 的转换。
- 更清晰的混频结构：daily 和 minute 分支保持为独立 tensor，而不是合并成巨型宽表。
- 更可解释的缓存：`"auto"` 跟随实际复用窗口，不再固定保留 120 个 session。
- 不影响 tabular 因子：`logsig_alpha_v`、raw panel、factor frame、monthly tabular 模型仍走原来的数据路径。

## GRU CUDA Memory Cleanup / GRU CUDA 显存清理

English:

The three GRU-based end-to-end factor models automatically move the model back to CPU and release PyTorch CUDA cache at stage boundaries:

- `bar_gru_15m`
- `multi_bar_gru_daily_15m`
- `residual_multi_bar_gru`

This cleanup runs after model `fit`, `predict`, and `save`, and the factor pipeline also performs a window-level cleanup after each window's prediction/write stage. No TOML option is required. It is intentionally not run after every batch, because PyTorch's CUDA cache improves batch-to-batch reuse and clearing it too frequently can slow training.

中文：

三个基于 GRU 的端到端因子模型会在阶段边界自动把模型移回 CPU，并释放 PyTorch CUDA cache：

- `bar_gru_15m`
- `multi_bar_gru_daily_15m`
- `residual_multi_bar_gru`

清理会在模型 `fit`、`predict`、`save` 之后执行；因子 pipeline 也会在每个 window 的预测和写出结束后做一次 window 级清理。不需要在 TOML 里额外配置。清理不会放在每个 batch 后执行，因为 PyTorch 的 CUDA cache 对 batch 间复用有帮助，过于频繁地清理会拖慢训练。

## Formal Factor Output

Factor configs write daily wide parquet files under:

```text
data/factors/stock/daily/{year}/{YYYYMMDD}.parquet
```

Each coverage date has the target factor column. Missing predictions are written as `NaN`; rerunning a factor overwrites that factor's column for the covered dates.

Factor metadata is written to:

```text
data/factors/factor_metadata.parquet
```

Refresh all `ml_alpha` factor metadata rows with:

```powershell
python -m yq_ml_alpha factor-metadata
```

Refresh one factor metadata row with:

```powershell
python -m yq_ml_alpha factor-metadata --config factors\bar_gru_15m.toml
```

Rust and Python now maintain separate source files, `factor_metadata.rust.parquet` and `factor_metadata.ml_alpha.parquet`, and publish their union to `factor_metadata.parquet`. Rust refresh preserves Python entries; duplicate identifiers across sources fail explicitly. A shared exclusive lock prevents simultaneous metadata writers. Legacy combined metadata is migrated on first refresh.

Rust 和 Python 分别维护上述两个来源文件，再合并到原来的 metadata 文件。Rust 刷新不再覆盖 ML 条目；跨来源重名会报错，共享锁防止同时刷新。首次刷新会迁移旧合并表。若进程异常退出留下 `factor_metadata.lock`，必须确认没有写入进程后才能删除该锁。

Or refresh the shared metadata in one step:

```powershell
python -m yq_ml_alpha factor-metadata-all
```

This command runs Rust `factor_engine -- metadata` first, then refreshes the `ml_alpha` factor rows, so the shared file contains both Rust-native and Python-generated factors.

For a formal factor such as `logsig_alpha_v`, `factor_id`, `name`, and `output_column` are all `logsig_alpha_v`.

## Logsig-Alpha-v

`logsig_alpha_v` is a formal end-to-end factor using volume-path signature features and an orthogonal MLP.

### Minute Input And Rolling State

Rust `derive-logsig` reads raw minute Parquets directly; no derived bar production is required:

```text
data/chn_stock_data/minute/{year}/{YYYYMMDD}.parquet
```

The Rust reader projects only `ts_code`, `trade_time`, and `vol`. It aggregates finite minute volume in the 09:31-11:30 and 13:01-15:00 sessions into 48 five-minute bars in memory. As in `derive-bar --columns volume`, duplicate minute records are included and incomplete five-minute groups may be summed; a stock must have all 48 finite bar values in every one of the 20 trading days. The 09:30 record is excluded. Missing days are not skipped or replaced with older days.

Rolling state retains only sorted stock codes and compact 48-value arrays for the required 20 trading days. Overlapping dates are reused; old days are evicted. Each run automatically reads 19 warmup sessions before its first target, without writing warmup output. Raw minute tables are released after daily aggregation. This is incremental input/state maintenance, not an O(1) signature algorithm: each target still computes its signature over the full 960-point path.

The old Python `logsig_signature` provider remains available for reference tests; it is not used by the production model config. `derive-bar` is not a prerequisite.

Rust builds an aligned `N x 960` volume matrix and computes `log(max(volume, 1)) -> lead-lag -> tensor signature -> Lyndon-basis logsignature`. Each daily Snappy Parquet contains `trade_date`, `ts_code`, and `logsig_0001` through `logsig_0226`. The production command has no Python/Numba fallback and requires no Python extension.

The derived dataset has a fixed definition (5-minute volume, 20 trading days, order 10). Its canonical path is:

```text
data/derived/stock/logsig_v/{year}/{YYYYMMDD}.parquet
```

### Materialize, Then Train

Run from the repository root. Rust processes dates sequentially and uses `--threads` for within-date signature calculation (default 2). Training uses the existing Python 3.8.3 GPU environment.

```powershell
$py = "D:\Users\Devin\anaconda383\python.exe"
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-logsig `
  --asset stock --start-date 20110101 --end-date 20260424 --threads 2
& $py -m yq_ml_alpha factor-run --config ml_alpha\factors\logsig_alpha_v.toml
```

The Rust command writes every requested trading day separately and releases its result. Only bounded rolling state remains in memory. No labels or training samples are loaded. Default `--overwrite true` recomputes and replaces requested dates atomically; `--overwrite false` skips existing output files without checking freshness. After correcting source minute data, rerun the affected dates with overwrite enabled. Missing/incomplete windows produce an empty file with the full schema; missing source dates are reported.

The factor config uses `features.type = "derived_logsig"`: training/prediction only project the 226 persisted features and keys. Missing files or columns fail explicitly; there is no implicit recomputation or source fingerprint validation. `factor-materialize` rejects this config and directs users to Rust `derive-logsig`. Training still loads sampled training/validation feature tables into CPU memory; prediction uses date batches as before. `model_workspace` remains for model artifacts, diagnostics and optional temporary sample caches, not the canonical derived features.

For an opt-in small real-data equivalence/performance comparison, run `tests/benchmark_logsig_minute.py` from the repository root with the Python 3.8.3 interpreter and a new `--scratch` directory. Its default is five target sessions (20110104-20110110) plus 19 warmup sessions. It uses hard links to input data in the isolated directory, generates real Rust volume bars, compares both paths twice with reversed ordering, checks all feature values, and reports wall time and peak process RSS. It neither trains a model nor writes formal factors. Recycle the scratch directory after checking results; never remove the source data.

The config uses:

- label: `future_vwap_return_5d`
- annual rolling training: preceding 36 calendar months, first 33 for training
- validation: final 3 calendar months; shared label-maturity purging applies
- first window: train 2013-01-01 through 2015-09-30, validate 2015-10-01 through 2015-12-31
- out-of-sample prediction: 2016-01-01 through 2026-04-24, one model per calendar year
- artifacts: `data/model_workspace/logsig_alpha_v/artifacts/annual_<prediction_year>/`
- sample frequency: every 5 trading days
- prediction frequency: daily
- model: `LogsigOrthogonalMLPAlphaModel`
- base factors: 8
- orthogonal penalty: `0.05`
- model-owned Rust neutralization: `model.params.neutralize = "barra:SIZE+sector"`

Base factors are model artifacts/diagnostics only. The formal factor library receives only the final neutralized `logsig_alpha_v` column.

## Annual Training And Safety / 年度训练与安全边界

All end-to-end factors use annual rolling windows: the preceding 36 calendar months, split into 33 training months and 3 validation months, sampled every 5 trading sessions. Machine learning remains in Python. Rust continues to provide existing feature operators and metadata interoperability only.

GRUPatch1 and GRUPatch2 reuse the 15-minute bar input and shared BarGRU training/inference pipeline. See [GRUPatch implementation](docs/GRU_PATCH.md) for report alignment and CLI commands.

Rawbar AGRU and LSTM use adjusted daily OHLC/VWAP and volume over 30 sessions,
with one output per stock and DFZQ tags. See [rawbar implementation](docs/RAWBAR_ALPHA.md).

三个 GRU、logsig 与 chip-age 统一使用年度滚动训练：过去 36 个自然月，前 33 个月训练、后 3 个月验证，每 5 个交易日采样。公共层按标签到期日截断，清除训练/验证交界的标签重叠。三个 GRU 使用 10 日 VWAP 标签（到期需 11 个交易日）、单种子 42、batch_size 3000；详见 [研报核对说明](docs/BAR_GRU_REPORT_ALIGNMENT.md)。机器学习继续使用 Python，不迁移到 Rust。

Train once, then predict / 先训练，再推理（从项目根目录执行 / run from repository root）：

```powershell
$env:PYTHONPATH = "$PWD\ml_alpha"
$factors = @("bar_gru_15m", "multi_bar_gru_daily_15m", "residual_multi_bar_gru")
foreach ($factor in $factors) {
    python -m yq_ml_alpha factor-train --config "ml_alpha\factors\$factor.toml"
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $factor" }
    python -m yq_ml_alpha factor-predict --config "ml_alpha\factors\$factor.toml"
    if ($LASTEXITCODE -ne 0) { throw "Prediction failed: $factor" }
}
```

Dates are feature dates. Training labels must settle strictly before the first validation date, and validation labels before the first prediction date. Built-in future VWAP return labels have an N+1 trading-day horizon (the 5-day label settles at t+6). Boundary samples are purged. Custom labels must declare `[label].lookahead_days`. Thus the configured date ranges are upper bounds, not a promise to include every boundary sample.

这些区间指特征日。训练标签必须在验证开始前结算，验证标签必须在预测开始前结算。内置 5 日 VWAP 收益标签从 t+1 到 t+6，因此边界会剔除未结算样本。自定义标签必须设置 `[label].lookahead_days`。Logsig 预测的基因子标准化按完整交易日截面执行，不受推理 mini-batch 大小影响，也不混合不同日期。

Each trained artifact has a `model.manifest.json` fingerprint covering actual training/validation dates, feature order, preprocessing, label, model settings and `data_version`. Resume and prediction reject missing/mismatched manifests; old artifacts must be retrained. Extending prediction dates alone is allowed. Set top-level `data_version` when rebuilding source data: the manifest does not hash the database contents.

每次训练保存配置与特征顺序指纹；恢复和预测时校验，不会仅因输出列存在就跳过训练。旧模型没有 manifest，需要重新训练。单纯延长预测区间可以复用；重建底层数据后应修改顶层 `data_version`，指纹并不校验整库文件内容。因子特征自动发现只检查训练结束前的文件，并排除自身、deprecated 和 model_generated 列；显式列清单仍可用于有意的模型堆叠。

## Resource Controls / 资源控制

Labeled bar-sequence bundles default to `[materialize].tensor_storage = "mmap"`. Tensors are spooled per date into temporary float32 files under the configured cache directory, avoiding a second full concatenated tensor in RAM. Row metadata remains in memory; prediction uses bounded date chunks. Use `"memory"` to opt out. Programmatic bundle owners should call `bundle.close()` when finished, especially on Windows.

带标签的 bar 序列默认使用 mmap 临时文件，按日追加 float32 tensor，避免在内存同时保留每日 tensor 和完整拼接副本。行索引仍在内存，mmap 也会使用操作系统页缓存，并不保证固定内存上限。直接使用 bundle 的调用者应在结束后调用 `close()`；CLI 正常阶段结束释放对象及临时文件。磁盘空间不足时可改为 `tensor_storage = "memory"`，但会增加内存需求。

Logsig supports `features.params.feature_cache_dir` (enabled in its config). Deterministic per-date features are reused only when settings and source file sizes/mtime match. This disk cache persists across runs; remove its directory when no longer needed. It stores features, not fitted normalization or model weights. Daily output schema discovery is memoized per writer instead of rescanning all dates for every prediction chunk.

Logsig 配置已启用确定性特征磁盘缓存：参数和源文件大小/修改时间一致才复用，不缓存训练标准化参数或模型权重。该缓存跨命令保留，可按需清理目录。写出 schema 每个 writer 按日期记忆，避免每个预测块重扫全区间。以上不代表已完成实际全量性能基准；训练规模和可用磁盘仍需按本机资源设置。
# Chip-age annual rolling factor

`chip_age_alpha` uses precomputed daily chip-age tensors and a CNN/GRU with
date-wise IC loss. It follows logsig_alpha_v's annual 36-month window (33 train + 3 validation) and stock universe.
See [implementation and commands](docs/CHIP_AGE_ALPHA.md) for confirmed source
assumptions, resource controls and the required historical label restoration.
