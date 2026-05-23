# ml_alpha

`ml_alpha` separates model experiments from formal end-to-end factors.

## Configuration Layers

- `models/*.toml`: model or experiment configs. These write signals to `data/models`.
- `factors/*.toml`: formal single-factor configs. The TOML file name, `factor_id`, output column, and factor metadata identifier must all use the same semantic snake_case factor name.

Current formal factor configs:

- `factors/bar_gru_15m.toml`
- `factors/multi_bar_gru_daily_15m.toml`
- `factors/residual_multi_bar_gru.toml`
- `factors/logsig_alpha_v.toml`

`mdl_*` is reserved for model-layer configs. `e2e_fct_*` is no longer a valid formal factor identifier.

## CLI

Use explicit model/factor commands only:

```powershell
python -m yq_ml_alpha model-run --config models\mdl_000001.toml
python -m yq_ml_alpha factor-run --config factors\bar_gru_15m.toml
python -m yq_ml_alpha factor-run --config factors\logsig_alpha_v.toml
```

Available commands:

- `model-run`, `model-train`, `model-predict`, `model-materialize`
- `factor-run`, `factor-train`, `factor-predict`, `factor-materialize`

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

For a formal factor such as `logsig_alpha_v`, `factor_id`, `name`, and `output_column` are all `logsig_alpha_v`.

## Logsig-Alpha-v

`logsig_alpha_v` is a formal end-to-end factor using volume-path signature features and an orthogonal MLP.

### Prepare 5-Minute Bars

First generate 5-minute stock bars if they do not exist:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- derive-bar --asset stock --source minute --bar-size 5 --start-date 20110101 --end-date 20260424
```

`logsig_alpha_v` reads those bars from:

```text
data/derived/stock/bar/5m/{year}/{YYYYMMDD}.parquet
```

The 20-day order-10 lead-lag volume logsignatures are computed on demand. Python reads the 5-minute bars with column projection, builds an aligned `N x 960` volume matrix, and calls the Rust extension for `log(max(volume, 1)) -> lead-lag -> tensor signature -> Lyndon-basis logsignature`. The provider returns `trade_date`, `ts_code`, and `logsig_0001` through `logsig_0226`; the old Numba tensor-signature path remains only as a compatibility fallback and is logged as such.

Rust logsignature computation uses a small dedicated thread pool by default (`3` threads). Override it before launching training when needed:

```powershell
$env:YQ_LOGSIG_THREADS="2"
```

### Train And Materialize

```powershell
D:\Users\Devin\anaconda383\python.exe -m pip install maturin
D:\Users\Devin\anaconda383\python.exe -m maturin develop --manifest-path ..\factor_engine\python\yq_factor_engine_py\Cargo.toml
python -m yq_ml_alpha factor-run --config factors\logsig_alpha_v.toml
```

The config uses:

- label: `future_vwap_return_5d`
- rolling window: 4 years
- refit frequency: annual end
- train/validation split: first 75% sampled dates for train, last 25% for validation
- sample frequency: every 5 trading days
- prediction frequency: daily
- model: `LogsigOrthogonalMLPAlphaModel`
- base factors: 8
- orthogonal penalty: `0.05`
- model-owned Rust neutralization: `model.params.neutralize = "barra:SIZE+sector"`

Base factors are model artifacts/diagnostics only. The formal factor library receives only the final neutralized `logsig_alpha_v` column.
