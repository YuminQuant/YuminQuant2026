# yq_ml_alpha

`ml_alpha` is the Python-side ML alpha package. It reads existing factor or raw
data, applies the configured universe/filter/preprocess pipeline, trains a
model, and writes alpha columns to `data/models/{year}/{trade_date}.parquet`.

## Run Built-In Examples

Run commands from the `ml_alpha` directory so the package is importable without
installing it:

```powershell
cd ml_alpha
python -m yq_ml_alpha run --config configs\examples\monthly_lr_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_xgb_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_xgb_optuna_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_lgbm_optuna_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_mlp_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_ic_sign_equal_weight.toml
python -m yq_ml_alpha run --config configs\examples\monthly_lasso_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_ridge_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_elasticnet_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_rf_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_lstm_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_gru_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_rnn_36.toml
python -m yq_ml_alpha run --config configs\examples\monthly_cnn_36.toml
```

If you want to use the local Python 3.8.3 GPU environment from the repository
root, call the interpreter explicitly and set `PYTHONPATH` for that shell:

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant
$env:PYTHONPATH = "D:\yuminwu_workspace\Internship\YuminQuant\ml_alpha"
D:\Users\Devin\anaconda383\python.exe -m yq_ml_alpha run --config ml_alpha\configs\examples\monthly_mlp_36.toml
```

The outputs are standard daily alpha parquet files:

```text
data/models/{year}/{trade_date}.parquet
```

You can backtest them with the Rust backtest CLI:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --groups 10 --rebalance 5
```

## CLI Entry Points

```powershell
python -m yq_ml_alpha run --config configs\examples\monthly_mlp_36.toml
python -m yq_ml_alpha train --config configs\examples\monthly_mlp_36.toml
python -m yq_ml_alpha predict --config configs\examples\monthly_mlp_36.toml
python -m yq_ml_alpha materialize --config configs\examples\monthly_mlp_36.toml
python -m yq_ml_alpha tune --config configs\examples\monthly_mlp_36.toml
```

`run` trains, predicts, and writes alpha files. `train` only fits and saves
artifacts. `predict` loads saved artifacts and writes predictions. `materialize`
builds sample data for inspection. `tune` delegates hyperparameter search to the
model implementation.

## Workflow And Config

The current v1 flow is:

```text
TOML config
  -> build rolling/static training windows
  -> DatasetBuilder reads factor/label/universe/filter data
  -> cross-sectional preprocessing
  -> model.fit(train, valid, context)
  -> model.predict(predict, context)
  -> AlphaWriter writes data/models
```

For the monthly examples:

- train samples are month-end cross sections
- predictions are daily
- the rolling scheme uses the previous 36 completed month-end samples
- features and labels use `zscore(log(rank))` by date
- missing features are filled with `0`
- rows with missing labels are excluded from training

The IC-sign equal-weight example reads existing IC detail files from
`data/backtest/stock/daily/ic`, orients each feature by `sign(mean(rank_ic))`,
and then averages the oriented features row-wise.

Important TOML sections:

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
validation_sample_count = 0

[label]
id = "future_vwap_return_20d"

[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = "__all__"

[preprocess]
cross_section_transform = "zscore_log_rank"
feature_fill_value = 0.0

[model]
name = "mlp"
class = "yq_ml_alpha.models.mlp_model.MLPAlphaModel"
artifact_dir = "data/model_workspace/monthly_mlp_36/artifacts"
```

`[dates].train` is required. `[dates].valid` and `[dates].predict` can be
omitted or set to `[]`; this is useful when you only want to fit and save a
model for later live prediction. In `[sample]`, `train_frequency` is required.
`predict_frequency` is required only when `[dates].predict` is non-empty.

```toml
[dates]
train = [20110101, 20260424]
valid = []
predict = []

[sample]
train_frequency = "monthly_end"
```

Supported sampling/refit frequencies:

```text
daily
weekly or weekly_end
monthly or monthly_end
5, 10, 20, ...
every_5_days, every_20_days, ...
```

For Python 3.8, prefer numeric forms such as `"5"` over `"every_5_days"` unless
the string helper compatibility patch has been applied in your environment.

To train on every factor column under a feature root, use:

```toml
[features]
type = "factor_frame"
root = "data/factors/stock/daily"
columns = "__all__"
```

`__all__` scans the parquet schemas under the root and uses every non-key
column except `trade_date`, `ts_code`, and `trade_time`. It does not use factor
metadata, so deprecated metadata flags do not affect ML feature discovery.

## Built-In Model Modules

Current model modules live under `yq_ml_alpha/models/`:

```text
linear_model.py     LinearRegressionAlphaModel, ordinary least squares via numpy.
regularized_linear_model.py
                    LassoAlphaModel, RidgeAlphaModel, ElasticNetAlphaModel with
                    optional GridSearchCV or RandomizedSearchCV inside fit().
tree_model.py       RandomForestAlphaModel, wraps sklearn RandomForestRegressor.
xgb_model.py        XGBoostAlphaModel, wraps xgboost.XGBRegressor.
xgb_optuna_model.py
                    XGBoostOptunaAlphaModel, runs Optuna TPE tuning inside
                    each fit window, then trains a final XGBoost regressor.
lgbm_optuna_model.py
                    LightGBMOptunaAlphaModel, Optuna-tuned LightGBM regressor.
mlp_model.py        MLPAlphaModel, PyTorch MLP for factor-frame alpha combination.
sequence_model.py   RNNAlphaModel, LSTMAlphaModel, GRUAlphaModel. Current
                    factor-frame inputs are reshaped by feature order; examples
                    use sequence_length=6.
cnn_model.py        CNNAlphaModel, 1D CNN over feature dimension. Pooling code is
                    present but disabled by default.
ic_sign_model.py    ICSignEqualWeightAlphaModel, equal-weight features by RankIC sign.
lgbm_model.py       LightGBMAlphaModel placeholder/wrapper for LightGBM style configs.
sklearn_model.py    Generic sklearn-style wrapper utilities.
base.py             AlphaModel and ModelContext interfaces.
```

Built-in configs:

```text
configs/examples/monthly_lr_36.toml
configs/examples/monthly_lasso_36.toml
configs/examples/monthly_ridge_36.toml
configs/examples/monthly_elasticnet_36.toml
configs/examples/monthly_rf_36.toml
configs/examples/monthly_xgb_36.toml
configs/examples/monthly_xgb_optuna_36.toml
configs/examples/monthly_lgbm_optuna_36.toml
configs/examples/monthly_mlp_36.toml
configs/examples/monthly_rnn_36.toml
configs/examples/monthly_lstm_36.toml
configs/examples/monthly_gru_36.toml
configs/examples/monthly_cnn_36.toml
configs/examples/monthly_ic_sign_equal_weight.toml
```

Model-specific parameters go under `[model.params]` and are passed through as
`context.model_params`. The shared pipeline does not interpret loss functions,
metrics, or search spaces.

Lasso/Ridge/ElasticNet can tune themselves during each rolling window. The
example configs enable `RandomizedSearchCV` and use a 36+1 monthly split:
the previous 36 completed month-end samples are used for fitting, and the
next month-end sample is used as the explicit validation set.

```toml
[train_scheme]
type = "rolling"
refit_frequency = "monthly_end"
train_sample_count = 36
validation_sample_count = 1
```

The explicit validation month is preferred for parameter selection. If it is
empty after label filtering, the models fall back to their internal CV setting.

```toml
[model.params.search]
enabled = true
method = "random"  # random | grid
cv = 3
n_iter = 40
scoring = "neg_mean_squared_error"
n_jobs = -1
random_state = 42
```

To use an explicit search space, add `params` below the search block:

```toml
[model.params.search.params]
alpha = [0.001, 0.01, 0.1]
fit_intercept = [true, false]
```

The built-in tuned configs expose their full default search grids in TOML, so
you can edit ranges without touching model code.

For RNN/LSTM/GRU configs, `sequence_length = 6` means the current factor-frame
feature vector is padded if needed and reshaped into six pseudo time steps. This
is a compatibility layer for factor combination, not a true six-date history
input. True time-series/raw-panel inputs should use a later raw-panel model.

XGBoost and LightGBM also have Optuna-tuned variants. They live in separate
model paths so the plain configs stay fast:

```toml
[model]
name = "xgboost_optuna"
class = "yq_ml_alpha.models.xgb_optuna_model.XGBoostOptunaAlphaModel"

[model.params.search]
n_trials = 50
valid_fraction = 0.2
random_state = 42
show_progress_bar = false

[model.params.search.space.n_estimators]
type = "int"
low = 100
high = 800
step = 50

[model.params.search.space.learning_rate]
type = "float"
low = 0.005
high = 0.2
log = true
```

For the tuned monthly examples, `validation_sample_count = 1` supplies the
Optuna objective with the month immediately after the 36 training samples. If no
valid rows survive label filtering, the model internally carves out
`valid_fraction` of the current training window for tuning, then fits the final
model on the full training window.

## Add A New Model

Create a new model file, for example:

```text
yq_ml_alpha/models/my_model.py
```

Implement the common interface:

```python
from yq_ml_alpha.models.base import AlphaModel, ModelContext

class MyAlphaModel(AlphaModel):
    def fit(self, train_data, valid_data, context: ModelContext) -> None:
        ...

    def predict(self, data, context: ModelContext):
        ...
```

Then point the config at the class:

```toml
[model]
name = "my_model"
class = "yq_ml_alpha.models.my_model.MyAlphaModel"
artifact_dir = "data/model_workspace/my_model/artifacts"

[model.params]
learning_rate = 0.03
```

`model.params` is passed through as `context.model_params`; the shared pipeline
does not interpret model-specific loss, metrics, or hyperparameters.

## Preprocess Transforms

Transforms are registered in `yq_ml_alpha/features/transforms.py`. The config
uses:

```toml
[preprocess]
cross_section_transform = "zscore_log_rank"
feature_fill_value = 0.0
```

To add a transform, implement a function in `transforms.py` and decorate it
with `@register_transform("your_name")`. Then use `your_name` in TOML. This
keeps transform lookup centralized and avoids editing the training pipeline.

## Output And Backtest

`AlphaWriter` writes or updates one daily parquet per date:

```text
data/models/{year}/{trade_date}.parquet
columns: trade_date, ts_code, alpha_id_1, alpha_id_2, ...
```

Parquet cannot append a single column in-place, so if a date file already
exists, the writer reads it, merges or replaces the alpha column, and rewrites
the file.

Backtest ML alpha with:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --groups 10 --rebalance 5
```

If an alpha is lower frequency, use forward fill:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_monthly_alpha --factor-root data\models --factor-fill ffill
```
