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
python -m yq_ml_alpha run --config configs\examples\monthly_mlp_36.toml
```

The MLP example requires PyTorch:

```powershell
python -m pip install torch
```

The outputs are standard daily alpha parquet files:

```text
data/models/{year}/{trade_date}.parquet
```

You can backtest them with the Rust backtest CLI:

```powershell
cargo run --release --manifest-path ..\factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --groups 10 --rebalance 5
```

## Workflow

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
