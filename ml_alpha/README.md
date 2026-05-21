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

The 20-day order-10 lead-lag volume signatures are computed on demand in Python with Numba. They are not written as intermediate parquet files. The provider returns `trade_date`, `ts_code`, and `sig_0001` through `sig_2046`, while reusing an in-process LRU cache of 5-minute bar days during train, validation, and prediction loads.

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
