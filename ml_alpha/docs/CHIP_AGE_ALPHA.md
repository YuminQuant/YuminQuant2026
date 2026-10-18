# chip_age_alpha

## Definition and provenance

Reference: Huatai Securities, 2026-06-02, 基于筹码分层结构的端到端AI因子,
pages 5 and 7 (including Figures 6 and 7). This is an implementation with
explicit engineering choices, not the authors' code or a claim of exact replication.

Report-defined elements: VWAP-centered triangular daily distribution; age groups
1-2, 3-10, 11-100 and >=101; 32 relative log-price bins in [-0.7,0.7]; joint
normalization then square root; two Conv1D layers (32 channels, 5/2 and 3/1
kernel/padding), two-layer 64-hidden GRU, last step, linear output, BatchNorm;
30 daily observations; AdamW, lr=0.001, weight_decay=0.0001, IC loss.

User-confirmed engineering choices:

- Free-float turnover `turnover_rate_f` is divided by 100. Replacement probability
  is `1-exp(-rate)` for ALL rates. Rates above 100% remain valid; no clipping.
  This deliberately differs from the report's linear turnover recursion.
- Use a fixed 250-trading-session window for BOTH training and out-of-sample
  features. Start with zero mass. Every day's cohort enters with actual replacement
  mass `1-exp(-rate)`; no first-day full-holdings assumption, no 60-day warmup,
  and no initialization-residual threshold.
- For a cohort born at s, its mass at t is
  `(1-exp(-r_s))*exp(-sum(r_u for u=s+1..t))`. Keep only s >= t-249.
  Maintain individual ages 1-250 so the oldest cohort can expire exactly. Never
  normalize the internal state: normalize only the final output distribution.
  The channels are 1-2, 3-10, 11-100 and 101-250, not an infinite-age tail.
- Require 250 consecutive valid market sessions (confirmed full-day suspensions
  count as zero-turnover sessions). A reset requires refilling this full window.
  A zero surviving-mass window has no valid output. Thirty consecutive daily
  feature tensors require at least 279 sessions of source history.
- Integrate the triangle over an adjusted log-cost grid of step 0.01. Sparse
  price bins grow as needed. Prices use contemporaneous adj_factor, never a
  future terminal factor. Rolling state and independent 250-day calculation
  implement the same formula, without requiring a persistent checkpoint.
- Each day maps its own historical costs relative to its own adjusted close.
  Map internal bin centers to 32 output bins. Accumulate outside mass in edge
  bins, record `outside_mass`, jointly normalize and square-root. Do not
  re-normalize after square root and do not normalize each age channel separately.
- No special VWAP out-of-range clipping, snapping or resetting. VWAP unit is
  `amount * 10 / vol` with current source units.
- Missing/invalid key inputs, negative turnover or non-suspension zero volume:
  skip the symbol, remove its state and rewarm on the next valid observation.
  Explicit full-day suspension (`S`, empty suspend_timing) with no trades keeps
  chips, advances ages, and carries the previous adjusted close. Unknown missing
  dates are not treated as confirmed suspensions.
- ReLU after each convolution; dropout=0.1 between GRU layers; raw adjusted
  10-day VWAP labels. No rank-Gauss, orthogonality penalty or neutralization.
- Seed 42 initially; `seeds` accepts multiple distinct seeds and averages their
  evaluation scores. BatchNorm uses learned running statistics in evaluation.

## Annual rolling split and stock universe

Follow logsig_alpha_v: refit at each prior year-end using the preceding 36
calendar months (first 33 months train, last 3 months validation). First window:
train 20130101-20150930, valid 20151001-20151231, predict during 2016.
Advance all ranges one year per model; prediction ends 20260424; sample training endpoints every five trading days,
with ALL thirty daily observations inside each sample; predict daily.
The public runtime purges labels at train/valid and prediction boundaries.
The label is `future_vwap_return_10d`, with lookahead 11 sessions. Validation
selects the epoch; it is not merged into a final refit. Each new model activates on the first trading day of its prediction year.

`mkt_all`, exclude_bj=true, exclude_st=true, exclude_limit=false match logsig.
These inherit current TradeFilters behavior, including its existing lack of ST
exclusion before 2016. This is not a historical CSI All Share constituent universe.

The pre-2016 label files were removed from the local library previously.
Generate the required 10-day labels before training; do not shorten the rolling
training windows to compensate for missing files.

## Resources and storage

The Rust estimator stores each daily cohort as sorted contiguous price bins and
a separate surviving weight. Daily decay updates one weight per cohort rather
than every price bin; feature emission also collects retained mass in the same
pass. The 250-session window and output definition are unchanged, but floating
point operation ordering can produce rounding differences (not bitwise identity).
Daily logs report `load_ms` (including conversion/indexing), `compute_ms`,
`write_ms` (including output table construction), `day_ms` and `elapsed_s`.
`--date-batch-size` (default 20) controls target trading dates per processing batch.
Primary input tables are loaded per batch; dates are processed sequentially.
Warmup is batched separately and state persists across batches. Larger batches
increase input memory; no disk cache or concurrent writes are introduced.
`batch_load_ms` measures primary loading; daily `load_ms` measures extraction
and suspension loading.
Equivalence test source is included; execution and performance measurement are
pending user authorization.

- `data/derived/stock/chip_age/YYYY/YYYYMMDD.parquet`: Snappy daily tensors,
  128 float32 feature columns, keys, outside_mass, retained_window_mass and
  history_days. Each invocation automatically preloads the preceding 249 market
  sessions before its first target date. Days then advance chronologically using
  a bounded sliding state. No replay from a global historical origin is needed.
  No persistent state checkpoint or hidden temporary cache is written.
- `manifest.json` marks successful derivation and identifies the calculation
  definition, not a run timestamp/date range. Normal new-date appends do not
  invalidate fixed-model fingerprints. The marker is removed before writes;
  interrupted writes require a successful rerun before consumption. Missing
  daily input files fail explicitly in the provider. The manifest version identifies the rolling250 v2 definition independently
  of the directory name.
- Revisions to existing historical source data require regeneration of affected
  daily features and a deliberate `data_version` change/retraining when training
  inputs change. Definition identity alone does not hash raw historical data.
- Python keeps keys/labels in memory and loads daily tensors through a 128 MiB
  LRU buffer. This bound covers cached DataFrames, not all process memory. Expanded
  windows are constructed per encoder chunk, not stored for the whole dataset.
- Encoder chunks use activation recomputation during training. A single complete
  date supplies one BatchNorm update and one IC loss; validation averages daily
  ICs equally. Chunking does not approximate IC by averaging local correlations.
- Feature writes use same-directory temporary files on the configured D: data
  root. C: derivation output is rejected. Training sample disk caching is disabled.
- Models: `data/model_workspace/chip_age_alpha/artifacts/annual_<prediction_year>/model.pkl`
  (torch checkpoint); manifests, loss_history.parquet and model_info.json alongside.
- Scores: `data/factors/stock/daily/YYYY/YYYYMMDD.parquet`, column `chip_age_alpha`.

## Commands (not executed during implementation)

From the repository root, after source review and authorization to compute:

```powershell
# Emit dates before 2013 too, so the first training sample has 30 daily inputs.
cargo run --release --manifest-path factor_engine\Cargo.toml -- derive-chip-age `
  --asset stock --start-date 20120101 --end-date 20260424 --date-batch-size 120

cargo run --release --manifest-path factor_engine\Cargo.toml -- label-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-run `
  --asset stock --frequency daily --start-date 20130101 --end-date 20251231 `
  --labels future_vwap_return_10d --date-batch-size 120

$py = 'D:\Users\Devin\anaconda383\python.exe'
$env:PATH = 'D:\Users\Devin\anaconda383;D:\Users\Devin\anaconda383\Library\bin;D:\Users\Devin\anaconda383\Scripts;' + $env:PATH
& $py -u -m yq_ml_alpha factor-train --config ml_alpha\factors\chip_age_alpha.toml
& $py -u -m yq_ml_alpha factor-predict --config ml_alpha\factors\chip_age_alpha.toml
```

Derivation is currently sequential and does not support date-batch-size, which
is a label-run flag above. No automatic training or label restoration occurs.

## Verification status

No compilation, tests, derivation or training was launched during implementation.
Added source-level test cases cover first-cohort weighting, high turnover, exact
250-day expiry, direct-formula equivalence, independence from earlier history,
joint normalization, calendar alignment and missing input data.
GPU compatibility, performance and prediction consistency remain to be verified
when running tests is authorized.

For daily out-of-sample feature updates, request only the new date(s). The Rust
command automatically reads 249 earlier sessions, so it uses the same definition
as historical generation. The neural input still needs the existing preceding
29 DAILY FEATURE files. If those are absent, generate the feature range first.
The fixed model weights are unchanged by ordinary future-date appends.
