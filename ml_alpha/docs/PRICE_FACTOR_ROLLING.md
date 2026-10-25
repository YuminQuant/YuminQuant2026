# Price-factor rolling training / 量价因子滚动训练

The 17 price/volume configurations in `factors/`, including Logsig and chip-age,
use the following project settings. Analyst-coverage and report-text factors
retain their separate configurations.

- Train history begins in 2011; prediction remains 2016-01-01 to 2026-04-24.
- Refit each year using the preceding 60 calendar months: 57 train, 3 validation.
- Sample every five trading sessions; retain the existing single seed and labels.
- Stock encoder chunks use `batch_size=5000`; whole-date IC/BatchNorm and global
  stock attention keep the complete eligible cross-section.
- Chip-age accepts `batch_size` as the encoder micro-batch name; old checkpoints
  with `micro_batch_size` remain readable. ASTGNN permits encoder chunks up to5000.

Logsig retains its 5-day VWAP target (6-session maturity); chip-age retains the
10-day VWAP target (11-session maturity). Public window construction requires
`sample_position + label_horizon <= training_cutoff_position`, and also purges
training labels at the validation boundary. A 10-day VWAP fit on2015-12-31 may
therefore use no feature date later than 11 prior trading sessions, with the
actual sampled endpoint potentially earlier. This is not a simple calendar-day
subtraction or delayed prediction using labels unavailable at fit time.

Both first cluster fits write under
`data/model_workspace/<factor>/rolling60m_bs5000/artifacts/annual_<year>/` to
preserve previously produced weights. Other price configurations retain their
configured artifact roots until submitted.

Chip-age numerical semantics remain `chip_age_rolling250_v2`: every output is
computed from250 trading sessions with zero initial mass. Each annual shard
automatically replays249 prior sessions. Its manifest is a stable definition
identity, published atomically under a short shared lock, not a marker that an
entire requested history is complete. Daily files are owned by disjoint annual
tasks. Consumers must wait for the scheduler's complete-array success barrier;
missing daily files continue to fail in the reader. Annual jobs never delete
another shard's manifest or use a shared temporary manifest filename.

Deployment-specific arrays, GPU environments, authentication and job records
belong in private deployment configuration, not in the public repository.
