# Factor Development Navigation / 因子开发导航

这份文件只保留快速导航，完整教程请看：

This file is intentionally a short navigation page. For the full tutorial, read:

```text
factor_engine/FACTOR_DEVELOPMENT_README.md
```

## 常用入口 / Common Entry Points

```text
factor_engine/src/factor/chn_stock/daily/      stock daily factor files
factor_engine/src/factor/chn_stock/minute/     true minute-frequency factors
factor_engine/src/factor/future/daily/         future daily factor files
factor_engine/src/factor/common/               reusable panel/minute/raw helpers
factor_engine/src/operators/time_series/       time-series operators
factor_engine/src/operators/cross_sectional/   cross-sectional operators
```

## 最小流程 / Minimal Workflow

1. 新建一个 `.rs` 文件，文件名和 factor id 使用 snake_case。
2. 在 `spec()` 中声明 metadata、tags、dependencies、lookback、aliases。
3. 在 `compute()` 中写正式公式。
4. 如果是分钟因子，增加 intraday raw spec 和 `minute_compute()`。
5. 运行 metadata、plan、单日 run 验证。

1. Create one `.rs` file with a snake_case id.
2. Declare metadata, tags, dependencies, lookback, and aliases in `spec()`.
3. Implement the formal formula in `compute()`.
4. For minute-derived factors, add intraday raw specs and `minute_compute()`.
5. Validate with metadata, plan, and a one-day run.

## 常用命令 / Useful Commands

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- plan --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor --profile
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors your_factor --profile --refresh-minute-cache
```

## 详细教程 / Full Guide

See [../FACTOR_DEVELOPMENT_README.md](../FACTOR_DEVELOPMENT_README.md).
