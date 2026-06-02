<p align="center">
  <img src="docs/logo_v1.png" alt="YuminQuant logo" width="360">
</p>

# YuminQuant / 本地量化研究系统

YuminQuant 是一个本地 parquet 数据湖、Rust 因子/回测/策略引擎，以及 Python ML alpha 和分析工具的组合项目。数据默认存放在本机 `data/` 下，不进入 Git。

YuminQuant is a local parquet-based quantitative research workspace. It combines Python data downloaders, a Rust factor/backtest/strategy engine, Python ML alpha tooling, and Python analysis helpers. The local `data/` directory is generated data and is intentionally not tracked by Git.

## License

YuminQuant is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

## 模块总览 / Module Map

```text
YuminQuant/
  data_manager/          # Python downloaders and Tushare client
  scripts/               # Incremental update and maintenance scripts
  factor_engine/         # Rust factors, labels, Barra, backtest, strategy
  ml_alpha/              # Python ML alpha training and prediction package
  analysis_tools/        # Python backtest result analysis helpers
  strategy_config/       # Strategy TOML configs by asset class
  docs/                  # Data-update docs
  data/                  # Local generated parquet data, ignored by Git
```

主要输出路径 / Main output paths:

```text
data/factors/{asset}/{frequency}/{year}/{trade_date}.parquet
data/factors/_cache/intraday_daily/chn_stock/{year}/{trade_date}.parquet
data/barra/{asset}/daily/CNE6/{year}/{trade_date}.parquet
data/label/{asset}/{frequency}/{year}/{trade_date}.parquet
data/backtest/stock/daily/{returns,ic,factor_stats,holdings,industry_weights}/
data/models/{year}/{trade_date}.parquet
data/strategy/{asset_class}/{strategy_id}/holdings.parquet
```

## 环境与数据 / Setup And Data

安装基础 Python 依赖，并复制本地配置：

Install basic Python dependencies and create your local config:

```powershell
pip install pandas numpy pyarrow tqdm tomli tushare
copy config.example.toml config.toml
```

在 `config.toml` 中填入 Tushare token。默认数据根目录是：

Fill your Tushare token in `config.toml`. The default data root is:

```text
D:/yuminwu_workspace/Internship/YuminQuant/data
```

常用数据更新命令 / Common update commands:

```powershell
python scripts\update_incremental.py --groups calendar stock_static future_static index_static
python scripts\update_incremental.py --groups stock_daily future_daily --start-date 20260214
python scripts\update_incremental.py --groups stock_minute future_minute --start-date 20260214
python scripts\update_incremental.py --groups stock_trade_filter --start-date 20160101 --end-date 20260424
python scripts\update_incremental.py --groups index_weight --start-date 20090101 --end-date 20260331
python scripts\update_incremental.py --groups stock_dividend --start-date 20090101 --end-date 20260424 --rebuild
```

更多下载说明见 / More downloader docs:

- [docs/UPDATE_COMMANDS.md](docs/UPDATE_COMMANDS.md)
- [docs/DOWNLOAD_GRANULARITY.md](docs/DOWNLOAD_GRANULARITY.md)

## 常用 CLI 速查 / CLI Cheatsheet

以下命令均从仓库根目录运行，除非特别说明。

Run these commands from the repository root unless noted otherwise.

### 因子 / Factors

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine\Cargo.toml -- plan --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors utd
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors utd --profile
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --tags GFZQ --factor-batch-size 20 --date-batch-size 120 --profile --refresh-minute-cache
```

多 raw 共享 provider 示例：下面这组 V-shape 因子来自同一个分钟 provider。引擎会在同一批次里共享一次分钟扫描，并只物化本次选中因子需要的 sibling raw。

Multi-raw provider example: these V-shape factors share one minute provider. In one factor batch, the engine scans the minute file once and materializes only the selected sibling raw columns.

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- run --asset stock --frequency daily --start-date 20260424 --end-date 20260424 --factors negv_mean,negv_max,negvwgt_mean,negvwgt_max,flash_crash_prob_v --profile --refresh-minute-cache
```

### Barra 与 Label / Barra And Labels

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- barra-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- barra-run --asset stock --frequency daily --start-date 20200101 --end-date 20201231 --families DIVIDEND_YIELD,GROWTH,LIQUIDITY,MOMENTUM,QUALITY,SENTIMENT,VALUE,VOLATILITY --date-batch-size 240 --profile

cargo run --release --manifest-path factor_engine\Cargo.toml -- label-metadata
cargo run --release --manifest-path factor_engine\Cargo.toml -- label-run --asset stock --frequency daily --start-date 20260101 --end-date 20260424 --label-batch-size 20 --profile
```

### 截面回测 / Cross-Sectional Backtest

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --factors peer_ds_by_t --groups 10 --rebalance 5
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --tags XYZQ --groups 10 --rebalance weekly --factor-batch-size 10 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --all-factors --groups 10 --rebalance 5 --factor-batch-size 10 --date-batch-size 120
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors ml_alpha_mlp --factor-root data\models --groups 10 --rebalance 20 --factor-fill ffill
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20110101 --end-date 20260424 --all-factors --factor-root data\barra\stock\daily\CNE6 --groups 10 --rebalance 20 --neutralize barra:all+sector
```

常用参数 / Common flags:

```text
--factors a,b,c | --tags XYZQ | --all-factors
--factor-root data\models
--label future_vwap_return_1d | future_vwap_return_20d
--groups 5|10|20
--rebalance daily|5|10|weekly|biweekly|monthly|quarterly
--factor-fill none|ffill
--universe mkt_all|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id
--benchmark mkt_mean|000300.SH|000905.SH|000852.SH|000985.CSI|custom_id
--neutralize none|sector|barra:SIZE|barra:SIZE+sector|barra:all+sector
--detail none|holdings|industry_weights|all
--detail-sector sw_l1|ci_l1
--factor-batch-size 10
--date-batch-size 120
```

### ML Alpha

不需要安装包时，进入 `ml_alpha` 目录运行：

Run from `ml_alpha` when you do not want to install the package:

```powershell
Push-Location .\ml_alpha
python -m yq_ml_alpha model-run --config models\mdl_000001.toml
python -m yq_ml_alpha model-run --config models\monthly_xgb_36.toml
python -m yq_ml_alpha model-run --config models\monthly_mlp_36.toml
python -m yq_ml_alpha model-run --config models\monthly_elstm_ranknet_36.toml
Pop-Location
```

端到端 bar 序列因子现在使用 tensor 数据通路：`bar_panel` / `multi_bar_panel` 直接构造 `[N,T,F]`，并使用 `max_cache_sessions = "auto"` 控制 session cache。详见 [ml_alpha/README.md](ml_alpha/README.md)。

End-to-end bar sequence factors now use a tensor data path: `bar_panel` / `multi_bar_panel` build `[N,T,F]` tensors directly and use `max_cache_sessions = "auto"` for session cache sizing. See [ml_alpha/README.md](ml_alpha/README.md).

输出写入 / Output:

```text
data/models/{year}/{trade_date}.parquet
```

回测 ML alpha / Backtest ML alpha:

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- backtest --asset stock --frequency daily --start-date 20200101 --end-date 20260424 --factors mdl_000001 --factor-root data\models --groups 10 --rebalance 20
```

### Strategy

策略模块是事件驱动的真实策略模拟器，输出账户和持仓 PnL，不是快速因子收益检验。

The strategy module is an event-driven simulator for concrete trading strategies. It outputs account and holding snapshots rather than factor-test returns.

```powershell
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\stock\ml_xgb_top20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml
cargo run --release --manifest-path factor_engine\Cargo.toml -- strategy-run --config strategy_config\future\ag_sma_20.toml --detail true
```

输出 / Output:

```text
data/strategy/stock/ml_xgb_top20/holdings.parquet
data/strategy/future/ag_sma_20/holdings.parquet
```

## 开发教程入口 / Development Guides

- 因子开发 / Factor development: [factor_engine/FACTOR_DEVELOPMENT_README.md](factor_engine/FACTOR_DEVELOPMENT_README.md)
- Factor engine CLI: [factor_engine/README.md](factor_engine/README.md)
- Strategy development: [factor_engine/STRATEGY_README.md](factor_engine/STRATEGY_README.md)
- Strategy config: [strategy_config/README.md](strategy_config/README.md)
- ML model development: [ml_alpha/README.md](ml_alpha/README.md)
- Python analysis helpers: [analysis_tools/README.md](analysis_tools/README.md)

## Python 分析工具 / Python Analysis Tools

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("analysis_tools").resolve()))

from yq_analysis.io import load_backtest_result
from yq_analysis.report import make_backtest_report
from yq_analysis.plots import plot_return_summary

result = load_backtest_result(r"data/backtest/stock/daily", "acf")
report = make_backtest_report(result["returns"], result["ic"], result["factor_stats"])
fig = plot_return_summary(result["returns"], groups=10)
```

## Git 与数据安全 / Git And Data Safety

本地密钥、数据和输出不会进入 Git：

Local secrets and generated data are not tracked:

```text
config.toml
data/
target/
.pytest_cache/
analysis_tools/plots/
```

提交前建议检查 / Before committing:

```powershell
git status
git diff --stat
```
