<p align="center">
  <img src="docs/logo_v1.png" alt="YuminQuant logo" width="360">
</p>

# YuminQuant / 量化研究工作台

YuminQuant 将数据获取、派生数据、因子研究、机器学习和回测连接在一个本地优先的工作流中。Python 负责数据接口、模型与分析，Rust 负责因子、标签、风险暴露及回测计算。数据以 Parquet 存储，代码与生成的数据分开管理。

YuminQuant is a local-first quantitative research workspace connecting data collection, derived datasets, factor research, machine learning and backtesting. Python provides downloaders, models and analysis; Rust provides factors, labels, risk exposures and simulation. Data is stored as Parquet and managed separately from source code.

## 功能与模块 / Features and modules

| 模块 / Module | 覆盖内容 / Scope | 使用说明 / Guide |
|---|---|---|
| `data_manager`、`scripts` | 行情、财报、分析师、主营业务、基金等接口与历史/增量更新 / Market, financial, analyst, business and fund data; historical and incremental updates | [数据命令 / Data commands](docs/UPDATE_COMMANDS.md) |
| `factor_engine` | 股票与期货因子、标签、Barra CNE6、派生数据、截面回测、事件驱动策略 / Stock and futures factors, labels, risk exposures, derived data and backtests | [Rust 引擎 / Rust engine](factor_engine/README.md) |
| `ml_alpha` | 模型信号与端到端因子训练、推理、特征物化 / Model signals and end-to-end factors: training, inference and feature materialization | [ML Alpha](ml_alpha/README.md) |
| `analysis_tools` | 回测读取、绩效/IC 报表与图表 / Backtest loading, performance/IC reports and plots | [分析工具 / Analysis](analysis_tools/README.md) |
| `strategy_config` | 股票与期货策略 TOML / Stock and futures strategy configurations | [策略配置 / Strategy configs](strategy_config/README.md) |

各数据接口的权限、历史范围与更新方式不同，接口存在不代表本地已下载数据。
Data permissions, historical coverage and update schedules vary by endpoint. Available code does not imply that its input data is already present.

## 准备工作 / Setup

安装 Rust stable/Cargo，并准备适合所选模块的 Python 环境。以下命令从仓库根目录运行；路径使用正斜杠，适用于 PowerShell 和常见 Linux shell。

Install Rust stable/Cargo and a Python environment suitable for the selected modules. Run the following commands from the repository root; forward-slash paths work in PowerShell and common Linux shells.

```bash
python -m pip install pandas numpy pyarrow tqdm tomli tushare
cp config.example.toml config.toml
cargo check --manifest-path factor_engine/Cargo.toml
```

在 `config.toml` 中设置自己的 `[paths].base_data_dir` 及数据接口凭证、频率限制，不要照搬他人的绝对路径。模型和绘图库按各模块说明另行安装。

Set your own `[paths].base_data_dir`, API credentials and rate limits in `config.toml`. Do not copy another machine's absolute paths. Install model and plotting dependencies separately as described in their module guides.

示例配置若未包含数据根目录，请在已有 `[paths]` 节内补充，不要重复声明该节：
If the template omits the data root, add it inside the existing `[paths]` section without duplicating that section:

```toml
base_data_dir = "data"
```

此相对路径要求从仓库根目录运行；也可设置自己的绝对路径。
This relative path assumes commands run from the repository root; an absolute path is also supported.

## 一般流程 / Typical workflow

先准备日历和所需原始数据，再生成依赖的派生数据、标签或风险暴露。使用单个因子和短日期区间验证配置后，再扩大计算范围。

Prepare calendars and required raw inputs first, followed by any required derived data, labels or risk exposures. Start with one factor over a short interval before expanding a run.

```bash
python scripts/update_incremental.py --groups calendar stock_static
python scripts/update_incremental.py --groups stock_daily --start-date 20260420 --end-date 20260424
cargo run --release --manifest-path factor_engine/Cargo.toml -- metadata
cargo run --release --manifest-path factor_engine/Cargo.toml -- list --asset stock --frequency daily --ids-only true
cargo run --release --manifest-path factor_engine/Cargo.toml -- plan --asset stock --frequency daily --start-date 20260420 --end-date 20260424 --factors ret20
cargo run --release --manifest-path factor_engine/Cargo.toml -- run --asset stock --frequency daily --start-date 20260420 --end-date 20260424 --factors ret20 --date-batch-size 120 --profile
```

`plan` 查看计划而不生成因子，但不是完整的数据质量验证。示例日期不是运行限制，且回看窗口可能需要更早的数据。
`plan` inspects execution without producing factors; it is not a complete data-quality check. Example dates are not restrictions, and lookback windows can require earlier inputs.

## 数据与输出 / Data and outputs

下列路径相对于配置的数据根目录；原始数据和计算产物不随 Git 分发。
These paths are relative to the configured data root. Raw datasets and generated outputs are not distributed through Git.

```text
chn_stock_data/                          A 股原始数据 / A-share inputs
fund_data/                              基金数据 / Fund data
derived/                                派生数据 / Derived datasets
factors/{asset}/{frequency}/{year}/      正式因子 / Formal factors
label/{asset}/{frequency}/{year}/        收益标签 / Return labels
barra/{asset}/daily/CNE6/{year}/          风险暴露 / Risk exposures
models/{year}/                          模型信号 / Model signals
model_workspace/                        模型与特征缓存 / Model artifacts and feature caches
backtest/                               因子回测 / Factor backtests
strategy/                               策略模拟 / Strategy simulations
```

模型信号、正式因子和策略账户收益是不同产物，不能互换解释。标准命令默认本地 IO；可选存储扩展不代表所有输入、回测或派生入口都支持远程文件系统。

Model signals, formal factors and strategy account returns are distinct outputs. Standard commands use local IO. Optional storage adapters do not imply universal remote-filesystem support for inputs, backtests or derived-data commands.

## 文档导航 / Documentation

- [下载粒度 / Download granularity](docs/DOWNLOAD_GRANULARITY.md) · [基金接口 / Fund data](docs/fund_data.md)
- [因子开发 / Factor development](factor_engine/docs/FACTOR_DEVELOPMENT_README.md)
- [财务 PIT 因子 / Financial PIT factors](factor_engine/docs/FINANCIAL_FACTOR_DEVELOPMENT_README.md)
- [基金持仓因子 / Fund-holdings factors](factor_engine/docs/FUND_HOLDINGS_FACTOR_DEVELOPMENT_README.md)
- [一致预期指标 / Consensus indicators](factor_engine/docs/CONSENSUS_DERIVED_README.md)
- [模型开发 / Model development](ml_alpha/docs/DEVELOPMENT_README.md)
- [策略开发 / Strategy development](factor_engine/docs/STRATEGY_README.md)

## 安全与许可 / Safety and license

凭证、本地配置、数据、模型和临时验证产物应保持未跟踪；提交前检查 `git status`。部署环境的鉴权和公司专用配置不属于公开项目。验证脚本及产物放在被忽略的 `tmp/`。

Keep credentials, local configuration, datasets, models and validation artifacts untracked; inspect `git status` before committing. Environment-specific authentication and company deployment configuration belong outside the public project. Keep validation scripts and artifacts in ignored `tmp/`.

本项目采用 Apache License 2.0，见 [LICENSE](LICENSE)。
Licensed under Apache License 2.0; see [LICENSE](LICENSE).
