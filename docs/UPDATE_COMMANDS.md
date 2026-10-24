# Data Updates / 数据更新命令

本文档介绍本地原始数据的常用历史下载与增量更新方式。命令从仓库根目录运行，路径使用正斜杠，适用于 PowerShell 和常见 Linux shell。先在 `config.toml` 中设置自己的数据根目录、接口凭证与访问频率。

This guide covers common historical downloads and incremental updates for local source data. Run commands from the repository root; forward-slash paths work in PowerShell and common Linux shells. Configure your data root, API credentials and rate limits in `config.toml` first.

## 更新顺序 / Update order

先更新日历和基础信息，再更新日线，最后更新分钟线及其派生 bar。股票和期货分钟下载器依赖本地日线确定有效标的范围；计算派生 bar 前需要已有分钟数据。

Update calendars and reference data first, daily data second, and minute data and derived bars last. Stock and futures minute downloaders use local daily data to identify eligible instruments. Derived bars require minute inputs.

```bash
python scripts/update_incremental.py --groups calendar stock_static future_static
python scripts/update_incremental.py --groups stock_daily future_daily --start-date 20260401 --end-date 20260424
python scripts/update_incremental.py --groups stock_minute future_minute --start-date 20260401 --end-date 20260424
python scripts/update_incremental.py --groups stock_derived_bar --derived-bar-sizes 5,15 --start-date 20260401 --end-date 20260424
```

`stock_daily` 同时更新行情、复权因子、涨跌停、估值、停牌、资金流与 ST 数据，并生成交易过滤表；`stock_derived_bar` 调用 Rust 引擎，需安装 Cargo。

The daily stock group includes prices, adjustment factors, limits, valuation data, suspensions, money flows and ST data, and builds trade filters. Derived bars invoke the Rust engine and require Cargo.

## 默认流程 / Default pipeline

```bash
python scripts/update_incremental.py --start-date 20260401 --end-date 20260424
```

默认任务按下列顺序执行；也可以用 `--groups` 只运行其中几组。

Defaults run in the following order. Use `--groups` to select only the groups you need.

| 任务组 / Group | 内容 / Scope |
|---|---|
| `calendar` | 交易日历 / Trading calendars |
| `stock_static` | A 股基础信息 / A-share reference data |
| `fund_basic` | 基金基础信息快照 / Current fund reference snapshot |
| `fund_portfolio` | 基金持仓，按区间内季度末报告期触发 / Fund holdings, triggered by quarter-end periods within the range |
| `stock_daily` | A 股日频数据和交易过滤 / Daily A-share data and trade filters |
| `stock_minute` | A 股分钟数据 / A-share minute data |
| `stock_derived_bar` | 默认 5m、15m 聚合 bar / Aggregated bars, default 5m and 15m |
| `future_static` | 期货基础信息 / Futures reference data |
| `future_daily` | 期货日频数据 / Futures daily data |
| `future_minute` | 期货分钟数据 / Futures minute data |

默认流程不包含所有已接入接口。其他基金接口、财报、分红、分析师明细和主营业务需要显式选择。按组补数更便于控制数据量和排查问题。

The default pipeline does not run every available endpoint. Other fund sources, financial statements, dividends, analyst reports and main-business data require explicit selection. Separate groups make data volume and failures easier to manage.

## 静态与分类数据 / Reference and classification data

```bash
python scripts/update_incremental.py --groups static
```

`static` 包含交易日历、股票、期货、ETF、期权、指数及港美股基础数据；指数部分包括分类和行业成分，港美股部分包括各自日历。基金基础信息使用单独的 `fund_basic` 组。

The static group covers calendars and stock, futures, ETF, options, index, Hong Kong and US reference data. Index updates include classifications and industry membership; Hong Kong and US updates include their calendars. Fund reference data uses the separate `fund_basic` group.

可独立选择以下任务组，多个组用空格分隔：
Select individual groups below, separated by spaces:

```bash
python scripts/update_incremental.py --groups stock_static future_static fund_basic
python scripts/update_incremental.py --groups etf_static option_static index_static hk_static us_static
```

基础信息是接口当前快照，不应将历史起止日期理解为历史版本还原。
Reference tables are current API snapshots; historical date arguments do not reconstruct past versions.

## 财报、分红与分析师 / Financial statements, dividends and analysts

三大报表的增量调度从指定区间中选出季度末报告期，按报告期请求，而非逐个公告日请求。未包含季度末的区间会跳过；`stock_financial` 只更新三大报表。

Statement updates select quarter-end reporting periods within the requested range and query by period, rather than by each announcement date. Ranges without quarter ends are skipped. This group updates only income, balance-sheet and cash-flow statements.

```bash
python scripts/update_incremental.py --groups stock_financial --start-date 20260331 --end-date 20260331
python scripts/update_incremental.py --groups stock_dividend stock_alt --start-date 20260401 --end-date 20260424
```

`stock_dividend` 更新分红，`stock_alt` 更新分析师明细。它们与三大报表使用不同的下载入口；是否有可用数据取决于接口权限和实际披露情况。

The dividend group updates distributions, while the alternative-data group updates analyst reports. Their download paths differ from statements; availability depends on permissions and actual disclosures.

主营业务在区间包含 `0430/0831/1031` 检查日时触发；`0430` 请求上一年年报和当年一季报，另外两个检查日分别请求半年报和三季报。

Main-business updates trigger on April 30, August 31 and October 31 within the range. April 30 requests the prior annual and current Q1 periods; the other checkpoints request H1 and Q3.

```bash
python scripts/update_incremental.py --groups stock_mainbz --start-date 20260430 --end-date 20260430
```

## 历史初始化 / Historical initialization

以下示例覆盖 2009 年起的历史，截止日期为示例值；运行前确认所需时间范围。历史下载和增量调度是不同入口。

These examples initialize history from 2009 with an example cutoff. Choose your range before running. Historical initialization and incremental scheduling use different entrypoints.

```bash
python scripts/init_stk_financial.py --start-year 2009 --end-date 20260424
python scripts/init_stk_mainbz.py --start-year 2009 --end-date 20260424
python scripts/init_fund_data.py --datasets basic portfolio --start-year 2009 --end-date 20260424
```

三大报表脚本只处理 income、balancesheet、cashflow，不下载分析师或分红；其截止参数同时用于限制报告期和实际披露日期。主营业务没有公告日，截止参数只限制报告期。基金 `basic` 为当前快照，忽略历史截止日期；`portfolio` 使用历史参数。

The statement script handles only income, balance-sheet and cash-flow data, not analyst reports or dividends; its cutoff restricts periods and disclosure dates. Main-business data has no announcement date, so its cutoff restricts periods only. Fund basic data is a current snapshot and ignores historical cutoffs; portfolio data uses the historical parameters.

## 常用参数 / Common options

| 参数 / Option | 含义 / Meaning |
|---|---|
| `--groups` | 选择一个或多个任务组，按给定顺序执行 / Select groups to run in the specified order |
| `--start-date / --end-date` | `YYYYMMDD`；具体含义由任务组决定 / Dates whose semantics depend on the group |
| `--calendar-end-date` | 日历截止日，默认北京时间当年的年末 / Calendar cutoff, defaulting to the current Beijing year end |
| `--derived-bar-sizes` | 聚合周期，逗号分隔，默认 `5,15` / Comma-separated bar sizes, default `5,15` |
| `--ts-code` | 为指数行情或权重选择单个指数 / Select one index for daily prices or weights |
| `--rebuild` | 仅对支持该参数的组生效，例如分红 / Rebuild only where supported, such as dividends |

大部分组的结束日期默认北京时间当天；指数权重默认取上一完整月末。三大报表和基金持仓未指定开始日期时从当天判断报告期，不自动补齐全部历史。不要把 `--groups all` 当作轻量日常更新，它会调用所有注册组。

Most groups default to today's Beijing date; index weights default to the prior complete month end. Statements and fund holdings default to checking today's reporting period rather than backfilling all history. The all-groups option invokes every registered group and is not a lightweight daily update.

```bash
python scripts/update_incremental.py --help
```

更多存储粒度见[下载粒度说明](DOWNLOAD_GRANULARITY.md)，基金接口见[基金数据说明](fund_data.md)。
See [download granularity](DOWNLOAD_GRANULARITY.md) and [fund data](fund_data.md) for additional details.
