# Analysis Tools / 回测分析工具

`yq_analysis` 读取因子回测与策略模拟结果，生成收益、超额收益、IC、换手及账户诊断表，并绘制分组和策略对比图。它不执行因子计算，也不替代回测引擎。

`yq_analysis` reads factor backtests and strategy simulations, builds return/excess-return, IC, turnover and account diagnostics, and plots portfolio or strategy comparisons. It does not calculate factors or replace the backtest engine.

## 安装 / Installation

在仓库根目录执行；绘图需要可选的 Matplotlib 依赖。
Run from the repository root; plotting requires the optional Matplotlib dependency.

```bash
python -m pip install -e "./analysis_tools[plot]"
```

## 因子回测 / Factor backtests

先用 Rust backtest 生成结果。读取根目录和因子 ID 必须与该次回测匹配。
Generate results using Rust backtest first. Match the input root and factor ID to that run.

```python
from yq_analysis.io import load_backtest_result
from yq_analysis.report import make_backtest_report
from yq_analysis.plots import plot_return_summary

result = load_backtest_result("data/backtest/stock/daily", "ret20")
returns = result["returns"]
ic = result["ic"]
ic_main = (
    ic.query("horizon == 1")
    if ic is not None and "horizon" in ic.columns
    else ic
)
report = make_backtest_report(
    returns, ic_main, result["factor_stats"], periods_per_year=240
)
print(report["portfolio_total"])
print(report["excess_total"])
print(report["ic"])
fig = plot_return_summary(
    returns, groups=10, ic=ic,
    barra_exposure=result.get("barra_exposure"),
    index_group_returns=result.get("index_group_returns"),
    save=True,
)
```

主 IC 报告使用 horizon=1，完整 IC 表可用于衰减图，避免把不同 horizon 当作额外日期样本。`groups` 应与回测一致；缺少可选诊断表时不要假设每个图面板都有数据。

Use horizon=1 for the main IC report and the full IC table for decay plots, rather than treating horizons as extra dates. Match `groups` to the backtest. Optional diagnostic tables may be absent.

### 股票因子总报告 / Stock factor review

从项目根目录运行以下命令，增量更新现有 `data/reports/factor_review_20160101.html`。
首次执行登记已有本地文件，不用历史文件覆盖旧报告；之后新增或替换完整的五份
Parquet 结果即可更新。删除曾登记的因子目录后，下次运行会从报告移除该因子。
不读取因子元数据来决定增删，也不会删除回测文件。

Run from the project root. The existing HTML is the template and initial baseline.
Subsequent runs add or refresh complete local backtests and drop previously observed
factor directories that have been deleted. Use comparable backtest settings; the
script does not run the backtest or infer its settings.

```powershell
python scripts/chn_stock_factor_analysis.py
python scripts/chn_stock_factor_analysis.py --dry-run
python scripts/chn_stock_factor_analysis.py --refresh some_factor
python scripts/chn_stock_factor_analysis.py --remove some_factor
```

`--remove` 只修改报告，并记住排除项；用 `--refresh` 可恢复。首次运行前刚覆盖的
已有因子也可用 `--refresh` 强制结算。新因子可用 `--category "factor_id=分类"`
指定展示分类。相关性仅重算本地收益与报告版本一致的因子对；数据不齐时，
报告会说明覆盖范围。默认曲线目录为 `analysis_tools/plots/factor_review_2016_2026`，
原始和折叠曲线分别存入 `original/`、`folded/`。更新时覆盖对应因子的图片，
不为每次运行另建目录；自定义报告可用 `--plots-root` 指定专用目录。

## 期货因子评价 / Futures factor evaluation

先用 Rust 期货分钟回测生成摘要。默认的过去五日阈值模式写入 `data/backtest/future/minute_1m`。读取、结算和绘图可以分别调用。

```python
from yq_analysis.future_io import load_future_backtest_result, load_future_sectors
from yq_analysis.future_report import make_future_backtest_report, make_future_sector_metrics
from yq_analysis.future_plots import plot_future_return_summary

sectors = load_future_sectors("factor_engine/docs/FUTURE_SECTORS.md")
trailing = load_future_backtest_result("data/backtest/future/minute_1m", "yq_001")
report = make_future_backtest_report(trailing["summary"], sectors)
print(make_future_sector_metrics(report, sectors))
fig = plot_future_return_summary(
    report, title="yq_001 | Trailing 5", save="analysis_tools/plots/future/yq_001.png",
)
```

运行 `future_factor.ipynb` 会显示指标表，并将当前因子的图保存到 `analysis_tools/plots/future/{factor_id}.png`。

从项目根目录更新期货因子总报告：

```bash
python -B analysis_tools/scripts/report_future_factors.py
```

脚本扫描 `data/backtest/future/minute_1m/yq_*/summary.parquet`，更新
`data/reports/future_factor_review.html`。拉回新的摘要或替换旧摘要后，直接重跑同一条命令。
报告会复用已有结算，补画缺少或早于摘要的图到 `analysis_tools/plots/future/`。
总表在因子和类别之后先显示 absIC，默认按 absIC 降序排列，并显示摘要中的 All IC 和 CosIC。末列为「分钟值日均覆盖率」：读取 factor_stats.parquet 中 All 行的 coverage，按交易日等权平均。可按回测配置和因子类别筛选，另有板块、品种收益明细以及同配置内的日收益相关性。
因子类别按本地因子元数据的 tags 映射为简短名称。默认元数据缺少新因子时，脚本会尝试用本地已编译的引擎生成临时元数据。也可用 `--metadata` 指定更新后的元数据表。未收录的新标签会显示在「其他」类别中。
总报告的收益统计使用现有期货分析的 IC 定向日收益。年化收益取日均收益乘以 240，Sharpe 按 240 日年化，最大回撤由初始净值 1 加逐日收益累计后计算。收益有效日包括有有效日收益的零收益日；总表覆盖率独立读取分钟值日均覆盖率，不用收益有效日占比代替。收益明细沿用现有回测图的口径。
在逐因子页选择因子即可查看嵌入的回测图。报告通过项目内相对路径读取 PNG 文件。
报告底部附完整板块与品种中英文对照表，直接来自 `factor_engine/docs/FUTURE_SECTORS.md`，包含板块缩写、板块中英文名称及品种代码、中英文名称和交易所；新增映射后重跑报告会自动更新。
如需重新结算全部文件并重画图，使用 `--rebuild`。

新版摘要含 `holding_rtn` 时，总表、日收益相关性和累计收益图直接使用该列（引擎已按品种 IC 定向的1分钟持仓收益，不再二次翻转）。IC、CosIC及两端收益仍取5分钟标签，图中分别标明周期；旧摘要不含持仓收益时保留旧口径。

逐因子「分布与覆盖」页只以表格展示同目录 `factor_stats.parquet` 的现有统计量，不绘制分布图。表格先展示 All、再展示板块和品种；均值、标准差和每日分位数为交易日等权平均，极值取区间极值。「日中位数均值」并非跨品种中位数。常数日和四分位退化日不是零值占比。报告不新增上游统计字段，该文件的更新也会使报告缓存失效。1分钟持仓收益仍为合计，不增加多空分项。

默认使用4个独立进程绘图，可用 `--workers N` 按机器资源调整，跨平台采用 spawn，避免 Arrow/Matplotlib 在 fork 后共享运行状态。

### 直接从 HDFS 生成股票、期货报告

后续统一在 xz336 运行报告，输入直接读取 HDFS，不下载原始回测文件。使用部署的认证执行入口，在其环境中执行以下命令；`<cluster.toml>` 为该部署的配置文件，配置同目录应提供 `yq_cluster` 包。认证由外部执行入口负责，不在报告脚本中存放或传递密码。

```bash
python -B analysis_tools/scripts/report_future_factors.py --hdfs-config <cluster.toml> --workers 4
python -B scripts/chn_stock_factor_analysis.py --hdfs-config <cluster.toml>
```

HDFS 输入复用部署的原生客户端、50并发和每秒1000次读取额度；没有额外限流、预读预算或数据落地缓存。绘图进程各自初始化客户端并继承认证环境。股票五类结果并发读取，文件哈希和相关性也直接读 HDFS。期货摘要、分布统计和元数据均从 HDFS 读取，缓存指纹使用远端文件大小与修改时间。可用 `--summary-root` / `--backtest-root` / `--metadata` 指定项目内相对 HDFS 路径或部署 HDFS 根路径下的绝对路径。

HTML、图片和绘图临时文件在 xz336 生成；只把 HTML 与相应 `analysis_tools/plots/` 子目录拉回本地，保持相对目录结构即可离线查看。股票脚本需要现有股票总报告作为模板和分类基线；首次从本地源切换至 HDFS 时，会重算 HDFS 中齐全的已有因子，保留原分类与结算口径。本地模式仍可使用，省略 `--hdfs-config` 即可。

指标表的 `trade_num` 是每合约日均有效阈值触发数。全体、板块和品种均先将每日上下端合计触发数除以当日有阈值和有效配对的合约数，再对有效交易日等权平均。零触发的可统计合约计入分母，连续分钟触发分别计数。

指标表首行展示 All，随后按板块展示原始 IC、CosIC、年化收益、夏普、最大回撤、卡玛、胜率、盈亏比和有效交易日数。All 和各板块的 IC、CosIC 使用回测按交易日拼接合约分钟样本后直接计算的结果，再对有效交易日等权平均。All 收益使用回测的 `all` 两端收益，按 All 平均 IC 定向后相加。各板块先按品种 IC 定向，再对有有效双端收益的品种等权。收益指标将日度 bp 转为小数后沿用股票报告函数，默认每年 240 个交易日。盈亏比为盈利日平均收益除以亏损日平均亏损的绝对值。累计图按交易日算术累加。

## 策略分析 / Strategy analysis

```python
from yq_analysis.strategy_io import load_strategy_run
from yq_analysis.plots import plot_strategy_summary

report, fills, config = load_strategy_run("data/strategy/stock", "strategy_001")
print(report["portfolio_total"])
print(report["diagnostics"])
fig = plot_strategy_summary(report, save=True)
```

`initial_cash` 必须匹配策略配置。先用完整账户历史计算收益与对账，再截取展示区间。多策略比较使用 `plot_strategy_comparison`，各报告应具有相同观察日期。

`load_strategy_run` 优先读 `trades.parquet`，也支持旧版 `fills.csv` 和 `holdings.parquet` 的成交数组。期货分析可用 `yq_analysis.strategy.futures_holding_duration(fills, minute_bars)` 计算按已平仓手数加权的平均持仓分钟数。传入覆盖整个回测期间的合约分钟行情（`symbol`、`trade_time`），按对应合约完成的交易分钟计时，剔除休盘和隔夜间隔。

Match `initial_cash` to the strategy configuration. Calculate returns and reconciliation on the full account history before slicing the display interval. Use `plot_strategy_comparison` for reports with matching observation dates.

### CTA HTML 报告

从仓库根目录运行以下命令，为指定信号生成独立的本地 HTML 文件。

```powershell
python analysis_tools/scripts/report_cta.py --signal mdl_100000
```

默认输出到 `data/reports/cta/mdl_100000.html`，可用 `--output` 指定路径。报告读取 `data/strategy/future/<signal>/` 下的单次或年度回测结果，也支持 `--root` 和 `--market-root` 指定本地回测及期货分钟行情目录。用浏览器打开文件即可离线使用。

“组合绩效”随品种范围（全品种、板块、单品种）及年度选择联动。各范围先汇总逐日盈亏、成本、成交和成交额，再重新计算收益、波动、夏普及回撤；资金基数沿用原回测，每个独立年度重置资金，未交易日期计零收益。
组合累计净盈亏曲线与同一组筛选联动，所选区间起点归零；全日期范围按净盈亏金额跨年度累加，悬停可查看对应日期的累计金额。

两个筛选器可以组合使用。品种范围支持全品种、按 [期货板块](../factor_engine/docs/FUTURE_SECTORS.md) 筛选或选择单个品种。日期范围支持全期和单个年度。品种绩效及累计净 PnL 同时响应两个筛选器。各品种的累计净 PnL 在所选区间内首个纳入交易日的前一个自然日标零，随后绘制日结曲线。组合表显示真实账户在所选日期范围的绩效。品种持仓时间按平仓交易日先用已平仓手数加权，再对所选区间内有平仓的交易日等权平均。

### CTA 品种日内复盘

```powershell
python -m pip install -e "./analysis_tools[intraday]"
python analysis_tools/scripts/report_cta_intraday.py --signal mdl_100000 --product AG --start-date 20260105 --end-date 20260424
```

报告在交易日下拉框里一次展示一天，同轴对齐分钟价格、实际成交点、持仓、原始信号和品种日内净 PnL。净 PnL 按日初归零，用实际成交成本和分钟收盘估值重建，逐日与 `product_pnl.parquet` 核对。夜盘按回测交易日归属，休市间隔在横轴上压缩。

主力合约的信号图显示 P99 和 P1。以回测交易日历取前五个交易日，先计算每天对应主力合约信号的 P99 和 P1，再分别取五日均值。任一天缺少有效信号或主力映射时，该日阈值留空。默认交易日历为上期所，可用 `--calendar` 指定。

默认输出为 `data/reports/cta/<signal>_<product>_<start>_<end>_intraday.html`，可以离线打开。回测需要本地逐笔成交、品种日结和分钟行情。原始预测与主力映射优先读本地，缺失的日期通过已有的 `xz336` 部署读取 HDFS，可用 `--hdfs-host` 修改主机，也可用 `--local-only` 只读本地。其他路径通过 `--root`、`--market-root`、`--signal-root`、`--mapping-root` 和 `--output` 设置。

## 输入与输出 / Inputs and outputs

默认因子输入位于 `data/backtest/stock/daily/{factor_id}/`，包括 `returns.parquet`、`ic.parquet`、`factor_stats.parquet` 及可选持仓、行业、Barra、指数内收益表。策略输入为对应目录的 `holdings.parquet`。

Default factor inputs live under `data/backtest/stock/daily/{factor_id}/`: returns, IC, factor statistics and optional holdings, industry, Barra and in-index returns. Strategy inputs use their `holdings.parquet`.

报表以 Python 字典/DataFrame 返回；保存图像通常位于 `analysis_tools/plots/`，不进入 Git。可用 `dpi` 等绘图参数控制输出。

Reports return Python dictionaries/DataFrames. Saved plots normally live under `analysis_tools/plots/` and remain untracked; plotting options such as `dpi` control output.

## 解释边界 / Interpretation

- 默认年化交易日数为 240，无风险收益率为 0；确认这与研究假设一致。
  Defaults use 240 annual periods and a zero risk-free rate; verify these assumptions.
- 覆盖率、收益和 IC 需要有效样本；空表不代表零收益。先检查日期、列名和上游因子是否非空。
  Coverage, returns and IC require valid samples. Empty tables do not imply zero returns; check dates, schema and upstream factor coverage.
- 策略收益已包含模拟成交规则及成本，不能简单加回成本来替代重新回测；账户模型不包含外部资金流。
  Strategy returns include execution rules and costs. Adding costs back is not a replacement for a rerun; external account cash flows are not modeled.
- 按全样本统计选择 long 侧属于事后分析，不能直接作为当时可用的交易信号。
  Choosing the long side using full-sample statistics is retrospective analysis, not a contemporaneously available signal.

回测参数见 [Rust 引擎](../factor_engine/README.md)，策略口径见[策略开发文档](../factor_engine/docs/STRATEGY_README.md)。
See the [Rust engine](../factor_engine/README.md) for backtest commands and the [strategy guide](../factor_engine/docs/STRATEGY_README.md) for execution semantics.
