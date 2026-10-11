# Fund Holdings Factors / 基金持仓因子开发

## Data Access And Ownership / 数据访问与所有权

Declare `DatasetId::FundPortfolio` and consume `data.fund_portfolio_reader()`.
The source is `data/fund_data/portfolio/{ann_year}.parquet`, partitioned by
announcement year, not report-period year. No downloader changes or new
download metadata are required.

依赖声明使用 `DatasetId::FundPortfolio`，查询入口为
`data.fund_portfolio_reader()`。原表按公告年份保存，不按报告期年份保存。
本次不改变下载器，不新增本地观察时间或其他下载字段。

The shared index keeps raw disclosure groups identified by
`(ts_code, end_date, ann_date)`, with row offsets into immutable annual sources.
It does not remove duplicates, resolve conflicts, filter securities or select
the largest holdings. These policies belong to the factor.

共享索引按 `(基金代码, 报告期, 公告日)` 定位原始披露组，紧凑行号引用年度
source。不在数据层去重、解决同键冲突、过滤证券或筛选前十大；这些规则在
因子代码中实现。`FundDisclosureView` 借用原表字符串并按行读取数值，不复制
完整持仓表。

## Batch Lifecycle / Batch 生命周期

`FinancialBatchContext` builds one fund batch view for all selected provider
batches. `DisclosureTableCache` reuses the same annual table and annual index
while the file stamp and column set remain valid. The current batch needs only
the announcement years spanning its load dates. Old sources/indexes are evicted
when no longer needed, and freed after their last shared owner drops.

一个 date batch 的所有 provider/factor batches 共用基金 batch view；年度表
和年度索引通过已有 `DisclosureTableCache` 有界复用。列集合或文件大小/修改
时间变化时重建对应年度索引。跨 batch 仅保留仍需要的公告年份；过期 source
与 index 在最后一个引用释放后回收，不逐年无界累积。

Initialisation must not discard a fund's latest visible disclosure simply
because it belongs to an old year. `bootstrap_until(as_of, callback)` streams
older files one year at a time, then visits loaded sources through `as_of`.
The callback updates factor-owned compact fund snapshots. Historical source
tables are released after each year and do not enter the persistent raw cache.
This bootstrap runs only when provider state is empty, not every batch.

首次启动调用 `bootstrap_until(as_of, callback)`，逐年度扫描旧公告文件，
callback 把必要持仓更新到因子自己的紧凑状态。每个历史年度处理完就释放原表
及索引；下一 batch 不重复 bootstrap。旧年份数据释放不能顺便删除某只基金
尚未被新披露替代的最新持仓。

Annual sources may contain future rows. Readers cap visibility by
`ann_date <= min(as_of, batch_end_date)`; factor state consumes only events
after its previous processing date and through the current trading date.
For monthly factors, consume disclosures through the selected month-end only.

原表含未来行不等于未来行可见：reader 以
`ann_date <= min(as_of, batch_end_date)` 控制 PIT，状态只消费
`(last_processed_date, trade_date]` 的公告。周末公告在下一交易日生效；
`traction20d` 在月末统一消费截至该月末的公告，月间公告不会提前改变最终截面。

## Factor Development / 因子开发顺序

1. Declare source columns and genuinely needed market inputs in `spec()`.
   在 `spec()` 声明真实使用的持仓字段、量价字段与行业字段。
2. Define provider state containing compact fund snapshots and the current
   final cross-section or genuinely needed network/window. Implement `initial_compute_state()` and
   `compute_many_stateful()` using the existing Factor interface.
   定义基金级慢状态与当前最终截面或必要网络/滚动窗口，使用现有 stateful Factor 接口。
3. Apply factor-specific disclosure policies, retaining the maximum visible
   report period and then the latest visible announcement within that period.
   因子自行处理披露规则：先选可见最大报告期，再选该期最新可见公告。
4. Choose the calculation cadence explicitly. For `traction20d`, consume
   disclosures and run all downstream calculations at trading month-end,
   then replay final values by stock code until the next month-end.
   明确计算频率：`traction20d` 在交易月末更新持仓、计算网络和回归；月间仅
   按股票代码复用最终值，不复用跨 batch 的裸数组位置。
5. Keep only the current final cross-section; release temporary monthly graphs
   after finalization. Complete daily output using the usual engine writer.
   状态中仅保留当前最终截面，月末临时网络完成计算后释放，每日输出走现有引擎。

## Traction20d Rules / Traction20d 口径

`traction20d` is deprecated; its historical factor-library column has been removed.
The implementation below remains as a development reference, not an active
tag-selection target. `traction20d` 已弃用，历史因子列已清理；以下保留为开发参考。

- The object is **each fund's top ten stock holdings**, not each stock's top
  ten shareholders. Current SH/SZ-only storage gives an A-share top-ten
  approximation; funds are counted independently by `ts_code`.
  对象是每只基金前十大持仓，不是股票前十大股东；现有库仅保留 SH/SZ，因此
  使用 A 股前十大近似，不合并基金不同份额代码。
- In `traction20d.rs`, exact original-field duplicates are ignored. Different
  numeric values under `(fund, stock, end_date, ann_date)` invalidate the whole
  disclosure, including conflicts outside the eventual top ten. No summation,
  arbitrary selection or fallback to an old disclosure is allowed.
  因子层删除原字段完全重复行；同键数值冲突则整次基金披露置为无效状态，
  清除旧贡献且不回退旧披露。筛选前十大之前检测冲突并输出 warning。
- Valid securities match `^\d{6}\.(SH|SZ)$`. Finite positive `mkv` is sorted
  descending, ties by stock code, then truncated to ten. Zero ratios alone do
  not remove a holding. A late old-period report cannot overwrite a newer one.
  仅保留标准 SH/SZ 股票和有限正持仓市值；市值降序、代码稳定破同值后取十只。
  持仓比例为零本身不是删除条件，迟到的旧报告期不能覆盖新报告期。
- At each trading month-end, consume all newly visible disclosures, compute
  `AMT = mean(amount, 20, min_periods=1)` and
  `K_ij = sum_f min(H_fi / AMT_i, H_fj / AMT_j)`. Refresh all fund contributions
  using the same month-end AMT even if no new disclosure exists. Each fund produces at most 45 pairs;
  pair aggregation is sparse, symmetric and excludes self edges.
  每个交易月末用当时可见最新持仓和月末20日平均成交额构网；即使当月无新
  公告，也使用该月末成交额统一刷新所有基金贡献。
  每只基金最多45个股票对，不构造全市场 dense 矩阵，不使用内部 Rayon。
- Convert stock `amount` from thousand CNY to CNY before dividing fund `mkv`
  (CNY). Fund `amount` means share quantity and is **not** stock turnover.
  股票成交额乘1000统一到元；基金表 `amount` 是持股数量，不是成交额。
  Units: [Tushare daily](https://tushare.pro/document/2?doc_id=27),
  [Tushare fund portfolio](https://tushare.pro/document/2?doc_id=121).
- Returns use adjusted close (`close * adj_factor`). Compound available daily
  returns over the last 20 sessions, `min_periods=1`; do not bridge missing
  daily prices. A rolling result with no valid daily return is null.
  收益用调整价相邻交易日收益在20日内复合，有一个有效样本即可；缺日不跨日
  拼接收益，完全无有效日收益为空。完整窗口需21个价格点。
- At month-end, subtract the valid held-stock pool return median to get
  Alpha. `raw_i = sum_j(K_ij * Alpha_j) / count(valid distinct peers)`.
  Peers with no usable 20-day return are excluded from that day's sum and
  denominator; no available peers gives null, not zero.
  月末用持仓股票池有效收益中位数算 Alpha；原始值除以可得不同关联股票数量，
  不是边权之和。月末缺收益邻居不进分子/分母。
- Month-end equal-weight regression controls own Alpha and SW L1 fixed effects,
  with an intercept-equivalent grouped demean. No SIZE, extra zscore
  or between-month re-neutralization is used. Replay the final residual cross-section
  until the next trading month-end.
  月末做自身 Alpha 与申万一级行业中性化，不加 SIZE，不额外 zscore；月间
  复用最终回归残差，不重算收益、不重建网络、不重新中性化。
- Initialization reconstructs the last completed trading month-end at or
  before the first target date, never an arbitrary start-date snapshot.
  `lookback.trading_days=52` conservatively covers the prior month-end plus
  its 20-return window; the calculation itself still uses 20 sessions.
  Month-end is confirmed by a next-month trading session in the full calendar,
  not by a truncated batch boundary. No known month-end means null output.
  首次启动按首日之前最近已完成交易月末重建截面；首日恰为月末则计算当日。
  预读52个交易日覆盖上月末及其20日收益窗口，实际计算窗口仍为20日。
  月末由完整交易日历中下一交易日跨月确认，不把 batch 尾日当月末。
  未知历史月末时输出空值。跨 batch 保留当前最终截面，不强制首日重算。
- Final values are keyed by stock code and replayed only for stocks present
  on the output date. Newly listed stocks without a cached value remain null;
  delisted stocks are not emitted. Announcement dates later than the monthly
  anchor are never consumed when reconstructing that anchor.
  最终值按股票代码缓存，当天在市才回放；新上市且无缓存股票为空，退市后
  不输出。月间已发生、但晚于上月末的公告不能用于重建上月末截面。

## FOA Rules / FOA 口径

`foa` is deprecated and excluded from default tag selection. Its historical
factor column has been cleared; the following rules remain as a reference.
`foa` 已弃用并清理历史因子列，以下口径仅作为开发参考。

`foa` combines a half-year ownership change signal and a quarterly overweight
residual. Compute only at completed trading month-ends and hold the **final**
cross-section by stock code between anchors; do not re-neutralize every day.
`lookback.trading_days=32` covers initialization at the previous month-end.
Sparse market inputs must not reduce the daily universe output grid.

`foa` 将半年持仓超配变化与季度超配回归残差合成。仅在已完成交易月末计算，
月间按股票代码持有最终截面，不每天重新中性化。首次预读32个交易日覆盖上个
月末；市值和 Barra 的稀疏取数日期不能代替日频 universe 输出日期。

- Declare projected `FundBasic` columns `fund_type/invest_type/name`; its source
  is `data/fund_data/basic/fund_basic.parquet`, a current classification snapshot.
  Keep stock/mixed funds, **including enhanced index funds**; exclude passive
  index, ETF/feeder and QDII. Ignore current fund survival status, and count
  each fund/share-class code independently. Declare the existing eight
  `FundPortfolio` columns, `StockDailyBasic.total_mv`, and Barra
  `VOLATILITY/MOMENTUM/SIZE`. No PV anchor or industry exposure is required.
  当前分类近似筛选股票型/混合型，保留增强指数；排除被动指数、ETF/联接、QDII，
  不按当前存续状态剔除历史基金，各份额代码分别统计。
- Select the latest common visible quarterly report period, then each fund's
  latest disclosure for that period, taking the ten largest valid SH/SZ stock
  holdings by `mkv`. Half-year full holdings use the confirmed source heuristic:
  June 30 reports announced on/after August 1, December 31 reports on/after
  February 1 next year. Holding count does not determine report type.
  季度分支取最新共同季度及各基金该期最新可见披露的十大；全持仓按8/1、次年2/1
  日期近似识别，不用持仓数量区分季度与半年披露。
- For report period `p`, use market caps from the last open date at/before `p`.
  `FOP_i = (H_i / sum(H)) / (MV_i / sum(MV)) - 1`; both normalized denominators
  use stocks with valid positive market caps from the legal SH/SZ universe.
  An unheld stock with a valid cap has FOP=-1. Fund `mkv` and market cap units
  cancel within their separately normalized weights.
  用报告期末前最后交易日总市值构造市场权重，合法非BJ股票参与分母。
- Half-year FOC uses the latest visible half-year period and its immediately
  preceding half-year. Only funds with valid full holdings in **both** periods
  contribute to either aggregate. `FOC=(1+FOP_now)/(1+FOP_old)-1`; missing prior
  holdings or a zero prior ownership weight gives null, not zero.
  两期共同有效基金构造两个持仓池；上期无持仓导致分母为零时为空。
- Equal-weight OLS with intercept residualizes quarterly FOP against
  `VOLATILITY`, `MOMENTUM`, `SIZE`; **no industry neutralization**. Separately
  cross-sectionally zscore FOC and quarterly residuals, then average them.
  Both standardized components must be valid; no extra final neutralization.
  季度超配对三个暴露做带截距等权回归；与半年变化分别标准化后等权平均，
  严格要求两个子项有效，不加行业回归或合成后的二次回归。
- Filter nonstandard codes/BJ. Exact duplicates contribute once; matching
  original numeric fields permit `mkv` rounding differences up to 0.5, using
  their maximum deterministically. Larger differences or other numeric
  conflicts invalidate the whole fund disclosure and clear its old contribution.
  Market and holdings aggregation use stable ordering for batch reproducibility.
  同键冲突清空该次基金贡献；固定汇总顺序保证跨 batch 可重复。
- State retains only eligible codes, current quarterly snapshots, two half-year
  snapshots, at most three report-end market-cap maps, and the current final
  cross-section. Historical sources bootstrap one announcement year at a time,
  then release; future announcements never enter the selected monthly anchor.
  状态有界保留必要快照和最终截面；历史原表逐年释放，不无界累计。

## Run And Tags / 运行与标签

Both fund-holdings factors below are deprecated; do not include them in production.
以下两个基金持仓因子均已弃用，不再作为生产目标。

`foa` retains `KYZQ, alternative, deprecated` (not `fundamental`).
`traction20d` retains `KYZQ, alternative, cs_network, deprecated`.
`mainbz_mom_resvol` additionally has `alternative`; its existing fundamental
tag and calculation are unchanged. 基金持仓因子不进入 fundamental 标签选集；
主营业务因子补充 alternative 标签，不改变原有公式和其他标签。
