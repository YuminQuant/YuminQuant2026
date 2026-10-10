# 基金数据接入 / Fund Data

## 范围 / Scope

覆盖 Tushare 公募基金栏目：基础信息、持仓、管理人、经理、业绩基准、规模、净值、分红和技术面因子。复用 BaseDownloader、TushareClient、配置、日志和财报 schema-preserving concat。没有 report_type，不直接继承三大报表下载器。本版不含 Rust reader、持仓因子或基金份额合并。

The public-fund downloaders reuse the existing client, configuration, logger, throttling and schema-preserving concatenation. They do not inherit statement report-type semantics. Rust readers, factors and share-class consolidation are out of scope.

| 类 / Class | Tushare API | 分区 / Partition under `data/fund_data` |
| --- | --- | --- |
| FundBasicDownloader | fund_basic | `basic/fund_basic.parquet` + fetch-date snapshots |
| FundPortfolioDownloader | fund_portfolio | `portfolio/{ann_year}.parquet` |
| FundCompanyDownloader | fund_company | `company/fund_company.parquet` + snapshots |
| FundManagerDownloader | fund_manager | `manager/{ann_year}.parquet` |
| FundBenchmarkDownloader | mkt_idx_bmk | `benchmark/mkt_idx_bmk.parquet` + snapshots |
| FundShareDownloader | fund_share | `share/{trade_date}.parquet` |
| FundNavDownloader | fund_nav | `nav/{nav_date}.parquet` |
| FundDividendDownloader | fund_div | `dividend/{ann_year}.parquet` |
| FundFactorProDownloader | fund_factor_pro | `factor_pro/{trade_date}.parquet` |

业绩基准接口返回官方基准指数目录，不是逐基金的基准收益序列。基金行情和复权因子继续使用现有 ETF 下载器的 `fund_daily/fund_adj`，不重复建设。基金规模 `fund_share` 的官方范围为沪深 ETF，不应当成所有场外基金的份额库。

The benchmark API supplies a benchmark-index catalog, not per-fund benchmark return histories. Existing ETF downloaders remain responsible for `fund_daily/fund_adj`. The documented `fund_share` coverage is Shanghai/Shenzhen ETFs, not all OTC funds.

## 调用方式 / Commands

以下命令会实际下载。Basic/company/benchmark 总是当前资料快照，忽略 `--start-year/--end-date`，不能生成过去时点的基础信息。

These commands perform real downloads. Basic/company/benchmark are current snapshots; historical start/end arguments do not reconstruct their historical vintages.

```powershell
# Historical initialization / 历史初始化
python scripts/init_fund_data.py --start-year 2009 --end-date 20260424
# Portfolio only / 只下载持仓
python scripts/init_fund_data.py --datasets portfolio --start-year 2009 --end-date 20260424
# Refresh a report period / 按报告期增量重拉（包含查询时已可得的后续公告）
python scripts/update_incremental.py --groups fund_basic fund_portfolio --start-date 20260331 --end-date 20260331
# Refresh all quarter-end periods in range / 重拉区间内季度报告期
python scripts/update_incremental.py --groups fund_portfolio --start-date 20250101 --end-date 20260331
# Optional sources / 其他接口需显式选择
python scripts/init_fund_data.py --datasets company manager benchmark share nav dividend factor_pro --start-year 2009 --end-date 20260424
python scripts/update_incremental.py --groups fund_company fund_manager fund_benchmark fund_share fund_nav fund_div fund_factor_pro --start-date 20260401 --end-date 20260424
```

`fund_basic` 和 `fund_portfolio` 已加入 DEFAULT_GROUPS。基础快照每次刷新；持仓与三大财报使用相同调度：不传日期时使用当天，仅处理区间内 `0331/0630/0930/1231` 报告期，其他日期打印 skip。增量通过 `period` 重拉，不按 ann_date 查询、不使用 7 日回看；`--fund-ann-lookback-days` 只影响其他日期型基金接口。其他基金组仍需显式指定，`--groups all` 包含全部组。没有持久化完成游标，后续披露或修订通过重拉对应报告期补齐。

Both sources are in DEFAULT_GROUPS. Basic refreshes on every run; portfolio follows statement scheduling: default dates are today, and only quarter-end dates within the range trigger period refreshes. Other dates log a skip. Portfolio no longer queries by announcement date or applies the seven-day overlap; that option remains for other dated fund sources. Refresh the same period again to capture later disclosures or corrections. Other fund groups remain opt-in except explicit `--groups all`.

历史按季度 period 分页下载，`--end-date` 同时约束报告期和公告日。增量入口 `sync(mode="incremental", target_date="20260331")` 中 target_date 是报告期，不是披露截止日，默认保留供应商当前返回的该期全部公告；所以今天重拉旧报告期，会纳入报告期之后的公告。调度脚本的 start/end 只选择报告期，不作为增量公告过滤条件。直接调用 sync 时可另传 end_date 作为可选公告截止日。

History enumerates quarterly periods and applies the explicit history cutoff to both report and announcement dates. Incremental target_date selects a report period, not an announcement cutoff; later announcements returned by the vendor remain included. Script start/end dates select periods only. Direct sync calls may optionally pass a separate end_date cutoff.

历史 CLI 默认仍只下载 basic/portfolio。新增日期型接口逐自然日查询（公告和净值不限定交易日），内存仅合并当前查询及当前分区，不积累全历史。经理/分红以 ann_date 查询；规模/技术面以 trade_date 查询；净值以 nav_date 查询。增量均回看配置的 7 天；更早修订需要扩大区间。净值的截止日期限制 nav_date，不删除晚于 nav_date 的 ann_date；后续 PIT 必须额外检查公告日。

The default historical selection stays basic/portfolio. Additional dated sources query one natural day at a time and retain only the current query/partition in memory. Managers/dividends use announcement dates, share/factor data use trade dates, and NAV uses valuation dates. Incremental queries apply the configured seven-day overlap. NAV history cutoffs constrain NAV dates, not disclosure dates; PIT consumers must still check `ann_date`.

新增接口重复键：manager `(ts_code,ann_date,name,begin_date)`；nav `(ts_code,nav_date,ann_date)`；dividend `(ts_code,ann_date,base_date,div_proc,ex_date)`；share/factor `(ts_code,trade_date)`。同查询同键冲突报错，不任意求和或选一条；跨次修订归档至 `revisions`。可空事件属性保持为空，不能伪造公告日。年度分区逐日合并会重复写当年文件，优先保证幂等与有界内存，后续可按实测再优化写入批次。

The keys above preserve disclosed events and share classes. Same-query conflicting keys fail; later corrections are archived under `revisions`. Nullable event attributes remain null, with no invented disclosure dates. Annual partitions are merged per query, trading extra yearly-file IO for bounded memory and idempotence.

## 存储与字段 / Storage And Schema

```text
data/fund_data/basic/fund_basic.parquet
data/fund_data/basic/snapshots/{fetch_date}.parquet
data/fund_data/portfolio/{ann_year}.parquet
data/fund_data/portfolio/revisions/{ann_year}.parquet
```

持仓按 ann_date 年份保存，与三大财报分区一致，不改成每日小文件。主文件、修订文件及分页暂存均使用 ZSTD 压缩；读取兼容原有 Parquet，已有文件仅在下次合并写入时转换压缩方式，不主动全库重写。保留 float64 数值和 Int32 日期，不为了压缩降低数值精度。

Portfolio uses announcement-year partitions, matching statements rather than daily files. Main, revision and staged Parquet files use ZSTD. Existing files remain readable and switch compression on their next merge; there is no automatic full-store rewrite or numeric precision reduction.

Basic 遍历 `market=E/O`、`status=D/I/L`，保留已退市/到期基金；保存官方全部字段和本地 `fetch_date`，同一天快照重跑覆盖。基金名称、管理人和当前类型不是历史 PIT 字段，不能倒推历史。此接口没有可靠的统一份额组合 ID。

Basic covers both markets and all three statuses, including expired/delisted funds. All documented fields and the local fetch date are retained. Same-day snapshots are replaced. Current names, managers and types are not historically point-in-time, and no reliable share-class portfolio identifier is supplied.

Portfolio 官方字段：`ts_code` 是基金代码，`symbol` 是股票代码，`ann_date` 是公告日，`end_date` 是持仓报告期；`mkv` 为元，`amount` 为股。`stk_mkv_ratio`、`stk_float_ratio` 保留供应商原始比例，不擅自除以 100 或解释未核实的分母。增加 UTC `first_seen_at/last_seen_at` 记录本地观察时间。

Portfolio preserves fund/stock keys, announcement/report dates, market value in yuan, shares, and original vendor ratios. No guessed ratio scaling is applied. UTC first/last observation timestamps are ingestion provenance, not disclosure dates. Numeric values are float64; date columns are nullable Int32 YYYYMMDD; all-null columns are retained. Missing schema columns or invalid key dates fail explicitly; nonfinite numeric values become null, not zero.

## 分页与故障 / Pagination And Failures

进度沿用 QuantLogger，同时输出到控制台和 `data_sync.log`：报告期显示 `[当前/总数]` 和 `completed`，每页显示 offset、limit、返回行数及当前查询累计 fetched_rows，空页明确提示结束。查询完成另报 cutoff 过滤后 retained_rows 和去重后 unique_rows；这些不是年度文件总行数。请求前先打印进度，失败重试打印 attempt；阻塞中的单次网络请求不额外启动心跳线程。

Progress uses QuantLogger for both console and `data_sync.log`: period position/completion, per-page offset/limit/rows, cumulative fetched rows per query, and explicit empty-page messages. Completion includes cutoff-retained and deduplicated incoming rows, distinct from the total stored yearly rows. Requests log before network IO and retries log attempt counts; there is no background heartbeat during a blocked request.

配置 `api.page_limits.fund_basic=15000`，`fund_portfolio=8000`。2026-10-10 实测全市场 period=20251231：不传 limit/offset 返回 8000 行，请求 10000 仍返回 8000；offset=7995 的跨页重叠五行一致。此为当前账号实测，不保证未来服务端上限不变。`offset` 按实际返回行数递增，即使短页也继续请求到空页。整页无新增记录或达到 max_pages 就报错，不把重复页当作完成。每次请求及重试均经过 BaseDownloader 限速；异常耗尽重试后传播，不转成空表。None 响应也不是正常结束。

Portfolio now requests 8000 rows per page. A live 20251231-period query on 2026-10-10 returned 8000 rows both without pagination arguments and with limit=10000; an offset=7995 overlap check confirmed continuity. This is an observed account-specific limit, not a permanent API guarantee.

Offset advances by actual returned rows and stops only on an empty frame. No-progress pages and page limits fail loudly. Every attempt is throttled; exhausted retries propagate. Live validation of basic/portfolio checks a bounded sample only, not full-history completeness. Newly added APIs have only been syntax-checked; their server-side pagination still needs validation before large downloads. Company is the documented single-response, no-parameter exception.

新增分页请求值：manager/benchmark 5000，share 2000，factor_pro 8000；nav/dividend 默认请求 1000（不是宣称服务端上限）。每页继续到空页，拒绝静默截断。基金接口统一配置每分钟 500 次，附加 90% 安全系数；权限不足直接报错，需按账号等级配置。全空列保留，缺列报错，非有限数值转空，不填零。

New page sizes are 5000 for managers/benchmarks, 2000 for shares, 8000 for technical factors, and a configurable requested 1000 for NAV/dividends. Fund interfaces are configured for 500 calls/minute with the existing safety margin; adjust for account permissions. All-null columns are preserved, missing columns fail, and nonfinite numeric values become null rather than zero.

`api.rate_limits.fund_basic/fund_portfolio` 默认每分钟 500 次，BaseDownloader 另有 90% 安全系数，每请求等待约 0.133 秒；按账号权限调整。`fund_download.retries=3`、`max_pages=100000`、`ann_lookback_days=7` 可配置。

Rate settings default to 500 calls/minute with the existing 90% safety factor (about 0.133 seconds of sleep per attempt); configure them for your account. Retry count, maximum pages and announcement lookback are independently configurable.

分页持仓按公告年暂存到临时目录，整次查询完成、各年冲突检查通过后才合并；只合并一个公告年，不载入全部历史。每个文件校验行数/schema 后原子替换，下载目录持有排他锁。多个年份不是整体事务；磁盘故障可能已有某年成功写入，重跑可以补齐，不记录虚假的整体成功。异常退出的锁只能在确认无写入进程后手动清理。

Portfolio pages are staged on disk. All query partitions are checked before publication; only one announcement year is merged at a time. Each file is validated and atomically replaced under an exclusive directory lock. Multi-year publication is not one transaction; reruns repair partial publication after storage failures. Basic is a small table and is assembled in memory. Deduplication hashes are retained only for the current query.

## 重复持仓与 PIT / Duplicates And PIT

业务键是 `(ts_code, symbol, end_date, ann_date)`。完全重复行只留一条，不求和。持仓同次查询唯一精度例外：除 mkv 外所有字段（含缺失状态）相同，且 mkv 只有一个带小数的有限值及其四舍五入整数值时，保留带小数原记录并 warning 输出键和金额。实测 20170630 的 001875.OF/00981.HK 有 2395676.18 和 2395676.00 两种市值。其他同键冲突仍报错，并附具体键和记录。不同公告日分别保留；同键跨次发生修订时，将旧行写入 revisions，再用新值更新主表。相同值重跑保留 first_seen_at 并更新 last_seen_at。未返回旧行时，不删除它，更不推断持仓清零。

Exact duplicates are removed without summing. Portfolio permits one narrow precision exception: all other fields, including missingness, must match; mkv must contain one finite fractional value and its rounded integer. The fractional original is retained independently of page order, with a warning containing the key and amounts. All other conflicts still fail with diagnostic records. Different disclosure dates remain separate. Later same-key corrections archive the prior observed row before replacement. Absence from a response never implies zero holdings. A/C share classes remain separate raw records; downstream aggregation must resolve portfolio identity before summing.

后续 PIT 至少要求 `ann_date <= trade_date`，不能把后披露的持仓提前并入截面，也不能直接对所有公告版本求和。主表是供应商最新观察版本，不保证严格历史原貌；revisions 只能帮助恢复本地采集开始后的变化，无法恢复供应商之前已经覆盖的历史。基础表的 fetch_date 也不等于公告日期。

Future readers must filter by disclosure date and select versions rather than summing all announcements. The main table reflects latest observed vendor values, not a guaranteed vintage archive. Revision audit only covers changes observed since local collection began; it cannot reconstruct earlier overwritten vendor history.

## 验证 / Validation

运行 `python -B -m pytest tests/test_fund_downloaders.py`。测试使用模拟 API 和临时 parquet，覆盖分页、失败保护、版本、空列、公告年分区、重试、锁和原子写入，不访问 Tushare。

Existing unit tests use mocked APIs. This extension does not execute tests or live downloads for the seven new interfaces, only Python syntax compilation and configuration parsing. The explicitly requested basic/portfolio live smoke check uses an isolated temporary root; its output is moved to Recycle Bin after inspection. Production fund data is not modified.

本次新增的七个接口仅做语法编译和配置解析，不运行测试、不联网下载；basic/portfolio 的小范围实测使用独立临时目录，完成后移入回收站，不修改正式基金数据。

官方参考 / Official references: [管理人 Company](https://tushare.pro/document/2?doc_id=118), [经理 Manager](https://tushare.pro/document/2?doc_id=208), [基准 Benchmark](https://tushare.pro/document/2?doc_id=462), [规模 Share](https://tushare.pro/document/2?doc_id=207), [净值 NAV](https://tushare.pro/document/2?doc_id=119), [分红 Dividend](https://tushare.pro/document/2?doc_id=120), [技术面 Technical Factors](https://tushare.pro/document/2?doc_id=359).
