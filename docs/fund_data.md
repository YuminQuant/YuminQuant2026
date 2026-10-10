# 基金数据接入 / Fund Data

## 范围 / Scope

`FundBasicDownloader` 调用 `fund_basic`；`FundPortfolioDownloader` 调用 `fund_portfolio`。复用 BaseDownloader、TushareClient、配置、日志和财报 schema-preserving concat。没有 report_type，不直接继承三大报表下载器。本版不含 Rust reader、持仓因子或基金份额合并。

The two downloaders reuse the existing client, configuration, logger, throttling and schema-preserving concatenation. They do not inherit statement report-type semantics. Rust readers, factors and share-class consolidation are out of scope.

## 调用方式 / Commands

以下命令会实际下载，开发验证仅使用 mock，不运行这些命令。Basic 总是当前基础信息快照，`--end-date` 仅约束持仓，不能生成过去时点的基金基础信息。

These commands perform real downloads. Development tests use mocks only. Basic data is a current snapshot; `--end-date` limits portfolio periods and announcement dates, not basic information.

```powershell
# Historical initialization / 历史初始化
python scripts/init_fund_data.py --start-year 2009 --end-date 20260424
# Portfolio only / 只下载持仓
python scripts/init_fund_data.py --datasets portfolio --start-year 2009 --end-date 20260424
# Incremental updates / 增量更新
python scripts/update_incremental.py --groups fund_basic fund_portfolio --start-date 20260401 --end-date 20260424
# Exact announcement interval / 不额外回看公告日期
python scripts/update_incremental.py --groups fund_portfolio --start-date 20260401 --end-date 20260424 --fund-ann-lookback-days 0
```

新增组不加入 DEFAULT_GROUPS，但显式 `--groups all` 会包含基金组。持仓增量不传起日时，以结束日为起点再回看默认 7 个自然日，不默认重拉全部历史。没有持久化完成游标，重跑依靠幂等合并；更早的迟到或修订需扩大公告范围或重拉报告期。

The groups are opt-in, except explicit `--groups all`. Without a start date, incremental portfolio updates start at the end date minus the configured lookback. There is no persistent completion cursor; retries are idempotent. Older corrections require a wider interval or a historical period refresh.

## 存储与字段 / Storage And Schema

```text
data/fund_data/basic/fund_basic.parquet
data/fund_data/basic/snapshots/{fetch_date}.parquet
data/fund_data/portfolio/{ann_year}.parquet
data/fund_data/portfolio/revisions/{ann_year}.parquet
```

Basic 遍历 `market=E/O`、`status=D/I/L`，保留已退市/到期基金；保存官方全部字段和本地 `fetch_date`，同一天快照重跑覆盖。基金名称、管理人和当前类型不是历史 PIT 字段，不能倒推历史。此接口没有可靠的统一份额组合 ID。

Basic covers both markets and all three statuses, including expired/delisted funds. All documented fields and the local fetch date are retained. Same-day snapshots are replaced. Current names, managers and types are not historically point-in-time, and no reliable share-class portfolio identifier is supplied.

Portfolio 官方字段：`ts_code` 是基金代码，`symbol` 是股票代码，`ann_date` 是公告日，`end_date` 是持仓报告期；`mkv` 为元，`amount` 为股。`stk_mkv_ratio`、`stk_float_ratio` 保留供应商原始比例，不擅自除以 100 或解释未核实的分母。增加 UTC `first_seen_at/last_seen_at` 记录本地观察时间。

Portfolio preserves fund/stock keys, announcement/report dates, market value in yuan, shares, and original vendor ratios. No guessed ratio scaling is applied. UTC first/last observation timestamps are ingestion provenance, not disclosure dates. Numeric values are float64; date columns are nullable Int32 YYYYMMDD; all-null columns are retained. Missing schema columns or invalid key dates fail explicitly; nonfinite numeric values become null, not zero.

## 分页与故障 / Pagination And Failures

配置 `api.page_limits.fund_basic=15000`（按官方上限限制），`fund_portfolio=1000` 是请求值，不是已确认的服务端上限。`offset` 按实际返回行数递增，即使短页也继续请求到空页。整页无新增记录或达到 max_pages 就报错，不把重复页当作完成。每次请求及重试均经过 BaseDownloader 限速；异常耗尽重试后传播，不转成空表。None 响应也不是正常结束。

Offset advances by actual returned rows and stops only on an empty frame. No-progress pages and page limits fail loudly. Every attempt is throttled; exhausted retries propagate. Official pages do not explicitly document offset support, so real API completeness and pagination support still require verification on the first live run. Mock tests cannot establish server behavior.

`api.rate_limits.fund_basic/fund_portfolio` 默认每分钟 180 次，BaseDownloader 另有 90% 安全系数；按账号权限调整。`fund_download.retries=3`、`max_pages=100000`、`ann_lookback_days=7` 可配置。

Rate settings default to 180 calls/minute with the existing 90% safety factor; configure them for your account. Retry count, maximum pages and announcement lookback are independently configurable.

分页持仓按公告年暂存到临时目录，整次查询完成、各年冲突检查通过后才合并；只合并一个公告年，不载入全部历史。每个文件校验行数/schema 后原子替换，下载目录持有排他锁。多个年份不是整体事务；磁盘故障可能已有某年成功写入，重跑可以补齐，不记录虚假的整体成功。异常退出的锁只能在确认无写入进程后手动清理。

Portfolio pages are staged on disk. All query partitions are checked before publication; only one announcement year is merged at a time. Each file is validated and atomically replaced under an exclusive directory lock. Multi-year publication is not one transaction; reruns repair partial publication after storage failures. Basic is a small table and is assembled in memory. Deduplication hashes are retained only for the current query.

## 重复持仓与 PIT / Duplicates And PIT

业务键是 `(ts_code, symbol, end_date, ann_date)`。完全重复行只留一条，不求和。同次查询同键数值冲突直接报错。不同公告日分别保留；同键跨次发生修订时，将旧行写入 revisions，再用新值更新主表。相同值重跑保留 first_seen_at 并更新 last_seen_at。未返回旧行时，不删除它，更不推断持仓清零。

Exact duplicates are removed without summing. Conflicting values within a query fail. Different disclosure dates remain separate. Later same-key corrections archive the prior observed row before replacement. Absence from a response never implies zero holdings. A/C share classes remain separate raw records; downstream aggregation must resolve portfolio identity before summing.

后续 PIT 至少要求 `ann_date <= trade_date`，不能把后披露的持仓提前并入截面，也不能直接对所有公告版本求和。主表是供应商最新观察版本，不保证严格历史原貌；revisions 只能帮助恢复本地采集开始后的变化，无法恢复供应商之前已经覆盖的历史。基础表的 fetch_date 也不等于公告日期。

Future readers must filter by disclosure date and select versions rather than summing all announcements. The main table reflects latest observed vendor values, not a guaranteed vintage archive. Revision audit only covers changes observed since local collection began; it cannot reconstruct earlier overwritten vendor history.

## 验证 / Validation

运行 `python -B -m pytest tests/test_fund_downloaders.py`。测试使用模拟 API 和临时 parquet，覆盖分页、失败保护、版本、空列、公告年分区、重试、锁和原子写入，不访问 Tushare。

Tests use mocked APIs and temporary parquet files, never the live service. No production fund data is downloaded during development validation.
