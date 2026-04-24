# 下载器粒度说明

本文档记录 `data_manager/downloader` 中各下载器的请求粒度、增量逻辑和本地落盘方式，方便后续开发增量更新脚本或排查数据缺口。

## 结论速览

- 主要的 A 股、ETF、期货、期权日线类数据是按 `trade_date` 下载全市场横截面。
- 股票/ETF/期货分钟线整体也是按交易日落盘，但内部会按标的代码批量切分后并发请求。
- 指数日线、指数权重、指数分钟线是按单个指数代码下载。
- A 股财务历史数据按报告期 `period` 下载，增量模式按公告日 `ann_date` 加股票代码 batch 下载。
- 港股/美股财务数据按报告期 `period` 加标的代码 batch 下载。
- 静态基础信息通常是全量快照式覆盖。

## A 股与通用日历

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `CalendarDownloader` | `trade_cal` | 按交易所 + 日期区间 | 比较本地 `cal_date` 覆盖范围 | 每个交易所一个 `trade_cal_*.parquet` |
| `StockBasicDownloader` | `stock_basic` | 按上市状态全量拉取 | 无增量，全量覆盖 | `stock_basic.parquet` |
| `StockDailyPVDownloader` | `daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockAdjFactorDownloader` | `adj_factor` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockDailyLimitDownloader` | `stk_limit` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockDailyBasicDownloader` | `daily_basic` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockSuspendDownloader` | `suspend_d` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockMoneyflowDownloader` | `moneyflow` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StDownloader` | `stock_st` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `StockMinuteDownloader` | `stk_mins` | 外层按交易日，内层按股票代码 batch | 按日期文件是否存在 | `stock_data/minute/YYYY/YYYYMMDD.parquet` |
| `DividendDownloader` | `dividend` | 按自然日公告日 `ann_date` | 当前实现会请求传入区间内所有自然日 | 按公告年份 `YYYY.parquet` |
| `AnalystReportDownloader` | `report_rc` | 按自然日 `report_date` | 比较本地已存在 `report_date` | 按年 `YYYY.parquet` |
| `IncomeDownloader` 等财务类 | `*_vip` / 普通财报接口 | 历史按 `period`；增量按 `ann_date` + 股票代码 batch | 历史按报告期循环；增量由调用脚本指定日期 | 按公告年份 `YYYY.parquet` |

## ETF

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `ETFBasicDownloader` | `etf_basic` | 全量分页 | 无增量，全量覆盖 | `etf_basic.parquet` |
| `ETFIndexDownloader` | `etf_index` | 全量分页 | 无增量，全量覆盖 | `etf_index.parquet` |
| `ETFDailyPVDownloader` | `fund_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `ETFAdjFactorDownloader` | `fund_adj` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `ETFShareSizeDownloader` | `etf_share_size` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `ETFMinuteDownloader` | `stk_mins` | 外层按交易日，内层按 ETF 代码 batch | 按日期文件是否存在 | `etf_data/minute/YYYY/YYYYMMDD.parquet` |

## 期货

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `FutureBasicDownloader` | `fut_basic` | 按交易所全量拉取 | 无增量，全量覆盖 | `fut_basic.parquet` |
| `FutureDailyDownloader` | `fut_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `FutureLimitDownloader` | `ft_limit` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `FutureMinuteDownloader` | `ft_mins` | 外层按交易日，内层按期货合约 batch | 按日期文件是否存在 | `future_data/minute/YYYY/YYYYMMDD.parquet` |

## 指数

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `IndexBasicDownloader` | `index_basic` | 按市场全量拉取 | 无增量，全量覆盖 | `index_basic.parquet` |
| `IndexClassifyDownloader` | `index_classify` | 按分类源/层级全量拉取 | 无增量，全量覆盖 | 按分类源/层级文件 |
| `SWDailyDownloader` | `sw_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `CIDailyDownloader` | `ci_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `IndexDailyDownloader` | `index_daily` | 按单个指数 `ts_code` + 年份区间 | 已存在的历史整年跳过，当年合并去重 | 每个指数目录下按年 |
| `IndexWeightDownloader` | `index_weight` | 按单个指数代码 + 月度区间 | 已存在的历史整年跳过，当年合并去重 | 每个指数目录下按年 |
| `IndexMinuteDownloader` | `idx_mins` | 按单个指数 `ts_code` + 年份区间 | 已存在的历史整年跳过，当年合并去重 | 每个指数目录下按年 |
| `SWMemberDownloader` | `index_member_all` | 全量分页 | 无增量，全量覆盖 | 单文件 |
| `CIMemberDownloader` | `ci_index_member` | 全量分页 | 无增量，全量覆盖 | 单文件 |

## 期权

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `OptionBasicDownloader` | `opt_basic` | 按交易所全量分页 | 无增量，全量覆盖 | `opt_basic.parquet` |
| `OptionDailyDownloader` | `opt_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `OptionMinuteDownloader` | `opt_mins` | 按单个期权合约代码 | 从本地最大 `trade_time` 继续 | 按交易所/合约单文件 |

## 港股

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `HKBasicDownloader` | `hk_basic` | 按上市状态全量拉取 | 无增量，全量覆盖 | `hk_basic.parquet` |
| `HKCalendarDownloader` | `hk_tradecal` | 按年份区间 | 无精细增量，生成后覆盖 | `trade_cal_HKEX.parquet` |
| `HKDailyDownloader` | `hk_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `HKAdjFactorDownloader` | `hk_adjfactor` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `HKMinuteDownloader` | `hk_mins` | 外层按交易日，内层按港股代码 batch | 按日期文件是否存在 | `hkg_stock_data/minute/YYYY/YYYYMMDD.parquet` |
| `HKBalanceSheetDownloader` 等财务类 | `hk_*` 财务接口 | 按报告期 `period` + 港股代码 batch | 历史年份文件存在则跳过，仅重跑近两年 | 按年 `YYYY.parquet` |

## 美股

| 下载器 | Tushare 接口 | 请求粒度 | 本地增量判断 | 落盘方式 |
| --- | --- | --- | --- | --- |
| `USBasicDownloader` | `us_basic` | 全量分页 | 无增量，全量覆盖 | `us_basic.parquet` |
| `USCalendarDownloader` | `us_tradecal` | 按年份区间 | 无精细增量，生成后覆盖 | `trade_cal_US.parquet` |
| `USDailyDownloader` | `us_daily` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `USAdjFactorDownloader` | `us_adjfactor` | 按交易日全市场横截面 | 比较本地已存在 `trade_date` | 按年 `YYYY.parquet` |
| `USBalanceSheetDownloader` 等财务类 | `us_*` 财务接口 | 按报告期 `period` + 美股代码 batch | 历史年份文件存在则跳过，仅重跑近两年 | 按年 `YYYY.parquet` |

## 增量更新建议

1. 日常补行情：先跑日历，再跑基础信息，再跑日线，最后跑分钟线。
2. 股票和期货分钟线依赖本地日线文件构造有效标的池，所以分钟线之前必须先保证日线已经补齐。
3. 大多数日线下载器只补“本地不存在的交易日”。如果要重刷最近若干天的修订数据，需要后续给下载器增加 `force_refresh` 或删除对应日期后重跑。
4. 财务、分红、研报属于公告驱动数据，建议单独任务维护，并按自然日或公告日做增量。
