# 数据更新命令备忘

本文档记录本项目常用的数据补全命令。所有命令默认在项目根目录执行：

```powershell
cd D:\yuminwu_workspace\Internship\YuminQuant
```

## 推荐补数顺序

如果本地数据已经有一段时间没有更新，推荐顺序是：

1. 先更新静态数据和日历。
2. 再更新日线数据。
3. 最后更新分钟线数据。

原因：股票和期货分钟线下载器会依赖本地日线文件来确定当天有效标的池，所以分钟线之前应先保证日线已补齐。

## 轻量静态数据更新

当前主链路只需要先补 A 股和期货相关静态数据时，运行：

```powershell
python scripts\update_incremental.py --groups calendar stock_static future_static
```

这会更新：

- 交易日历
- A 股基础信息
- 期货基础信息

## 全量静态数据更新

如果希望一次性更新所有已接入资产类别的静态表，运行：

```powershell
python scripts\update_incremental.py --groups static
```

这会更新：

- 交易日历
- A 股基础信息
- 期货基础信息
- ETF 基础信息
- 期权基础信息
- 指数基础信息、分类、成分
- 港股基础信息和港股日历
- 美股基础信息和美股日历

## 从 2026-02-14 补主链路数据

先补日线：

```powershell
python scripts\update_incremental.py --groups stock_daily future_daily --start-date 20260214
```

日线补完后，再补分钟线：

```powershell
python scripts\update_incremental.py --groups stock_minute future_minute --start-date 20260214
```

## 一条命令补主链路

如果确认静态数据、日线依赖都没有问题，也可以直接运行默认主链路：

```powershell
python scripts\update_incremental.py --start-date 20260214
```

默认主链路包含：

- 交易日历
- A 股基础信息
- A 股日线
- A 股分钟线
- 期货基础信息
- 期货日线
- 期货分钟线

不过更推荐使用“先日线、再分钟线”的两步方式，出错时更容易定位。

## 单独更新基础信息

只更新 A 股基础信息：

```powershell
python scripts\update_incremental.py --groups stock_static
```

只更新期货基础信息：

```powershell
python scripts\update_incremental.py --groups future_static
```

只更新 ETF 静态信息：

```powershell
python scripts\update_incremental.py --groups etf_static
```

只更新期权静态信息：

```powershell
python scripts\update_incremental.py --groups option_static
```

只更新指数静态信息：

```powershell
python scripts\update_incremental.py --groups index_static
```

只更新港股静态信息：

```powershell
python scripts\update_incremental.py --groups hk_static
```

只更新美股静态信息：

```powershell
python scripts\update_incremental.py --groups us_static
```

## 单独更新财务、分红、研报

财务、分红、研报属于公告驱动数据，建议单独跑。例如更新 2026-04-01 到 2026-04-24：

```powershell
python scripts\update_incremental.py --groups stock_financial --start-date 20260401 --end-date 20260424
```

## 参数说明

`--groups` 指定要运行的任务组，可以组合多个任务组。

`--start-date` 指定补数开始日期，格式为 `YYYYMMDD`。

`--end-date` 指定补数结束日期，格式为 `YYYYMMDD`。不传时默认使用北京时间当天。

`--calendar-end-date` 指定日历补到哪一天。不传时默认补到当前北京时间年份的年末。

查看所有支持的任务组：

```powershell
python scripts\update_incremental.py --help
```
