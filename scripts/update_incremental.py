import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import (
    AnalystReportDownloader,
    BalanceSheetDownloader,
    CalendarDownloader,
    CashFlowDownloader,
    DividendDownloader,
    ETFAdjFactorDownloader,
    ETFBasicDownloader,
    ETFDailyPVDownloader,
    ETFIndexDownloader,
    ETFMinuteDownloader,
    ETFShareSizeDownloader,
    ExpressDownloader,
    ForecastDownloader,
    FutureBasicDownloader,
    FutureDailyDownloader,
    FutureLimitDownloader,
    FutureMinuteDownloader,
    HKBasicDownloader,
    HKCalendarDownloader,
    IncomeDownloader,
    CIMemberDownloader,
    CIDailyDownloader,
    IndexDailyDownloader,
    IndexWeightDownloader,
    IndexBasicDownloader,
    IndexClassifyDownloader,
    OptionBasicDownloader,
    OptionDailyDownloader,
    OptionMinuteDownloader,
    QuantLogger,
    StDownloader,
    StockAdjFactorDownloader,
    StockBasicDownloader,
    StockDailyBasicDownloader,
    StockDailyLimitDownloader,
    StockDailyPVDownloader,
    StockMinuteDownloader,
    StockMoneyflowDownloader,
    StockSuspendDownloader,
    SWDailyDownloader,
    SWMemberDownloader,
    USBasicDownloader,
    USCalendarDownloader,
)
from data_manager.processor import StockTradeFilterBuilder


DEFAULT_START_DATES = {
    "calendar": "20090101",
    "stock": "20090101",
    "stock_st": "20160101",
    "stock_alt": "20100101",
    "future": "20090101",
    "future_minute": "20100101",
    "etf": "20090101",
    "index_daily": "20090101",
    "index_weight": "20090101",
    "index_industry_daily": "20090101",
    "option": "20150209",
    "hk_calendar_year": 2000,
    "us_calendar_year": 1980,
}


def bj_today():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")


def iter_calendar_dates(start_date, end_date):
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


FINANCIAL_STATEMENT_SUFFIXES = {"0331", "0630", "0930", "1231"}


def is_financial_statement_period(date):
    return len(date) == 8 and date[4:] in FINANCIAL_STATEMENT_SUFFIXES


def run_task(logger, name, fn):
    logger.info(f">>> 开始任务: {name}")
    fn()
    logger.info(f"<<< 完成任务: {name}")


def update_calendar(args, logger):
    end_date = args.calendar_end_date or f"{bj_today()[:4]}1231"
    start_date = args.start_date or DEFAULT_START_DATES["calendar"]
    run_task(
        logger,
        "calendar",
        lambda: CalendarDownloader().sync(
            start_date=start_date,
            target_end_date=end_date,
        ),
    )


def update_stock_static(logger):
    run_task(logger, "stock_static", lambda: StockBasicDownloader().sync())


def update_index_static(logger):
    tasks = [
        ("index_basic", lambda: IndexBasicDownloader().sync()),
        ("index_classify", lambda: IndexClassifyDownloader().sync()),
        ("sw_member", lambda: SWMemberDownloader().sync()),
        ("ci_member", lambda: CIMemberDownloader().sync()),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_index_daily(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["index_daily"]
    end_date = args.end_date or bj_today()
    downloader = IndexDailyDownloader()
    if args.ts_code:
        run_task(
            logger,
            f"index_daily_{args.ts_code.strip().upper()}",
            lambda: downloader.sync(
                args.ts_code.strip().upper(),
                start_date=start_date,
                target_end_date=end_date,
                list_date=args.list_date,
            ),
        )
    else:
        run_task(
            logger,
            "index_daily",
            lambda: downloader.sync_many(
                start_date=start_date,
                target_end_date=end_date,
            ),
        )


def update_index_weight(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["index_weight"]
    end_date = args.end_date
    downloader = IndexWeightDownloader()
    if args.ts_code:
        run_task(
            logger,
            f"index_weight_{args.ts_code.strip().upper()}",
            lambda: downloader.sync(
                args.ts_code.strip().upper(),
                start_date=start_date,
                target_end_date=end_date,
            ),
        )
    else:
        for spec in IndexDailyDownloader.DEFAULT_BROAD_BASE_INDEXES:
            code = spec["ts_code"]
            list_date = spec.get("list_date", start_date)
            run_task(
                logger,
                f"index_weight_{code}",
                lambda code=code, list_date=list_date: downloader.sync(
                    code,
                    start_date=max(start_date, list_date),
                    target_end_date=end_date,
                ),
            )


def update_index_industry_daily(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["index_industry_daily"]
    end_date = args.end_date or bj_today()
    tasks = [
        ("sw_daily", lambda: SWDailyDownloader().sync(start_date, end_date)),
        ("ci_daily", lambda: CIDailyDownloader().sync(start_date, end_date)),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_etf_static(logger):
    tasks = [
        ("etf_basic", lambda: ETFBasicDownloader().sync()),
        ("etf_index", lambda: ETFIndexDownloader().sync()),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_option_static(logger):
    run_task(logger, "option_basic", lambda: OptionBasicDownloader().sync())


def update_hk_static(logger):
    tasks = [
        ("hk_basic", lambda: HKBasicDownloader().sync()),
        (
            "hk_calendar",
            lambda: HKCalendarDownloader().sync(
                start_year=DEFAULT_START_DATES["hk_calendar_year"]
            ),
        ),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_us_static(logger):
    tasks = [
        ("us_basic", lambda: USBasicDownloader().sync()),
        (
            "us_calendar",
            lambda: USCalendarDownloader().sync(
                start_year=DEFAULT_START_DATES["us_calendar_year"]
            ),
        ),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_static_all(logger):
    update_calendar(argparse.Namespace(start_date=None, calendar_end_date=None), logger)
    update_stock_static(logger)
    update_future_static(logger)
    update_etf_static(logger)
    update_option_static(logger)
    update_index_static(logger)
    update_hk_static(logger)
    update_us_static(logger)


def update_stock_daily(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock"]
    end_date = args.end_date or bj_today()
    tasks = [
        ("stock_daily_pv", lambda: StockDailyPVDownloader().sync(start_date, end_date)),
        ("stock_adj_factor", lambda: StockAdjFactorDownloader().sync(start_date, end_date)),
        ("stock_daily_limit", lambda: StockDailyLimitDownloader().sync(start_date, end_date)),
        ("stock_daily_basic", lambda: StockDailyBasicDownloader().sync(start_date, end_date)),
        ("stock_suspend", lambda: StockSuspendDownloader().sync(start_date, end_date)),
        ("stock_moneyflow", lambda: StockMoneyflowDownloader().sync(start_date, end_date)),
        ("stock_st", lambda: StDownloader().sync(args.start_date or DEFAULT_START_DATES["stock_st"], end_date)),
        ("stock_trade_filter", lambda: StockTradeFilterBuilder().sync(start_date, end_date)),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_stock_trade_filter(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock"]
    end_date = args.end_date or bj_today()
    run_task(
        logger,
        "stock_trade_filter",
        lambda: StockTradeFilterBuilder().sync(start_date, end_date),
    )


def update_stock_minute(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock"]
    end_date = args.end_date or bj_today()
    run_task(
        logger,
        "stock_minute",
        lambda: StockMinuteDownloader().sync(start_date=start_date, target_end_date=end_date),
    )


def update_stock_derived_bar(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock"]
    end_date = args.end_date or bj_today()
    sizes = [
        int(value)
        for value in args.derived_bar_sizes.split(",")
        if value.strip()
    ]
    manifest = os.path.join(project_root, "factor_engine", "Cargo.toml")
    for bar_size in sizes:
        run_task(
            logger,
            f"stock_derived_bar_{bar_size}m",
            lambda bar_size=bar_size: subprocess.run(
                [
                    "cargo",
                    "run",
                    "--release",
                    "--manifest-path",
                    manifest,
                    "--",
                    "derive-bar",
                    "--asset",
                    "stock",
                    "--source",
                    "minute",
                    "--bar-size",
                    str(bar_size),
                    "--start-date",
                    start_date,
                    "--end-date",
                    end_date,
                ],
                cwd=project_root,
                check=True,
            ),
        )


def update_stock_financial(args, logger):
    start_date = args.start_date or bj_today()
    end_date = args.end_date or bj_today()

    financial_downloaders = [
        IncomeDownloader,
        BalanceSheetDownloader,
        CashFlowDownloader,
    ]
    for date in iter_calendar_dates(start_date, end_date):
        if not is_financial_statement_period(date):
            logger.info(f"skip stock_financial {date}: not a financial statement period")
            continue
        for downloader_cls in financial_downloaders:
            run_task(
                logger,
                f"{downloader_cls.__name__}_incremental_{date}",
                lambda cls=downloader_cls, d=date: cls().sync(mode="incremental", target_date=d),
            )


def update_stock_dividend(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock"]
    end_date = args.end_date or bj_today()
    run_task(
        logger,
        "stock_dividend",
        lambda: DividendDownloader().sync(
            start_date=start_date,
            target_end_date=end_date,
            rebuild=args.rebuild,
        ),
    )


def update_stock_alt(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["stock_alt"]
    end_date = args.end_date or bj_today()
    run_task(
        logger,
        "analyst_report",
        lambda: AnalystReportDownloader().sync(start_date=start_date, target_end_date=end_date),
    )


def update_future_static(logger):
    run_task(logger, "future_static", lambda: FutureBasicDownloader().sync())


def update_future_daily(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["future"]
    end_date = args.end_date or bj_today()
    tasks = [
        ("future_daily", lambda: FutureDailyDownloader().sync(start_date, end_date)),
        ("future_limit", lambda: FutureLimitDownloader().sync(start_date, end_date)),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_future_minute(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["future_minute"]
    end_date = args.end_date or bj_today()
    run_task(
        logger,
        "future_minute",
        lambda: FutureMinuteDownloader().sync(start_date=start_date, target_end_date=end_date),
    )


def update_etf(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["etf"]
    end_date = args.end_date or bj_today()
    update_etf_static(logger)
    tasks = [
        ("etf_daily_pv", lambda: ETFDailyPVDownloader().sync(start_date, end_date)),
        ("etf_adj_factor", lambda: ETFAdjFactorDownloader().sync(start_date, end_date)),
        ("etf_share_size", lambda: ETFShareSizeDownloader().sync(start_date, end_date)),
        ("etf_minute", lambda: ETFMinuteDownloader().sync(start_date, end_date)),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


def update_option(args, logger):
    start_date = args.start_date or DEFAULT_START_DATES["option"]
    end_date = args.end_date or bj_today()
    update_option_static(logger)
    tasks = [
        ("option_daily", lambda: OptionDailyDownloader().sync(start_date, end_date)),
        ("option_minute", lambda: OptionMinuteDownloader().sync()),
    ]
    for name, fn in tasks:
        run_task(logger, name, fn)


GROUPS = {
    "calendar": update_calendar,
    "static": lambda args, logger: update_static_all(logger),
    "stock_static": lambda args, logger: update_stock_static(logger),
    "index_static": lambda args, logger: update_index_static(logger),
    "index_daily": update_index_daily,
    "index_weight": update_index_weight,
    "index_industry_daily": update_index_industry_daily,
    "etf_static": lambda args, logger: update_etf_static(logger),
    "option_static": lambda args, logger: update_option_static(logger),
    "hk_static": lambda args, logger: update_hk_static(logger),
    "us_static": lambda args, logger: update_us_static(logger),
    "stock_daily": update_stock_daily,
    "stock_trade_filter": update_stock_trade_filter,
    "stock_minute": update_stock_minute,
    "stock_derived_bar": update_stock_derived_bar,
    "stock_financial": update_stock_financial,
    "stock_dividend": update_stock_dividend,
    "stock_alt": update_stock_alt,
    "future_static": lambda args, logger: update_future_static(logger),
    "future_daily": update_future_daily,
    "future_minute": update_future_minute,
    "etf": update_etf,
    "option": update_option,
}

DEFAULT_GROUPS = [
    "calendar",
    "stock_static",
    "stock_daily",
    "stock_minute",
    "stock_derived_bar",
    "future_static",
    "future_daily",
    "future_minute",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Incrementally update local parquet data from Tushare."
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUPS.keys()) + ["all"],
        default=DEFAULT_GROUPS,
        help="Task groups to run. Defaults to the main A-share and futures pipeline.",
    )
    parser.add_argument(
        "--start-date",
        help="YYYYMMDD. If omitted, each group uses its own historical start and skips local dates already present.",
    )
    parser.add_argument(
        "--end-date",
        help="YYYYMMDD. Defaults to today's date in Beijing time, except index_weight defaults to the previous complete month end.",
    )
    parser.add_argument(
        "--calendar-end-date",
        help="YYYYMMDD. Defaults to the current Beijing year end, e.g. 20261231.",
    )
    parser.add_argument(
        "--ts-code",
        help="Optional single index ts_code for --groups index_daily/index_weight, e.g. 000016.SH. If omitted, downloads default broad-based indexes.",
    )
    parser.add_argument(
        "--list-date",
        help="Optional YYYYMMDD list date for --ts-code; effective start is max(start-date, list-date).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild supported groups from scratch for the requested date range, e.g. stock_dividend.",
    )
    parser.add_argument(
        "--derived-bar-sizes",
        default="5,15",
        help="Comma-separated stock derived minute bar sizes for --groups stock_derived_bar; default: 5,15.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = QuantLogger()
    groups = list(GROUPS.keys()) if "all" in args.groups else args.groups

    logger.info(f">>> 增量更新开始，任务组: {groups}")
    for group in groups:
        GROUPS[group](args, logger)
    logger.info(">>> 增量更新结束")


if __name__ == "__main__":
    main()
