import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import MainBusinessDownloader, QuantLogger

QUARTER_SUFFIXES = ("0331", "0630", "0930", "1231")


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize A-share main business data.")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument(
        "--end-date",
        type=int,
        default=None,
        help="Optional period cutoff date, e.g. 20260424. Periods after this date are skipped.",
    )
    return parser.parse_args()


def periods_until(start_year: int, end_date: int) -> list[str]:
    end_year = end_date // 10000
    periods = []
    for year in range(start_year, end_year + 1):
        for suffix in QUARTER_SUFFIXES:
            period = f"{year}{suffix}"
            if int(period) <= end_date:
                periods.append(period)
    return periods


def sync_until(downloader: MainBusinessDownloader, start_year: int, end_date: int):
    periods = periods_until(start_year, end_date)
    if not periods:
        downloader.logger.info("no main business periods to initialize")
        return
    downloader.logger.info(
        f"=== historical {downloader.task_name}: {periods[0]} -> {periods[-1]} ==="
    )
    for period in periods:
        df_period = downloader._fetch_period(period)
        downloader._process_and_save(df_period)
    downloader.logger.info(f"=== historical {downloader.task_name} complete ===")


def main():
    args = parse_args()
    logger = QuantLogger()
    logger.info(">>> init A-share main business data <<<")

    downloader = MainBusinessDownloader()
    if args.end_date is None:
        downloader.sync(mode="historical", start_year=args.start_year, target_date=None)
    else:
        sync_until(downloader, args.start_year, args.end_date)

    logger.info(">>> init A-share main business data complete <<<")


if __name__ == "__main__":
    main()
