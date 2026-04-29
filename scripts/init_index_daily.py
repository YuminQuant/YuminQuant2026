import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import IndexDailyDownloader


def bj_today():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize broad-based index daily parquet data.")
    parser.add_argument("--start-date", default="20090101", help="YYYYMMDD, default 20090101.")
    parser.add_argument("--end-date", default=bj_today(), help="YYYYMMDD, default Beijing today.")
    parser.add_argument(
        "--ts-code",
        help="Optional single index ts_code, e.g. 000016.SH. If omitted, downloads the default broad-based indexes.",
    )
    parser.add_argument(
        "--list-date",
        help="Optional YYYYMMDD list date for --ts-code; effective start is max(start-date, list-date).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    downloader = IndexDailyDownloader()
    if args.ts_code:
        downloader.sync(
            args.ts_code.strip().upper(),
            start_date=args.start_date,
            target_end_date=args.end_date,
            list_date=args.list_date,
        )
    else:
        downloader.sync_many(start_date=args.start_date, target_end_date=args.end_date)


if __name__ == "__main__":
    main()
