import os
import sys
import argparse

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import BalanceSheetDownloader, CashFlowDownloader, IncomeDownloader, QuantLogger

QUARTER_SUFFIXES = ("0331", "0630", "0930", "1231")


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize A-share financial statement data.")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument(
        "--end-date",
        type=int,
        default=None,
        help="Optional PIT cutoff date, e.g. 20260424. Rows with f_ann_date/ann_date after this date are skipped.",
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


def disclosure_series(df: pd.DataFrame) -> pd.Series:
    if "f_ann_date" in df.columns:
        source = df["f_ann_date"]
    elif "ann_date" in df.columns:
        source = df["ann_date"]
    else:
        return pd.Series([pd.NA] * len(df), index=df.index)
    normalized = source.astype(str).str.replace("-", "", regex=False).str.strip()
    return pd.to_numeric(normalized, errors="coerce")


def sync_until(downloader, start_year: int, end_date: int):
    periods = periods_until(start_year, end_date)
    downloader.logger.info(
        f"=== historical {downloader.task_name}: {periods[0]} -> {periods[-1]}, disclosure <= {end_date} ==="
    )
    for period in periods:
        df_period = downloader._fetch_period(period)
        if df_period is not None and not df_period.empty:
            df_period = df_period[disclosure_series(df_period) <= end_date].copy()
        downloader._process_and_save(df_period)
    downloader.logger.info(f"=== historical {downloader.task_name} complete ===")


def main():
    args = parse_args()
    logger = QuantLogger()
    logger.info(">>> init A-share financial statements: income, balance sheet, cashflow <<<")

    downloaders = [IncomeDownloader(), BalanceSheetDownloader(), CashFlowDownloader()]
    for downloader in downloaders:
        if args.end_date is None:
            downloader.sync(mode="historical", start_year=args.start_year, target_date=None)
        else:
            sync_until(downloader, args.start_year, args.end_date)

    logger.info(">>> init A-share financial statements complete <<<")


if __name__ == "__main__":
    main()
