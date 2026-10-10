"""Initialize fund sources; importing this script never downloads data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager import FundBasicDownloader, FundPortfolioDownloader


def main():
    parser = argparse.ArgumentParser(description="Initialize fund basic and disclosed portfolio data.")
    parser.add_argument("--datasets", nargs="+", choices=["basic", "portfolio"], default=["basic", "portfolio"])
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-date", help="Report-period and announcement cutoff YYYYMMDD (portfolio only).")
    args = parser.parse_args()
    if "basic" in args.datasets:
        FundBasicDownloader().sync()
    if "portfolio" in args.datasets:
        FundPortfolioDownloader().sync(start_year=args.start_year, end_date=args.end_date)


if __name__ == "__main__":
    main()
