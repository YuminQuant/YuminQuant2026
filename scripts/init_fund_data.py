"""Initialize fund sources; importing this script never downloads data."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_manager import (
    FundBasicDownloader, FundPortfolioDownloader, FundCompanyDownloader,
    FundManagerDownloader, FundBenchmarkDownloader, FundShareDownloader,
    FundNavDownloader, FundDividendDownloader, FundFactorProDownloader,
)

DOWNLOADERS = {
    "basic": FundBasicDownloader, "portfolio": FundPortfolioDownloader,
    "company": FundCompanyDownloader, "manager": FundManagerDownloader,
    "benchmark": FundBenchmarkDownloader, "share": FundShareDownloader,
    "nav": FundNavDownloader, "dividend": FundDividendDownloader,
    "factor_pro": FundFactorProDownloader,
}


def main():
    parser = argparse.ArgumentParser(description="Initialize public-fund sources; static sources are current snapshots.")
    parser.add_argument("--datasets", nargs="+", choices=list(DOWNLOADERS), default=["basic", "portfolio"])
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-date", help="YYYYMMDD history cutoff; ignored for basic/company/benchmark snapshots.")
    args = parser.parse_args()
    for name in dict.fromkeys(args.datasets):
        downloader = DOWNLOADERS[name]()
        if name in {"basic", "company", "benchmark"}:
            downloader.sync()
        else:
            downloader.sync(start_year=args.start_year, end_date=args.end_date)


if __name__ == "__main__":
    main()
