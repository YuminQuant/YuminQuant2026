import argparse
import os
import shutil
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import AnalystReportDownloader, ConfigManager, QuantLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize analyst report data.")
    parser.add_argument("--start-date", default="20100101", help="YYYYMMDD, default 20100101.")
    parser.add_argument("--end-date", help="YYYYMMDD. Defaults to Beijing today in downloader.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Backup and clear analyst_report before rebuilding from start-date.",
    )
    return parser.parse_args()


def analyst_report_dir():
    config = ConfigManager().config
    base_data_dir = config["paths"]["base_data_dir"]
    sub_dir = config["paths"].get("analyst_report_dir", "stock_data/analyst_report")
    return os.path.join(base_data_dir, sub_dir)


def backup_and_clear(path, logger):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}_backup_{timestamp}"
    shutil.copytree(path, backup_path)
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
    logger.info(f"analyst_report backup created: {backup_path}")
    return backup_path


def main():
    args = parse_args()
    logger = QuantLogger()
    if args.rebuild:
        backup_and_clear(analyst_report_dir(), logger)

    downloader = AnalystReportDownloader()
    downloader.sync(start_date=args.start_date, target_end_date=args.end_date)


if __name__ == "__main__":
    main()
