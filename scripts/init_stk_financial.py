import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import BalanceSheetDownloader, CashFlowDownloader, IncomeDownloader, QuantLogger


def main():
    logger = QuantLogger()
    logger.info(">>> init A-share financial statements: income, balance sheet, cashflow <<<")

    IncomeDownloader().sync(mode="historical", start_year=2009, target_date=None)
    BalanceSheetDownloader().sync(mode="historical", start_year=2009, target_date=None)
    CashFlowDownloader().sync(mode="historical", start_year=2009, target_date=None)

    logger.info(">>> init A-share financial statements complete <<<")


if __name__ == "__main__":
    main()
