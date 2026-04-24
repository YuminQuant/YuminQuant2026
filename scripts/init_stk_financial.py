import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import (
    IncomeDownloader,
    BalanceSheetDownloader,
    CashFlowDownloader,
    ForecastDownloader,
    ExpressDownloader,
    DividendDownloader,
    QuantLogger
)

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [股票财务指标 - 历史全量初始化] <<<")
    
    downloader = IncomeDownloader()
    downloader.sync(mode='historical', start_year=2009, target_date=None)

    downloader = BalanceSheetDownloader()
    downloader.sync(mode='historical', start_year=2009, target_date=None)

    downloader = CashFlowDownloader()
    downloader.sync(mode='historical', start_year=2009, target_date=None)
    
    downloader = ForecastDownloader()
    downloader.sync(mode='historical', start_year=2009, target_date=None)

    downloader = ExpressDownloader()
    downloader.sync(mode='historical', start_year=2009, target_date=None)

    downloader = DividendDownloader()
    downloader.sync(start_date='20090101', target_end_date=None)
    logger.info(">>> [股票财务指标 - 历史全量初始化] 执行完毕 <<<")

if __name__ == "__main__":
    main()