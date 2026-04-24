import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import StockMoneyflowDownloader, QuantLogger

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [股票每日估值指标 - 历史全量初始化] <<<")
    
    downloader = StockMoneyflowDownloader()
    downloader.sync(start_date='20090101')
    
    logger.info(">>> [股票每日估值指标 - 历史全量初始化] 执行完毕 <<<")

if __name__ == "__main__":
    main()