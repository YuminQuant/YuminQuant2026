import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import (
    AnalystReportDownloader,
    QuantLogger
)

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [股票另类数据 - 历史全量初始化] <<<")
    
    downloader = AnalystReportDownloader()
    downloader.sync(start_date='20100101', target_end_date=None)

    logger.info(">>> [股票另类数据 - 历史全量初始化] 执行完毕 <<<")

if __name__ == "__main__":
    main()