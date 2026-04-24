import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_manager import StockMinuteDownloader, QuantLogger

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [股票1分钟数据 - 历史初始化] <<<")
    downloader = StockMinuteDownloader()
    
    # 强烈建议：不要设为 1990，设为你想研究的最近年份，比如 20210101
    downloader.sync(start_date='20090101') 

if __name__ == "__main__":
    main()