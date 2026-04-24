import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import StockAdjFactorDownloader, QuantLogger

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [股票日线PV数据 - 历史全量初始化] <<<")
    
    downloader = StockAdjFactorDownloader()
    
    # 设定A股成立早期的节点，程序会自动找出直到今天所有缺失的交易日
    START_DATE = '20090101' 
    
    downloader.sync(start_date=START_DATE)
    
    logger.info(">>> [股票日线PV数据 - 历史全量初始化] 执行完毕 <<<")

if __name__ == "__main__":
    main()