import sys
import os

# 将项目根目录加入系统路径，确保能跨目录导入 data_manager
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager import (
    FutureBasicDownloader,
    QuantLogger
)

def main():
    logger = QuantLogger()
    logger.info(">>> 开始执行 [历史数据更新脚本] <<<")

    downloader = FutureBasicDownloader()
    downloader.sync() # 从接口支持的最早日期开始
    
    logger.info(">>> [历史数据更新脚本] 执行完毕 <<<")

if __name__ == "__main__":
    main()