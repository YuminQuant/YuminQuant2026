import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class ETFBasicDownloader(BaseDownloader):
    """ETF 基础信息下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_basic', 500))
        self.page_limit = config['api']['page_limits'].get('etf_basic', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [ETF 基础信息] ===")
        all_data = []
        offset = 0
        while True:
            try:
                df = self.pro.etf_basic(limit=self.page_limit, offset=offset)
                if df is None or df.empty:
                    break
                all_data.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取 ETF 基础信息失败: {e}")
                break
        
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            file_path = os.path.join(self.save_dir, "etf_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"成功保存 ETF 基础信息，共 {len(df_combined)} 条。")

class ETFIndexDownloader(BaseDownloader):
    """ETF 基准指数列表下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_index', 500))
        self.page_limit = config['api']['page_limits'].get('etf_index', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_index_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [ETF 基准指数列表] ===")
        all_data = []
        offset = 0
        while True:
            try:
                df = self.pro.etf_index(limit=self.page_limit, offset=offset)
                if df is None or df.empty:
                    break
                all_data.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取 ETF 基准指数失败: {e}")
                break
        
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            file_path = os.path.join(self.save_dir, "etf_index.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"成功保存 ETF 基准指数列表，共 {len(df_combined)} 条。")