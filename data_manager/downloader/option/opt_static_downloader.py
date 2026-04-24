import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class OptionBasicDownloader(BaseDownloader):
    """期权基础信息下载器 (按交易所极速全量提取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('opt_basic', 500))
        self.page_limit = config['api']['page_limits'].get('opt_basic', 10000)
        self.save_dir = self.get_full_path_and_ensure_dir('opt_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [期权全市场基础信息] ===")
        # 涵盖国内所有存在期权的交易所
        exchanges = ['SSE', 'SZSE', 'CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', 'GFEX']
        all_data = []
        
        for exc in exchanges:
            offset = 0
            while True:
                try:
                    df = self.pro.opt_basic(exchange=exc, limit=self.page_limit, offset=offset)
                    if df is None or df.empty:
                        break
                    all_data.append(df)
                    if len(df) < self.page_limit:
                        break
                    offset += self.page_limit
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取交易所 {exc} 期权基础信息失败: {e}")
                    break
                    
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code'], inplace=True)
            
            file_path = os.path.join(self.save_dir, "opt_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存期权基础信息，共涵盖 {len(df_combined)} 个历史与活跃期权合约。")