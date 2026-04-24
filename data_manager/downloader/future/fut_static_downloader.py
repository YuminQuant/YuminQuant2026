import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class FutureBasicDownloader(BaseDownloader):
    """期货基础信息下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('fut_basic', 500))
        self.page_limit = config['api']['page_limits'].get('fut_basic', 8000)
        self.save_dir = self.get_full_path_and_ensure_dir('fut_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [期货全市场基础信息] ===")
        # CFFEX-中金所 DCE-大商所 CZCE-郑商所 SHFE-上期所 INE-能源中心 GFEX-广期所
        exchanges = ['CFFEX', 'DCE', 'CZCE', 'SHFE', 'INE', 'GFEX']
        all_data = []
        
        for exc in exchanges:
            offset = 0
            while True:
                try:
                    df = self.pro.fut_basic(exchange=exc,
                                            limit=self.page_limit,
                                            offset=offset,
                                            fields='ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate,trade_time_desc')
                    if df is None or df.empty:
                        break
                    df = df.dropna(axis=1,how='all').dropna(subset=['list_date'],axis=0)
                    all_data.append(df)
                    if len(df) < self.page_limit:
                        break
                    offset += self.page_limit
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取交易所 {exc} 基础信息失败: {e}")
                    break
                    
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code'], inplace=True)
            
            file_path = os.path.join(self.save_dir, "fut_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存期货基础信息，共涵盖 {len(df_combined)} 个合约。")