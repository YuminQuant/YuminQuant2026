import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class HKBasicDownloader(BaseDownloader):
    """港股列表下载器 (包含 L上市 和 D退市)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('hk_basic', 200))
        self.save_dir = self.get_full_path_and_ensure_dir('hk_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [港股全市场基础列表] ===")
        all_data = []
        for status in ['L', 'D']:
            try:
                # 港股列表数据量不大，直接一把拉取
                df = self.pro.hk_basic(list_status=status)
                if df is not None and not df.empty:
                    all_data.append(df)
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取港股列表(状态:{status})失败: {e}")
                
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code'], inplace=True)
            file_path = os.path.join(self.save_dir, "hk_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存港股基础信息，共 {len(df_combined)} 只股票。")

class HKCalendarDownloader(BaseDownloader):
    """港交所交易日历下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('hk_tradecal', 200))
        self.save_dir = self.get_full_path_and_ensure_dir('hk_calendar_dir')

    def sync(self, start_year=2000):
        self.logger.info("=== 开始同步 [港交所交易日历] ===")
        current_year = datetime.now().year
        # 按照 5 年一个 Chunk 切分，确保不超过 2000 行
        years = list(range(start_year, current_year + 5, 5))
        
        all_chunks = []
        for i in range(len(years)-1):
            s_date = f"{years[i]}0101"
            e_date = f"{years[i+1]-1}1231"
            try:
                df = self.pro.hk_tradecal(start_date=s_date, end_date=e_date)
                if df is not None and not df.empty:
                    all_chunks.append(df)
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取港股日历 {s_date}-{e_date} 失败: {e}")
                
        if all_chunks:
            df_cal = pd.concat(all_chunks, ignore_index=True)
            df_cal.drop_duplicates(subset=['cal_date'], inplace=True)
            df_cal['cal_date'] = df_cal['cal_date'].astype(int) # 转整型优化查询
            df_cal.sort_values(by='cal_date', inplace=True)
            
            file_path = os.path.join(self.save_dir, "trade_cal_HKEX.parquet")
            df_cal.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存港交所日历，共 {len(df_cal)} 个日历日。")