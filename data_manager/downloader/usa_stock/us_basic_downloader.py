import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class USBasicDownloader(BaseDownloader):
    """美股列表下载器 (含分页机制)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('us_basic', 200))
        self.page_limit = config['api']['page_limits'].get('us_basic', 6000)
        self.save_dir = self.get_full_path_and_ensure_dir('us_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [美股全市场基础列表] ===")
        all_data = []
        offset = 0
        
        while True:
            try:
                df = self.pro.us_basic(limit=self.page_limit, offset=offset)
                if df is None or df.empty:
                    break
                    
                all_data.append(df)
                self.logger.info(f"   已拉取 {offset + len(df)} 只美股基础信息...")
                
                if len(df) < self.page_limit:
                    break
                    
                offset += self.page_limit
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取美股列表失败 (offset={offset}): {e}")
                break
                
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code'], inplace=True)
            file_path = os.path.join(self.save_dir, "us_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存美股基础信息，历史至今共包含 {len(df_combined)} 只股票。")


class USCalendarDownloader(BaseDownloader):
    """美股交易日历下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('us_tradecal', 200))
        self.page_limit = config['api']['page_limits'].get('us_tradecal', 6000)
        self.save_dir = self.get_full_path_and_ensure_dir('us_calendar_dir')

    def sync(self, start_year=1980):
        self.logger.info("=== 开始同步 [美股交易日历] ===")
        # 使用美东时间(UTC-5)获取当前年份
        current_year = datetime.now(timezone(timedelta(hours=-5))).year
        
        # 6000天大约是16年，我们按10年一个Chunk切分最为稳妥
        years = list(range(start_year, current_year + 10, 10))
        
        all_chunks = []
        for i in range(len(years)-1):
            s_date = f"{years[i]}0101"
            e_date = f"{years[i+1]-1}1231"
            
            try:
                # 日历接口通常比较稳定，如果超过限制也需要 offset，这里直接按10年切分，确保远小于6000天
                df = self.pro.us_tradecal(start_date=s_date, end_date=e_date, limit=self.page_limit)
                if df is not None and not df.empty:
                    all_chunks.append(df)
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取美股日历 {s_date}-{e_date} 失败: {e}")
                
        if all_chunks:
            df_cal = pd.concat(all_chunks, ignore_index=True)
            df_cal.drop_duplicates(subset=['cal_date'], inplace=True)
            df_cal['cal_date'] = df_cal['cal_date'].astype(int)
            df_cal.sort_values(by='cal_date', inplace=True)
            
            file_path = os.path.join(self.save_dir, "trade_cal_US.parquet")
            df_cal.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存美股日历，共 {len(df_cal)} 个日历日。")