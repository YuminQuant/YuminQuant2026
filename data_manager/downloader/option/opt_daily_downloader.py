import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class OptionDailyDownloader(BaseDownloader):
    """期权日线行情下载器 (按天全市场横截面提取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('opt_daily', 500))
        self.page_limit = config['api']['page_limits'].get('opt_daily', 15000)
        self.save_dir = self.get_full_path_and_ensure_dir('opt_daily_dir')

    def _get_trade_dates(self, start_date, end_date):
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')
        df_cal = pd.read_parquet(cal_file)
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= int(start_date)) & (df_cal['cal_date'] <= int(end_date))
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(os.path.join(self.save_dir, file), columns=['trade_date'])
                    local_dates.update(df['trade_date'].astype(str).unique().tolist())
                except: pass
        return local_dates

    def sync(self, start_date='20150209', target_end_date=None): # 上证50ETF期权2015年上市
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [期权日线行情] ({start_date} -> {target_end_date}) ===")
        missing_dates = sorted(list(set(self._get_trade_dates(start_date, target_end_date)) - self._get_local_dates()))
        
        if not missing_dates:
            self.logger.info("本地数据已覆盖目标区间。")
            return

        dates_by_year = {}
        for d in missing_dates:
            dates_by_year.setdefault(d[:4], []).append(d)

        for year, dates in dates_by_year.items():
            self.logger.info(f"-> 处理 {year} 年期权日线...")
            yearly_new = []
            for date in dates:
                try:
                    offset = 0
                    while True:
                        df = self.pro.opt_daily(trade_date=date, limit=self.page_limit, offset=offset)
                        if df is None or df.empty:
                            break
                        yearly_new.append(df)
                        if len(df) < self.page_limit:
                            break
                        offset += self.page_limit
                        self.safe_sleep()
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 日线失败: {e}")
            
            if yearly_new:
                file_path = os.path.join(self.save_dir, f"{year}.parquet")
                df_new = pd.concat(yearly_new, ignore_index=True)
                
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                    df_new.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                
                if 'trade_date' in df_new.columns: df_new['trade_date'] = df_new['trade_date'].astype(np.int32)
                for c in ['open', 'high', 'low', 'close', 'settle', 'vol', 'amount', 'oi']:
                    if c in df_new.columns: df_new[c] = df_new[c].astype(np.float32)
                    
                df_new.sort_values(by=['trade_date', 'ts_code'], inplace=True)
                df_new.to_parquet(file_path, index=False)
                self.logger.info(f"✅ 成功保存 {year}.parquet")