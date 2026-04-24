import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class BaseIndexDailyDownloader(BaseDownloader):
    """行业指数日线基类 (按天全市场)"""
    def _get_trade_dates(self, start_date, end_date):
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')
        df_cal = pd.read_parquet(cal_file)
        start_int, end_int = int(start_date), int(end_date)
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= start_int) & (df_cal['cal_date'] <= end_int)
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(os.path.join(self.save_dir, file), columns=['trade_date'])
                    local_dates.update(df['trade_date'].astype(str).unique().tolist())
                except:
                    pass
        return local_dates

    def sync(self, start_date='20000101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [{self.task_name}] ({start_date} -> {target_end_date}) ===")
        missing_dates = sorted(list(set(self._get_trade_dates(start_date, target_end_date)) - self._get_local_dates()))
        if not missing_dates:
            self.logger.info("本地数据已覆盖目标区间。")
            return

        dates_by_year = {}
        for d in missing_dates:
            dates_by_year.setdefault(d[:4], []).append(d)

        for year, dates in dates_by_year.items():
            yearly_new = []
            for date in dates:
                try:
                    df_chunk = self._fetch_api(date)
                    if df_chunk is not None and not df_chunk.empty:
                        yearly_new.append(df_chunk)
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 失败: {e}")
            
            if yearly_new:
                file_path = os.path.join(self.save_dir, f"{year}.parquet")
                df_new = pd.concat(yearly_new, ignore_index=True)
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                    df_new.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                
                if 'trade_date' in df_new.columns: df_new['trade_date'] = df_new['trade_date'].astype(np.int32)
                for c in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                    if c in df_new.columns: df_new[c] = df_new[c].astype(np.float32)
                    
                df_new.sort_values(by=['trade_date', 'ts_code'], inplace=True)
                df_new.to_parquet(file_path, index=False)
                self.logger.info(f"成功保存 {year}.parquet。")

class SWDailyDownloader(BaseIndexDailyDownloader):
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('sw_daily', 500))
        self.save_dir = self.get_full_path_and_ensure_dir('sw_daily_dir')
        self.task_name = "申万行业日线"
    def _fetch_api(self, date):
        return self.pro.sw_daily(trade_date=date)

class CIDailyDownloader(BaseIndexDailyDownloader):
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('ci_daily', 500))
        self.save_dir = self.get_full_path_and_ensure_dir('ci_daily_dir')
        self.task_name = "中信行业日线"
    def _fetch_api(self, date):
        return self.pro.ci_daily(trade_date=date)