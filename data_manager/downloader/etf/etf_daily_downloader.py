import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class BaseETFDailyDownloader(BaseDownloader):
    """ETF 日频数据下载基类"""
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
                except Exception:
                    pass
        return local_dates

    def sync(self, start_date='20090101', target_end_date=None):
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
            self.logger.info(f"-> 处理 {year} 年缺失数据 ({len(dates)} 天)...")
            yearly_new = []
            for date in dates:
                try:
                    offset = 0
                    while True:
                        df_chunk = self._fetch_api(date, offset)
                        if df_chunk is None or df_chunk.empty:
                            break
                        yearly_new.append(df_chunk)
                        if len(df_chunk) < self.page_limit:
                            break
                        offset += self.page_limit
                        self.safe_sleep()
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 失败: {e}")
            
            if yearly_new:
                self._save_yearly_data(year, yearly_new)

    def _save_yearly_data(self, year, new_data_list):
        file_path = os.path.join(self.save_dir, f"{year}.parquet")
        df_new = pd.concat(new_data_list, ignore_index=True)
        
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        else:
            df_combined = df_new

        # 核心优化：日期转 int32
        if 'trade_date' in df_combined.columns:
            df_combined['trade_date'] = df_combined['trade_date'].astype(np.int32)
            
        # 委托子类进行特定的数值类型强转
        df_combined = self._optimize_dtypes(df_combined)

        df_combined.sort_values(by=['trade_date', 'ts_code'], inplace=True)
        df_combined.to_parquet(file_path, index=False)
        self.logger.info(f"成功保存 {year}.parquet。")

# ============ 派生子类 ============

class ETFDailyPVDownloader(BaseETFDailyDownloader):
    """ETF 日线量价"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_daily', 500))
        self.page_limit = config['api']['page_limits'].get('etf_daily', 2000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_daily_dir')
        self.task_name = "ETF 日线行情"

    def _fetch_api(self, date, offset):
        return self.pro.fund_daily(trade_date=date, limit=self.page_limit, offset=offset)

    def _optimize_dtypes(self, df):
        cols = ['open', 'high', 'low', 'close']
        for c in cols:
            if c in df.columns: df[c] = df[c].astype(np.float32)
        return df

class ETFAdjFactorDownloader(BaseETFDailyDownloader):
    """ETF 复权因子"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_adj', 500))
        self.page_limit = config['api']['page_limits'].get('etf_adj', 2000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_adj_dir')
        self.task_name = "ETF 复权因子"

    def _fetch_api(self, date, offset):
        return self.pro.fund_adj(trade_date=date, limit=self.page_limit, offset=offset)

    def _optimize_dtypes(self, df):
        if 'adj_factor' in df.columns: df['adj_factor'] = df['adj_factor'].astype(np.float32)
        return df

class ETFShareSizeDownloader(BaseETFDailyDownloader):
    """ETF 份额规模"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_share_size', 500))
        self.page_limit = config['api']['page_limits'].get('etf_share_size', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_share_size_dir')
        self.task_name = "ETF 份额规模"

    def _fetch_api(self, date, offset):
        return self.pro.etf_share_size(trade_date=date, limit=self.page_limit, offset=offset)

    def _optimize_dtypes(self, df):
        return df # 份额通常较大，保持 float64