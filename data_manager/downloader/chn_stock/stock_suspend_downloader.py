# data_manager/downloaders/stock_suspend_downloader.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class StockSuspendDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        
        # 读取配置
        rate_limit = config['api']['rate_limits'].get('stock_suspend', 400)
        super().__init__(rate_limit=rate_limit)
        self.page_limit = config.get('api', {}).get('page_limits', {}).get('stock_suspend', 6000)
        
        # 路径初始化
        self.save_dir = self.get_full_path_and_ensure_dir('stock_suspend_dir')
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步脚本！")
        
        df_cal = pd.read_parquet(self.cal_file)
        
        # 将传入的字符串日期转为 int32 进行条件筛选 (适配底层类型统一)
        start_int = int(start_date)
        end_int = int(end_date)
        
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= start_int) & (df_cal['cal_date'] <= end_int)
        
        # 筛选出来后，强制转回字符串列表
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(self.save_dir, file)
                try:
                    df = pd.read_parquet(file_path, columns=['trade_date'])
                    dates_str = df['trade_date'].astype(str).unique().tolist()
                    local_dates.update(dates_str)
                except Exception as e:
                    self.logger.warning(f"读取本地文件 {file} 失败: {e}")
        return local_dates

    def sync(self, start_date='19901219', target_end_date=None):
        if target_end_date is None:
            bj_tz = timezone(timedelta(hours=8))
            target_end_date = datetime.now(bj_tz).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [股票停复牌数据] (区间: {start_date} -> {target_end_date}) ===")

        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地停复牌数据已完全覆盖目标区间，无需更新。")
            return
            
        self.logger.info(f"发现 {len(missing_dates)} 个交易日的数据缺失，开始抓取...")

        dates_by_year = {}
        for date in missing_dates:
            year = date[:4]
            if year not in dates_by_year:
                dates_by_year[year] = []
            dates_by_year[year].append(date)

        for year, dates in dates_by_year.items():
            self.logger.info(f"-> 正在处理 {year} 年缺失数据 (共 {len(dates)} 天)...")
            yearly_new_data = []
            
            for date in tqdm(dates, desc=f"{year}年进度", mininterval=10.0, ascii=True):
                try:
                    offset = 0
                    while True:
                        df_chunk = self.pro.suspend_d(
                            trade_date=date, 
                            limit=self.page_limit, 
                            offset=offset
                        )
                        
                        if df_chunk is None or df_chunk.empty:
                            break
                            
                        yearly_new_data.append(df_chunk)
                        
                        if len(df_chunk) < self.page_limit:
                            break
                            
                        offset += self.page_limit
                        self.safe_sleep()
                        
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 停复牌数据失败: {e}")
            
            if yearly_new_data:
                self._save_yearly_data(year, yearly_new_data)
                
        self.logger.info("=== [股票停复牌数据] 同步结束 ===")

    def _save_yearly_data(self, year, new_data_list):
        file_path = os.path.join(self.save_dir, f"{year}.parquet")
        df_new = pd.concat(new_data_list, ignore_index=True)
        
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        else:
            df_combined = df_new
            
        # ==========================================
        # 数据类型强转 (trade_date 统一为 int32)
        # ==========================================
        if 'trade_date' in df_combined.columns:
            df_combined['trade_date'] = df_combined['trade_date'].astype(np.int32)
                
        df_combined.sort_values(by=['trade_date', 'ts_code'], inplace=True)
        df_combined.to_parquet(file_path, index=False)
        self.logger.info(f"成功保存 {year}.parquet，当前包含 {len(df_combined)} 条记录。")