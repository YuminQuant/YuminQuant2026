# data_manager/downloaders/stock_daily_limit_downloader.py

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class StockDailyLimitDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        
        # 读取频控与分页配置
        rate_limit = config['api']['rate_limits'].get('stock_daily_limit', 400)
        super().__init__(rate_limit=rate_limit)
        self.page_limit = config.get('api', {}).get('page_limits', {}).get('stock_daily_limit', 5800)
        
        # 路径初始化
        self.save_dir = self.get_full_path_and_ensure_dir('stock_daily_limit_dir')
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步脚本！")
        
        df_cal = pd.read_parquet(self.cal_file)
        
        # 将传入的字符串日期转为 int32 进行条件筛选
        start_int = int(start_date)
        end_int = int(end_date)
        
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= start_int) & (df_cal['cal_date'] <= end_int)
        
        # 筛选出来后，强制转回字符串列表，供后续 Tushare 接口请求和 set 集合去重使用
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(self.save_dir, file)
                try:
                    df = pd.read_parquet(file_path, columns=['trade_date'])
                    # 由于我们存盘时把 trade_date 转成了 int32，这里读取后需要转回 str 进行比对
                    dates_str = df['trade_date'].astype(str).unique().tolist()
                    local_dates.update(dates_str)
                except Exception as e:
                    self.logger.warning(f"读取本地文件 {file} 失败: {e}")
        return local_dates

    def sync(self, start_date='19901219', target_end_date=None):
        if target_end_date is None:
            bj_tz = timezone(timedelta(hours=8))
            target_end_date = datetime.now(bj_tz).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [股票涨跌停价格] (区间: {start_date} -> {target_end_date}) ===")

        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地涨跌停数据已完全覆盖目标区间，无需更新。")
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
                        # 请求涨跌停数据
                        df_chunk = self.pro.stk_limit(
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
                    self.logger.error(f"拉取 {date} 涨跌停数据失败: {e}")
            
            if yearly_new_data:
                self._save_yearly_data(year, yearly_new_data)
                
        self.logger.info("=== [股票涨跌停价格] 同步结束 ===")

    def _save_yearly_data(self, year, new_data_list):
        file_path = os.path.join(self.save_dir, f"{year}.parquet")
        df_new = pd.concat(new_data_list, ignore_index=True)
        
        # 确保只要返回的这四列，避免接口未来新增字段干扰
        cols_needed = ['trade_date', 'ts_code', 'up_limit', 'down_limit']
        df_new = df_new[[c for c in cols_needed if c in df_new.columns]]
        
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        else:
            df_combined = df_new
            
        # ==========================================
        # 核心优化：数据类型强转
        # ==========================================
        if 'trade_date' in df_combined.columns:
            df_combined['trade_date'] = df_combined['trade_date'].astype(np.int32)
        if 'up_limit' in df_combined.columns:
            df_combined['up_limit'] = df_combined['up_limit'].astype(np.float32)
        if 'down_limit' in df_combined.columns:
            df_combined['down_limit'] = df_combined['down_limit'].astype(np.float32)

        # 排序并落盘 (排序时用 int 类型的 trade_date 速度极快)
        df_combined.sort_values(by=['trade_date', 'ts_code'], inplace=True)
        df_combined.to_parquet(file_path, index=False)
        self.logger.info(f"成功保存 {year}.parquet，当前包含 {len(df_combined)} 条记录。")