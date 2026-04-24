import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class ETFMinuteDownloader(BaseDownloader):
    """ETF 分钟线下载器 (多线程并发 Batch 架构)"""
    def __init__(self):
        config = ConfigManager().config
        # 统一使用 450次/分钟 的安全频率限制
        super().__init__(rate_limit=config['api']['rate_limits'].get('etf_minute', 400))
        self.page_limit = config['api']['page_limits'].get('etf_minute', 8000)
        self.save_dir = self.get_full_path_and_ensure_dir('etf_minute_dir')
        self.fund_daily_dir = self.get_full_path_and_ensure_dir('fund_daily_dir')
        
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

        # 多线程限流配置
        self.max_workers = 10
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 450.0 

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError("未找到日历文件！")
        df_cal = pd.read_parquet(self.cal_file)
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= int(start_date)) & (df_cal['cal_date'] <= int(end_date))
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for year_folder in os.listdir(self.save_dir):
            year_path = os.path.join(self.save_dir, year_folder)
            if os.path.isdir(year_path):
                for file in os.listdir(year_path):
                    if file.endswith('.parquet'):
                        local_dates.add(file.replace('.parquet', ''))
        return local_dates

    def _get_valid_etfs_for_day(self, date_str):
        """从日线提取当天的有效 ETF 列表"""
        year = date_str[:4]
        pv_file = os.path.join(self.fund_daily_dir, f"{year}.parquet")
        if not os.path.exists(pv_file):
            return []
        try:
            df_pv = pd.read_parquet(pv_file, columns=['trade_date', 'ts_code'])
            df_day = df_pv[df_pv['trade_date'] == int(date_str)]
            return df_day['ts_code'].tolist()
        except Exception as e:
            self.logger.error(f"读取 {date_str} ETF日线失败: {e}")
            return []

    def _fetch_batch_safe(self, ts_code_str, start_time, end_time, retry=3):
        """线程安全的 API 拉取"""
        with self.api_lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()
            
        for attempt in range(retry):
            try:
                # ETF 分钟线接口与股票一致
                df = self.pro.stk_mins(
                    ts_code=ts_code_str, 
                    freq='1min', 
                    start_date=start_time, 
                    end_date=end_time
                )
                return df
            except Exception as e:
                if attempt == retry - 1: return e
                time.sleep(1)

    def sync(self, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [ETF 分钟线] (并发Batch架构) ({start_date} -> {target_end_date}) ===")
        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地数据已完全覆盖目标区间。")
            return

        total_days = len(missing_dates)
        for idx, date in enumerate(missing_dates):
            self._fetch_and_save_single_day(date, current_idx=idx+1, total_days=total_days)
            
        self.logger.info("=== [ETF 分钟线] 同步完毕 ===")

    def _fetch_and_save_single_day(self, date, current_idx, total_days):
        year = date[:4]
        valid_codes = self._get_valid_etfs_for_day(date)
        if not valid_codes:
            self.logger.warning(f"[{date}] 未找到有效 ETF，跳过。")
            return
            
        batch_size = 30
        code_batches = [valid_codes[i:i + batch_size] for i in range(0, len(valid_codes), batch_size)]
        
        fmt_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        start_time, end_time = f"{fmt_date} 09:00:00", f"{fmt_date} 16:00:00"
        
        day_chunks = []
        total_batches = len(code_batches)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._fetch_batch_safe, ",".join(batch), start_time, end_time): i 
                for i, batch in enumerate(code_batches)
            }
            
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        self.logger.error(f"[{date}] Batch 失败: {result}")
                    elif result is not None and not result.empty:
                        day_chunks.append(result)
                except Exception as exc:
                    self.logger.error(f"[{date}] 线程异常: {exc}")

        if day_chunks:
            df_day = pd.concat(day_chunks, ignore_index=True)
            df_day['trade_date'] = int(date)
            
            # 转换类型
            for col in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                if col in df_day.columns:
                    df_day[col] = df_day[col].astype(np.float32)
                    
            if 'trade_time' in df_day.columns:
                df_day.sort_values(by=['ts_code', 'trade_time'], inplace=True)
                
            year_dir = os.path.join(self.save_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            df_day.to_parquet(os.path.join(year_dir, f"{date}.parquet"), index=False)
            
            remaining = total_days - current_idx
            self.logger.info(f"✅ [{date}] ETF落盘 (含 {len(df_day)} 行) [大盘进度: {current_idx}/{total_days} | 剩余: {remaining} 天]")