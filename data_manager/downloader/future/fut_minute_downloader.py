import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import daily_file_path

class FutureMinuteDownloader(BaseDownloader):
    """期货1分钟下载器 (本地日线联动 + 按交易日横截面切片，解决夜盘错位)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('fut_minute', 500))
        self.page_limit = config['api']['page_limits'].get('fut_minute', 8000)
        
        # 目录配置
        self.save_dir = self.get_full_path_and_ensure_dir('fut_minute_dir')
        
        # 羁绊回归：挂载本地的期货日线目录！
        self.daily_dir = self.get_full_path_and_ensure_dir('fut_daily_dir')
        
        # 本地日历路径
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'chn_stock_data/calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

        # 多线程与全局限流配置
        self.max_workers = 10
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 480.0

    def _get_trade_calendar(self, start_date, end_date):
        """获取交易日历，并计算每个交易日对应的上一个交易日 (pretrade_date)"""
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步！")
            
        df_cal = pd.read_parquet(self.cal_file)
        df_open = df_cal[df_cal['is_open'] == 1].copy()
        df_open.sort_values('cal_date', inplace=True)
        
        df_open['pretrade_date'] = df_open['cal_date'].shift(1)
        
        mask = (df_open['cal_date'] >= int(start_date)) & (df_open['cal_date'] <= int(end_date))
        df_target = df_open[mask].copy()
        
        cal_dict = {}
        for _, row in df_target.iterrows():
            trade_date = str(int(row['cal_date']))
            pre_date = str(int(row['pretrade_date'])) if pd.notna(row['pretrade_date']) else (pd.to_datetime(trade_date) - pd.Timedelta(days=1)).strftime('%Y%m%d')
            cal_dict[trade_date] = pre_date
            
        return cal_dict

    def _get_active_codes_from_local(self, trade_date):
        """羁绊大招：直接从本地存好的期货日线中提取存活合约，零网络开销！"""
        year = trade_date[:4]
        date_file = daily_file_path(self.daily_dir, trade_date)
        daily_file = date_file if os.path.exists(date_file) else os.path.join(self.daily_dir, f"{year}.parquet")
        
        if not os.path.exists(daily_file):
            self.logger.warning(f"本地缺少 {year} 年的期货日线数据 ({daily_file})，无法获取 {trade_date} 的存活合约。请先跑日线同步！")
            return []
            
        try:
            # 读取那一年的日线数据（如果内存够，Pandas读取很快。如果要极限性能，以后可以换 Polars）
            df_daily = pd.read_parquet(daily_file, columns=['ts_code', 'trade_date'])
            # 精准截取当天的全部存活代码
            active_codes = df_daily[df_daily['trade_date'] == int(trade_date)]['ts_code'].unique().tolist()
            return active_codes
        except Exception as e:
            self.logger.error(f"读取本地日线数据失败: {e}")
            return []

    def _fetch_batch_mins(self, ts_codes, start_time, end_time, retry=3):
        """批量拉取并合并这批合约的分钟线"""
        ts_code_str = ",".join(ts_codes)
        batch_chunks = []
        offset = 0
        
        while True:
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()

            success = False
            for attempt in range(retry):
                try:
                    df = self.pro.ft_mins(
                        ts_code=ts_code_str, freq='1min', 
                        start_date=start_time, end_date=end_time,
                        limit=self.page_limit, offset=offset
                    )
                    success = True
                    break
                except Exception as e:
                    if attempt == retry - 1:
                        self.logger.error(f"API拉取失败: {e}")
                    time.sleep(1)

            if not success or df is None or df.empty:
                break
                
            batch_chunks.append(df)
            if len(df) < self.page_limit:
                break
            offset += self.page_limit
            
        return pd.concat(batch_chunks, ignore_index=True) if batch_chunks else None

    def sync(self, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
            
        self.logger.info(f"=== 开始同步 [期货分钟线] ({start_date} -> {target_end_date}) ===")
        
        cal_dict = self._get_trade_calendar(start_date, target_end_date)
        if not cal_dict:
            self.logger.warning("未找到有效交易日！")
            return

        for trade_date, pre_date in cal_dict.items():
            year = trade_date[:4]
            year_dir = os.path.join(self.save_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            file_path = os.path.join(year_dir, f"{trade_date}.parquet")
            
            # 断点续传：本地已有则跳过
            if os.path.exists(file_path):
                continue
                
            # 1. 核心改变：从本地读取日线获取活跃合约！速度快如闪电！
            active_codes = self._get_active_codes_from_local(trade_date)
            if not active_codes:
                self.logger.info(f"[{trade_date}] 本地无存活合约数据，跳过。")
                continue
                
            # 2. 构造完美包围夜盘和日盘的物理时间戳
            start_time = f"{pre_date[:4]}-{pre_date[4:6]}-{pre_date[6:8]} 20:00:00"
            end_time = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]} 15:20:00"
            
            self.logger.info(f"-> 正在拉取 {trade_date} (存活合约数: {len(active_codes)}, 时间段: {start_time} 至 {end_time})")
            
            # 3. 分 Batch 多线程拉取
            batch_size = 5
            code_batches = [active_codes[i:i + batch_size] for i in range(0, len(active_codes), batch_size)]
            
            day_chunks = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._fetch_batch_mins, batch, start_time, end_time): batch 
                    for batch in code_batches
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    try:
                        df_res = future.result()
                        if df_res is not None and not df_res.empty:
                            day_chunks.append(df_res)
                    except Exception as exc:
                        self.logger.error(f"Batch 处理崩溃: {exc}")

            # 4. 数据清洗与落盘 (全市场当天聚合为一个文件)
            if day_chunks:
                df_daily = pd.concat(day_chunks, ignore_index=True)
                
                df_daily['trade_date'] = np.int32(trade_date)
                
                for c in ['open', 'high', 'low', 'close', 'vol', 'amount', 'oi']:
                    if c in df_daily.columns:
                        df_daily[c] = pd.to_numeric(df_daily[c], errors='coerce').astype(np.float32)
                
                df_daily.sort_values(by=['ts_code', 'trade_time'], inplace=True)
                df_daily.to_parquet(file_path, index=False)
                self.logger.info(f"✅ [{trade_date}] 落地 {len(df_daily)} 行 -> {file_path}")
                
        self.logger.info("=== [期货 1 分钟行情] 同步结束 ===")
