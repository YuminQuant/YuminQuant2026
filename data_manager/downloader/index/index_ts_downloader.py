import os
import numpy as np
import pandas as pd
import time
import threading
import concurrent.futures
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class IndexDailyDownloader(BaseDownloader):
    """单一指数日线下载器 (按 ts_code 提取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_daily', 500))
        self.base_save_dir = self.get_full_path_and_ensure_dir('index_daily_dir')

    def sync(self, ts_code, start_date='19900101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
            
        self.logger.info(f"=== 开始同步 [特定指数日线: {ts_code}] ({start_date} -> {target_end_date}) ===")
        # 建立专属于该 ts_code 的目录
        code_dir = os.path.join(self.base_save_dir, ts_code.replace('.', '_'))
        os.makedirs(code_dir, exist_ok=True)
        
        # 对于单一标的，单次接口可拉取8000天（约32年），我们为了稳妥，按年切片拉取
        years = [str(y) for y in range(int(start_date[:4]), int(target_end_date[:4]) + 1)]
        
        for year in years:
            # 确定该年的边界
            y_start = max(start_date, f"{year}0101")
            y_end = min(target_end_date, f"{year}1231")
            file_path = os.path.join(code_dir, f"{year}.parquet")
            
            # 如果是过去完整的年份，且文件已存在，则跳过 (增量更新逻辑)
            if year < target_end_date[:4] and os.path.exists(file_path):
                continue
                
            try:
                df = self.pro.index_daily(ts_code=ts_code, start_date=y_start, end_date=y_end)
                self.safe_sleep()
                
                if df is not None and not df.empty:
                    # 如果是当年，且有本地数据，则去重合并
                    if os.path.exists(file_path):
                        df_old = pd.read_parquet(file_path)
                        df = pd.concat([df_old, df], ignore_index=True)
                        df.drop_duplicates(subset=['trade_date'], keep='last', inplace=True)

                    if 'trade_date' in df.columns: df['trade_date'] = df['trade_date'].astype(np.int32)
                    for c in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                        if c in df.columns: df[c] = df[c].astype(np.float32)
                    
                    df.sort_values(by='trade_date', inplace=True)
                    df.to_parquet(file_path, index=False)
                    self.logger.info(f"-> {ts_code} {year}年 落地完毕 ({len(df)}条)。")
            except Exception as e:
                self.logger.error(f"拉取 {ts_code} {year}年 数据失败: {e}")

class IndexWeightDownloader(BaseDownloader):
    """单一指数成分权重下载器 (按月度提取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_weight', 500))
        self.base_save_dir = self.get_full_path_and_ensure_dir('index_weight_dir')

    def sync(self, index_code, start_date='20100101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
            
        self.logger.info(f"=== 开始同步 [特定指数权重: {index_code}] ===")
        code_dir = os.path.join(self.base_save_dir, index_code.replace('.', '_'))
        os.makedirs(code_dir, exist_ok=True)
        
        # 生成月度的头尾日期
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(target_end_date)
        # 用 MS (月初) 生成区间
        months = pd.date_range(start_dt.replace(day=1), end_dt, freq='MS')
        
        # 依然按年归档
        yearly_data = {}
        for d in months:
            m_start = d.strftime('%Y%m%d')
            # 计算当月最后一天
            next_m = d.replace(day=28) + timedelta(days=4)
            m_end = (next_m - timedelta(days=next_m.day)).strftime('%Y%m%d')
            year = d.strftime('%Y')
            
            # 只在当月无数据或当前月更新时请求
            file_path = os.path.join(code_dir, f"{year}.parquet")
            if year < target_end_date[:4] and os.path.exists(file_path):
                # 如果是过去的整年文件已存在，则不用每个月再拉
                # 我们在此处略过精细校验，假设整年文件包含12个月。
                continue
            
            try:
                df = self.pro.index_weight(index_code=index_code, start_date=m_start, end_date=m_end)
                self.safe_sleep()
                if df is not None and not df.empty:
                    yearly_data.setdefault(year, []).append(df)
            except Exception as e:
                self.logger.error(f"拉取权重 {index_code} {m_start} 失败: {e}")
                
        # 归档落盘
        for year, dfs in yearly_data.items():
            file_path = os.path.join(code_dir, f"{year}.parquet")
            df_new = pd.concat(dfs, ignore_index=True)
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new.drop_duplicates(subset=['con_code', 'trade_date'], keep='last', inplace=True)
            
            if 'trade_date' in df_new.columns: df_new['trade_date'] = df_new['trade_date'].astype(np.int32)
            if 'weight' in df_new.columns: df_new['weight'] = df_new['weight'].astype(np.float32)
            
            df_new.sort_values(by=['trade_date', 'con_code'], inplace=True)
            df_new.to_parquet(file_path, index=False)
            self.logger.info(f"-> {index_code} {year}年成分权重 落地完毕。")

class IndexMinuteDownloader(BaseDownloader):
    """单一指数1分钟行情 (多线程年份并发极速版)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_minute', 500))
        self.page_limit = config['api']['page_limits'].get('index_minute', 8000)
        self.base_save_dir = self.get_full_path_and_ensure_dir('index_minute_dir')

        self.max_workers = 10
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 450.0

    def _fetch_single_year(self, ts_code, year, start_time, end_time, file_path):
        yearly_chunks = []
        offset = 0
        
        while True:
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()

            try:
                df = self.pro.idx_mins(
                    ts_code=ts_code, freq='1min', 
                    start_date=start_time, end_date=end_time,
                    limit=self.page_limit, offset=offset
                )
                if df is None or df.empty:
                    break
                yearly_chunks.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
            except Exception as e:
                return False, f"API拉取错误: {e}"

        if not yearly_chunks:
            return True, "无数据"

        try:
            df_new = pd.concat(yearly_chunks, ignore_index=True)
            df_new['trade_date'] = df_new['trade_time'].str[:10].str.replace('-', '').astype(np.int32)
            
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new.drop_duplicates(subset=['trade_time'], keep='last', inplace=True)
                
            for c in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                if c in df_new.columns: df_new[c] = df_new[c].astype(np.float32)
            
            df_new.sort_values(by=['trade_time'], inplace=True)
            df_new.to_parquet(file_path, index=False)
            return True, f"成功 ({len(df_new)}条)"
        except Exception as e:
            return False, f"落盘错误: {e}"

    def sync(self, ts_code, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
            
        self.logger.info(f"=== 开始同步 [特定指数 {ts_code} 1分钟线] (并发版) ===")
        code_dir = os.path.join(self.base_save_dir, ts_code.replace('.', '_'))
        os.makedirs(code_dir, exist_ok=True)
        
        years = [str(y) for y in range(int(start_date[:4]), int(target_end_date[:4]) + 1)]
        tasks = []
        
        for year in years:
            file_path = os.path.join(code_dir, f"{year}.parquet")
            if year < target_end_date[:4] and os.path.exists(file_path):
                continue
                
            y_start = max(start_date, f"{year}0101")
            y_end = min(target_end_date, f"{year}1231")
            start_time = f"{y_start[:4]}-{y_start[4:6]}-{y_start[6:8]} 09:00:00"
            end_time = f"{y_end[:4]}-{y_end[4:6]}-{y_end[6:8]} 16:00:00"
            
            tasks.append((year, start_time, end_time, file_path))

        if not tasks:
            self.logger.info(f"{ts_code} 数据已最新。")
            return

        self.logger.info(f"分配 {len(tasks)} 个年份并行下载任务...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_year = {
                executor.submit(self._fetch_single_year, ts_code, t[0], t[1], t[2], t[3]): t[0] 
                for t in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    success, msg = future.result()
                    if success:
                        self.logger.info(f"✅ {ts_code} [{year}年] 落地完毕: {msg}")
                    else:
                        self.logger.error(f"❌ {ts_code} [{year}年] 失败: {msg}")
                except Exception as exc:
                    self.logger.error(f"❌ {ts_code} [{year}年] 线程崩溃: {exc}")