import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class OptionMinuteDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('opt_minute', 500))
        self.page_limit = config['api']['page_limits'].get('opt_minute', 8000)
        self.save_dir = self.get_full_path_and_ensure_dir('opt_minute_dir')
        self.basic_dir = self.get_full_path_and_ensure_dir('opt_basic_dir')

        self.max_workers = 10
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 450.0

    def _fetch_single_option(self, ts_code, start_time, end_time, file_path):
        contract_chunks = []
        offset = 0
        
        while True:
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()

            try:
                df = self.pro.opt_mins(
                    ts_code=ts_code, freq='1min', 
                    start_date=start_time, end_date=end_time,
                    limit=self.page_limit, offset=offset
                )
                if df is None or df.empty:
                    break
                contract_chunks.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
            except Exception as e:
                return False, f"API出错: {e}"

        if not contract_chunks:
            return True, "无增量数据"

        try:
            df_new = pd.concat(contract_chunks, ignore_index=True)
            df_new['trade_date'] = df_new['trade_time'].str[:10].str.replace('-', '').astype(np.int32)
            
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
                df_new.drop_duplicates(subset=['trade_time'], keep='last', inplace=True)
                
            for c in ['open', 'high', 'low', 'close', 'vol', 'amount', 'oi']:
                if c in df_new.columns: df_new[c] = df_new[c].astype(np.float32)
                
            df_new.sort_values(by=['trade_time'], inplace=True)
            df_new.to_parquet(file_path, index=False)
            return True, "成功"
        except Exception as e:
            return False, f"落盘出错: {e}"

    def sync(self):
        self.logger.info("=== 开始同步 [期权 1 分钟行情] (多线程并发极速版) ===")
        
        basic_file = os.path.join(self.basic_dir, "opt_basic.parquet")
        if not os.path.exists(basic_file):
            self.logger.error("未找到 opt_basic.parquet！请先运行期权基础信息同步。")
            return
            
        df_basic = pd.read_parquet(basic_file).dropna(subset=['list_date'])
        bj_today = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
        
        tasks = []
        for _, row in df_basic.iterrows():
            ts_code = row['ts_code']
            exchange = row.get('exchange', ts_code.split('.')[-1])
            list_date = row['list_date']
            delist_date = row['delist_date'] if pd.notna(row['delist_date']) else bj_today
            
            exc_dir = os.path.join(self.save_dir, exchange)
            os.makedirs(exc_dir, exist_ok=True)
            file_path = os.path.join(exc_dir, f"{ts_code.replace('.', '_')}.parquet")
            
            start_date = list_date
            if os.path.exists(file_path):
                if delist_date < bj_today:
                    continue
                df_local = pd.read_parquet(file_path, columns=['trade_time'])
                if not df_local.empty:
                    start_date = df_local['trade_time'].max()[:10].replace('-', '')
            
            if start_date > bj_today:
                continue

            fetch_end_date = min(delist_date, bj_today)
            start_time = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]} 09:00:00"
            end_time = f"{fetch_end_date[:4]}-{fetch_end_date[4:6]}-{fetch_end_date[6:8]} 23:59:00"
            
            tasks.append((ts_code, start_time, end_time, file_path))

        total_tasks = len(tasks)
        if total_tasks == 0:
            self.logger.info("所有期权合约已最新。")
            return
            
        self.logger.info(f"分配 {total_tasks} 个期权下载任务，线程池点火...")

        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {
                executor.submit(self._fetch_single_option, t[0], t[1], t[2], t[3]): t[0] 
                for t in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_code):
                ts_code = future_to_code[future]
                completed += 1
                try:
                    success, msg = future.result()
                    if not success:
                        self.logger.error(f"[{ts_code}] 异常: {msg}")
                except Exception as exc:
                    self.logger.error(f"[{ts_code}] 崩溃: {exc}")
                    
                if completed % 500 == 0 or completed == total_tasks:
                    self.logger.info(f"   大盘进度: {completed}/{total_tasks} ({completed/total_tasks:.1%})")

        self.logger.info("=== [期权 1 分钟行情] 同步结束 ===")