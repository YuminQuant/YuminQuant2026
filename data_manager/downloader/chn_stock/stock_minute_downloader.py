import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class StockMinuteDownloader(BaseDownloader):
    """股票分钟线下载器 (按天横截面 Batch 并发极限提取)"""
    def __init__(self):
        config = ConfigManager().config
        # 注意：这里的 rate_limit 只是给基类传参，我们自己用锁实现了更精准的控制
        super().__init__(rate_limit=config['api']['rate_limits'].get('stock_minute', 400))
        self.page_limit = config['api']['page_limits'].get('stock_minute', 8000)
        self.save_dir = self.get_full_path_and_ensure_dir('stock_minute_dir')
        self.daily_pv_dir = self.get_full_path_and_ensure_dir('stock_daily_pv_dir')
        
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

        # ==========================================
        # 终极多线程与限流配置
        # ==========================================
        self.max_workers = 10              # 开启 10 个并发线程
        self.api_lock = threading.Lock()   # 全局发车闸机锁
        self.last_call_time = 0.0          # 上次请求发出的时间戳
        # 目标限流：450次/分钟 (VIP 额度为 500，留 50 次余量防封)
        # 意味着：绝对不允许两次请求的间隔小于 0.1333 秒
        self.min_interval = 60.0 / 450.0   

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError("未找到日历文件，请先同步日历！")
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

    def _get_valid_stocks_for_day(self, date_str):
        """精准提取当天的活跃股票池"""
        year = date_str[:4]
        pv_file = os.path.join(self.daily_pv_dir, f"{year}.parquet")
        if not os.path.exists(pv_file):
            return []
        try:
            df_pv = pd.read_parquet(pv_file, columns=['trade_date', 'ts_code'])
            df_day = df_pv[df_pv['trade_date'] == int(date_str)]
            return df_day['ts_code'].tolist()
        except Exception as e:
            self.logger.error(f"读取 {date_str} 日线股票池失败: {e}")
            return []

    def _fetch_batch_safe(self, ts_code_str, start_time, end_time, retry=3):
        """线程安全的 API 拉取函数：带严格的最小发车间隔控制"""
        # ----------------------------------------------------
        # 1. 物理隔离区：过闸机 (仅耗时毫秒级，或最多休眠 0.133 秒)
        # ----------------------------------------------------
        with self.api_lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            # 记录当前这辆车驶出闸机的准确时间
            self.last_call_time = time.time()
            
        # ----------------------------------------------------
        # 2. 自由狂奔区：网络 I/O (耗时约 1~2 秒，线程完全并行)
        # ----------------------------------------------------
        for attempt in range(retry):
            try:
                df = self.pro.stk_mins(
                    ts_code=ts_code_str, 
                    freq='1min', 
                    start_date=start_time, 
                    end_date=end_time
                )
                return df
            except Exception as e:
                if attempt == retry - 1:
                    return e # 重试失败后，把异常当做结果返回，防止线程崩溃
                time.sleep(1) # 遇到网络抖动，休息 1 秒后重试

    def sync(self, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [股票分钟线] (多线程并发 Batch) ({start_date} -> {target_end_date}) ===")

        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地分钟线数据已完全覆盖目标区间。")
            return

        total_days = len(missing_dates)
        self.logger.info(f"发现 {len(missing_dates)} 个交易日缺失，开始启动并发引擎...")

        for idx, date in enumerate(missing_dates):
            self._fetch_and_save_single_day(date, current_idx=idx+1, total_days=total_days)
            
        self.logger.info("=== [股票分钟线] 同步完毕 ===")

    def _fetch_and_save_single_day(self, date, current_idx=1, total_days=1):
        year = date[:4]
        valid_codes = self._get_valid_stocks_for_day(date)
        if not valid_codes:
            self.logger.warning(f"[{date}] 未找到交易股票，跳过。")
            return
            
        # 30只 * 241行/天 = 7230行 < 8000行 (单次API上限)
        batch_size = 32
        code_batches = [valid_codes[i:i + batch_size] for i in range(0, len(valid_codes), batch_size)]
        
        fmt_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        start_time = f"{fmt_date} 09:29:00"
        end_time = f"{fmt_date} 15:01:00"
        
        day_chunks = []
        self.logger.info(f"-> 提取 [{date}] 分钟线，共 {len(valid_codes)} 只，切分为 {len(code_batches)} 个 Batch...")
        
        completed_count = 0
        total_batches = len(code_batches)
        
        # 启动线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务并建立字典映射，方便追溯是哪个 batch 出了问题
            future_to_batch = {
                executor.submit(self._fetch_batch_safe, ",".join(batch), start_time, end_time): idx 
                for idx, batch in enumerate(code_batches)
            }
            
            # as_completed: 哪个线程先回来，就先处理哪个，无须按顺序等待
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        self.logger.error(f"   [{date}] Batch {batch_idx+1} 彻底失败: {result}")
                    elif result is not None and not result.empty:
                        day_chunks.append(result)
                except Exception as exc:
                    self.logger.error(f"   [{date}] Batch {batch_idx+1} 抛出未捕获异常: {exc}")
                    
                # 只在进度为 20 的倍数或结束时打印，防止控制台刷屏影响性能
                if completed_count % 20 == 0 or completed_count == total_batches:
                    self.logger.info(f"   [{date}] 线程池进度: {completed_count}/{total_batches}")

        # 所有线程执行完毕，单线程清洗落盘
        if day_chunks:
            df_day = pd.concat(day_chunks, ignore_index=True)
            # df_day['trade_date'] = int(date)
            
            # 强转 float32，给内存和硬盘减负
            float_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
            for col in float_cols:
                if col in df_day.columns:
                    df_day[col] = df_day[col].astype(np.float32)
                    
            if 'trade_time' in df_day.columns:
                df_day.sort_values(by=['ts_code', 'trade_time'], inplace=True)
                
            year_dir = os.path.join(self.save_dir, year)
            os.makedirs(year_dir, exist_ok=True)
            file_path = os.path.join(year_dir, f"{date}.parquet")
            
            df_day.to_parquet(file_path, index=False)
            remaining_days = total_days - current_idx
            self.logger.info(f"✅ [{date}] 完美落盘！(含 {len(df_day)} 行) [大盘进度: {current_idx}/{total_days} | 剩余: {remaining_days} 天]")
        else:
            self.logger.warning(f"[{date}] 全部线程拉取完毕，但无数据返回！")