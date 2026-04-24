import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class BaseUSFinancialDownloader(BaseDownloader):
    """美股财务数据基类 (按 Period 划分，Batch 多线程并发 + 内部翻页)"""
    def __init__(self, api_method_name, dir_config_key, task_name):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('us_financial', 200))
        self.page_limit = config['api']['page_limits'].get('us_financial', 10000)
        self.save_dir = self.get_full_path_and_ensure_dir(dir_config_key)
        self.basic_dir = self.get_full_path_and_ensure_dir('us_basic_dir')
        
        self.api_method_name = api_method_name
        self.task_name = task_name

        # 多线程限流配置 (200次/分钟 -> 安全线180次 -> 0.33秒/次)
        self.max_workers = 8 
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 180.0 

    def _get_all_us_stocks(self):
        """从 us_basic 获取全部美股代码"""
        basic_file = os.path.join(self.basic_dir, "us_basic.parquet")
        if not os.path.exists(basic_file):
            self.logger.error("未找到 us_basic.parquet！请先同步美股基础列表。")
            return []
        df_basic = pd.read_parquet(basic_file, columns=['ts_code'])
        return df_basic['ts_code'].unique().tolist()

    def _generate_periods(self, start_year, end_year):
        """美股标准财报期，生成全量季度以防遗漏"""
        periods = []
        for year in range(start_year, end_year + 1):
            for q in ['0331', '0630', '0930', '1231']:
                periods.append(f"{year}{q}")
        return periods

    def _fetch_batch_with_pagination(self, ts_code_str, period, api_func, retry=3):
        """单 Batch 内部自带 Offset 翻页的健壮拉取逻辑"""
        batch_chunks = []
        offset = 0
        
        while True:
            # 闸机限流保护
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()
                
            success = False
            for attempt in range(retry):
                try:
                    df = api_func(ts_code=ts_code_str, period=period, limit=self.page_limit, offset=offset)
                    success = True
                    break
                except Exception as e:
                    if attempt == retry - 1:
                        return False, f"重试耗尽: {e}"
                    time.sleep(1)
                    
            if not success or df is None or df.empty:
                break
                
            batch_chunks.append(df)
            if len(df) < self.page_limit:
                break
            offset += self.page_limit

        if not batch_chunks:
            return True, None
            
        return True, pd.concat(batch_chunks, ignore_index=True)

    def sync(self, start_year=2010):
        self.logger.info(f"=== 开始同步 [{self.task_name}] (EAV长表模式) ===")
        all_codes = self._get_all_us_stocks()
        if not all_codes:
            return
            
        # 美股时区调整：防止提前跑到未来年份
        us_now = datetime.now(timezone(timedelta(hours=-5)))
        current_year = us_now.year
        
        periods = self._generate_periods(start_year, current_year)
        api_func = getattr(self.pro, self.api_method_name)

        # Batch Size = 40 (防止美股科目多突破 10000 行限制)
        batch_size = 40
        code_batches = [all_codes[i:i + batch_size] for i in range(0, len(all_codes), batch_size)]
        
        for period in periods:
            year = period[:4]
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            
            # 增量判定：如果历史整年的文件已存在，跳过
            if int(year) < current_year - 1 and os.path.exists(file_path):
                continue
                
            self.logger.info(f"-> 正在拉取报告期: {period}，切分为 {len(code_batches)} 个 Batch...")
            
            period_chunks = []
            completed = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._fetch_batch_with_pagination, ",".join(batch), period, api_func): idx 
                    for idx, batch in enumerate(code_batches)
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    completed += 1
                    try:
                        status, result = future.result()
                        if not status:
                            self.logger.error(f"[{period}] Batch {batch_idx+1} 失败: {result}")
                        elif result is not None:
                            period_chunks.append(result)
                    except Exception as exc:
                        self.logger.error(f"[{period}] 线程异常: {exc}")
                        
                    if completed % 50 == 0 or completed == len(code_batches):
                        self.logger.info(f"   进度 [{period}]: {completed}/{len(code_batches)}")
                        
            # 合并该 Period 数据落盘
            if period_chunks:
                df_period = pd.concat(period_chunks, ignore_index=True)
                
                # 强转日期与数值
                for date_col in ['end_date', 'ann_date', 'f_ann_date']:
                    if date_col in df_period.columns:
                        df_period[date_col] = pd.to_numeric(df_period[date_col].str.replace('-', ''), errors='coerce').fillna(0).astype(np.int32)
                
                if 'ind_value' in df_period.columns:
                    df_period['ind_value'] = df_period['ind_value'].astype(np.float32)
                    
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    df_combined = pd.concat([df_old, df_period], ignore_index=True)
                    # 去重保留最后一条：兼顾财报修正 (Point-in-Time)
                    subset_cols = ['ts_code', 'end_date', 'ind_name']
                    subset_cols = [c for c in subset_cols if c in df_combined.columns]
                    df_combined.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                else:
                    df_combined = df_period
                    
                df_combined.sort_values(by=['end_date', 'ts_code'], inplace=True)
                df_combined.to_parquet(file_path, index=False)
                self.logger.info(f"✅ 成功合并保存 {period} 数据至 {year}.parquet！")

# ================= 实例化具体的美股财务类 =================

class USBalanceSheetDownloader(BaseUSFinancialDownloader):
    def __init__(self):
        super().__init__('us_balancesheet', 'us_balance_dir', '美股资产负债表')

class USCashFlowDownloader(BaseUSFinancialDownloader):
    def __init__(self):
        super().__init__('us_cashflow', 'us_cashflow_dir', '美股现金流量表')

class USIncomeDownloader(BaseUSFinancialDownloader):
    def __init__(self):
        super().__init__('us_income', 'us_income_dir', '美股利润表')