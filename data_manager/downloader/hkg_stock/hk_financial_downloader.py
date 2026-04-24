import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class BaseHKFinancialDownloader(BaseDownloader):
    """港股财务数据基类 (按 Period 划分，Batch 多线程并发 + 内部翻页)"""
    def __init__(self, api_method_name, dir_config_key, task_name):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('hk_financial', 200))
        self.page_limit = config['api']['page_limits'].get('hk_financial', 10000)
        self.save_dir = self.get_full_path_and_ensure_dir(dir_config_key)
        self.basic_dir = self.get_full_path_and_ensure_dir('hk_basic_dir')
        
        self.api_method_name = api_method_name
        self.task_name = task_name

        # 多线程限流配置 (港股频控200次/分钟 -> 约180次安全线 -> 0.33秒/次)
        self.max_workers = 8 
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 180.0 

    def _get_all_hk_stocks(self):
        """从 hk_basic 获取全部港股代码"""
        basic_file = os.path.join(self.basic_dir, "hk_basic.parquet")
        if not os.path.exists(basic_file):
            self.logger.error("未找到 hk_basic.parquet！请先同步港股基础列表。")
            return []
        df_basic = pd.read_parquet(basic_file, columns=['ts_code'])
        return df_basic['ts_code'].unique().tolist()

    def _generate_periods(self, start_year, end_year):
        """港股主要发中报和年报，部分有一季报和三季报，全量生成以防遗漏"""
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
        all_codes = self._get_all_hk_stocks()
        if not all_codes:
            return
            
        current_year = datetime.now().year
        periods = self._generate_periods(start_year, current_year)
        api_func = getattr(self.pro, self.api_method_name)

        # Batch Size = 40 (40只 * ~150个科目 = ~6000行 < 10000行)
        batch_size = 40
        code_batches = [all_codes[i:i + batch_size] for i in range(0, len(all_codes), batch_size)]
        
        for period in periods:
            year = period[:4]
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            
            # 如果是非当前年的数据已存在且大于一定阈值（简单判断是否下完），实际业务中可加入更精细的增量判断
            if int(year) < current_year - 1 and os.path.exists(file_path):
                # 对于历史整年且已有文件的，可以选择跳过
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
                        
                    if completed % 20 == 0 or completed == len(code_batches):
                        self.logger.info(f"   进度 [{period}]: {completed}/{len(code_batches)}")
                        
            # 合并该 Period 的所有数据并落盘
            if period_chunks:
                df_period = pd.concat(period_chunks, ignore_index=True)
                
                # 类型优化
                for date_col in ['end_date', 'ann_date']:
                    if date_col in df_period.columns:
                        df_period[date_col] = pd.to_numeric(df_period[date_col].str.replace('-', ''), errors='coerce').fillna(0).astype(np.int32)
                
                if 'ind_value' in df_period.columns:
                    df_period['ind_value'] = df_period['ind_value'].astype(np.float32)
                    
                # 如果文件已存在（可能包含该年其他 Period 的数据），则读取后合并
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    df_combined = pd.concat([df_old, df_period], ignore_index=True)
                    # 去重逻辑：同股票、同报告期、同科目的保留最后一条
                    subset_cols = ['ts_code', 'end_date', 'ind_name']
                    subset_cols = [c for c in subset_cols if c in df_combined.columns]
                    df_combined.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                else:
                    df_combined = df_period
                    
                df_combined.sort_values(by=['end_date', 'ts_code'], inplace=True)
                df_combined.to_parquet(file_path, index=False)
                self.logger.info(f"✅ 成功合并保存 {period} 数据至 {year}.parquet！")

# ================= 实例化具体的港股财务类 =================

class HKBalanceSheetDownloader(BaseHKFinancialDownloader):
    def __init__(self):
        super().__init__('hk_balancesheet', 'hk_balance_dir', '港股资产负债表')

class HKCashFlowDownloader(BaseHKFinancialDownloader):
    def __init__(self):
        super().__init__('hk_cashflow', 'hk_cashflow_dir', '港股现金流量表')

class HKIncomeDownloader(BaseHKFinancialDownloader):
    def __init__(self):
        super().__init__('hk_income', 'hk_income_dir', '港股利润表')