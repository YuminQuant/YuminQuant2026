import os
import time
import threading
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class BaseFinancialDownloader(BaseDownloader):
    """财务报表双引擎下载器 (历史 VIP period 全量 + 每日普通 ann_date 增量)"""
    def __init__(self, vip_api_name, normal_api_name, dir_config_key, task_name):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('financial', 200))
        self.page_limit = config['api']['page_limits'].get('financial', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir(dir_config_key)
        
        # 接口分流
        self.vip_api_name = vip_api_name
        self.normal_api_name = normal_api_name
        self.task_name = task_name
        
        # 多线程限流配置 (针对增量更新的 ts_code batch 机制)
        self.max_workers = 5 
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 180.0 # 每分钟180次安全线

    def _generate_periods(self, start_year, end_year):
        periods = []
        for year in range(start_year, end_year + 1):
            for q in ['0331', '0630', '0930', '1231']:
                periods.append(f"{year}{q}")
        return periods

    def _get_all_active_ts_codes(self):
        """获取当前全市场存活的股票代码"""
        try:
            # 增量拉取只需关注上市(L)和暂停上市(P)的股票
            df_l = self.pro.stock_basic(list_status='L', fields='ts_code')
            df_p = self.pro.stock_basic(list_status='P', fields='ts_code')
            codes = pd.concat([df_l, df_p])['ts_code'].unique().tolist()
            return codes
        except Exception as e:
            self.logger.error(f"获取股票基础列表失败: {e}")
            return []

    def _fetch_incremental_batch(self, ts_codes, target_date, api_func, retry=3):
        """单次 Batch 拉取增量 (内置分页，防止超大单日公告量)"""
        ts_code_str = ",".join(ts_codes)
        all_chunks = []
        offset = 0
        
        while True:
            # 严格限流保护
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()
                
            success = False
            for attempt in range(retry):
                try:
                    # 核心：ts_code 与 ann_date 同时锁定
                    df = api_func(ts_code=ts_code_str, ann_date=target_date, limit=self.page_limit, offset=offset)
                    success = True
                    break
                except Exception as e:
                    if attempt == retry - 1:
                        return False, f"API错误: {e}"
                    time.sleep(1)
            
            if not success or df is None or df.empty:
                break
                
            all_chunks.append(df)
            if len(df) < self.page_limit:
                break
            offset += self.page_limit
            
        if all_chunks:
            return True, pd.concat(all_chunks, ignore_index=True)
        return True, None

    def _process_and_save(self, df_chunk):
        """统一的数据清洗与 PIT 归档落盘逻辑"""
        if df_chunk is None or df_chunk.empty:
            return

        if 'ann_date' not in df_chunk.columns:
            return
            
        # 1. 剔除无效公告日，提取公告年份归档
        df_chunk = df_chunk[df_chunk['ann_date'].notna()]
        df_chunk = df_chunk[df_chunk['ann_date'] != '']
        df_chunk['ann_year'] = df_chunk['ann_date'].astype(str).str[:4]
        
        # 2. 日期转 int32 优化查询
        for date_col in ['ann_date', 'f_ann_date', 'end_date']:
            if date_col in df_chunk.columns:
                df_chunk[date_col] = pd.to_numeric(df_chunk[date_col].str.replace('-', ''), errors='coerce').fillna(0).astype(np.int32)
                
        # ==========================================
        # 3. 核心大清洗：终结 Object 毒药
        # ==========================================
        if 'update_flag' in df_chunk.columns:
            df_chunk['update_flag'] = pd.to_numeric(df_chunk['update_flag'], errors='coerce').fillna(0).astype(np.int32)

        obj_cols = df_chunk.select_dtypes(include=['object']).columns
        safe_str_cols = ['ts_code', 'ann_year']
        for col in obj_cols:
            if col not in safe_str_cols:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')

        # 4. 极致降维：float64 -> float32
        float_cols = df_chunk.select_dtypes(include=['float64']).columns
        if not float_cols.empty:
            df_chunk[float_cols] = df_chunk[float_cols].astype(np.float32)
            
        # ==========================================
        # 5. 按公告年份 (ann_year) 落盘
        # ==========================================
        grouped = df_chunk.groupby('ann_year')
        for year, df_year in grouped:
            if not year or pd.isna(year):
                continue
                
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            df_save = df_year.drop(columns=['ann_year'])
            
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_save_clean = df_save.dropna(axis=1, how='all')
                df_save = pd.concat([df_old, df_save_clean], ignore_index=True)
                
                # PIT 核心：同股票、同报告期、同公告日，保留最后一条
                subset_cols = [c for c in ['ts_code', 'end_date', 'f_ann_date', 'ann_date'] if c in df_save.columns]
                df_save.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                
            df_save.sort_values(by=['ts_code', 'ann_date', 'end_date'], inplace=True)
            df_save.to_parquet(file_path, index=False)

    def sync(self, mode='historical', start_year=2009, target_date=None):
        if mode == 'historical':
            api_func = getattr(self.pro, self.vip_api_name)
            current_year = datetime.now().year
            periods = self._generate_periods(start_year, current_year)
            self.logger.info(f"=== [历史全量] 同步 {self.task_name} (VIP Period 模式) ===")
            
            for period in periods:
                self.logger.info(f"-> 正在拉取报告期: {period} ...")
                offset = 0
                all_chunks = []
                while True:
                    try:
                        df = api_func(period=period, limit=self.page_limit, offset=offset)
                        if df is None or df.empty:
                            break
                        df = df.dropna(axis=1, how='all')
                        all_chunks.append(df)
                        if len(df) < self.page_limit:
                            break
                        offset += self.page_limit
                        self.safe_sleep()
                    except Exception as e:
                        self.logger.error(f"拉取 {period} 失败: {e}")
                        break
                
                if all_chunks:
                    df_period = pd.concat(all_chunks, ignore_index=True)
                    self._process_and_save(df_period)
            self.logger.info(f"=== [历史全量] {self.task_name} 同步完毕 ===")

        elif mode == 'incremental':
            if target_date is None:
                target_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
                
            self.logger.info(f"=== [每日增量] 同步 {self.task_name} (普通 Batch 模式, 公告日: {target_date}) ===")
            
            api_func = getattr(self.pro, self.normal_api_name)
            all_codes = self._get_all_active_ts_codes()
            if not all_codes:
                self.logger.warning("未获取到存活股票代码！")
                return
                
            # Batch Size 设为 40只股票一组，兼顾网络速度与 Tushare 接口承载力
            batch_size = 40
            code_batches = [all_codes[i:i+batch_size] for i in range(0, len(all_codes), batch_size)]
            
            day_chunks = []
            completed = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self._fetch_incremental_batch, batch, target_date, api_func): idx
                    for idx, batch in enumerate(code_batches)
                }
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    completed += 1
                    status, result = future.result()
                    if status and result is not None and not result.empty:
                        day_chunks.append(result)
                    
                    if completed % 20 == 0 or completed == len(code_batches):
                        self.logger.info(f"   Batch 进度: {completed}/{len(code_batches)}")
                        
            if day_chunks:
                df_daily = pd.concat(day_chunks, ignore_index=True)
                self._process_and_save(df_daily)
            self.logger.info(f"=== [每日增量] {self.task_name} 同步完毕 ===")

# ====== 实例化具体的报表类 (传入 vip_api 和 normal_api 两个名字) ======
class IncomeDownloader(BaseFinancialDownloader):
    def __init__(self): super().__init__('income_vip', 'income', 'fin_income_dir', '利润表')

class BalanceSheetDownloader(BaseFinancialDownloader):
    def __init__(self): super().__init__('balancesheet_vip', 'balancesheet', 'fin_balance_dir', '资产负债表')

class CashFlowDownloader(BaseFinancialDownloader):
    def __init__(self): super().__init__('cashflow_vip', 'cashflow', 'fin_cashflow_dir', '现金流量表')

class ForecastDownloader(BaseFinancialDownloader):
    def __init__(self): super().__init__('forecast_vip', 'forecast', 'fin_forecast_dir', '业绩预告')

class ExpressDownloader(BaseFinancialDownloader):
    def __init__(self): super().__init__('express_vip', 'express', 'fin_express_dir', '业绩快报')