import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

DIVIDEND_FIELDS = (
    "ts_code,end_date,ann_date,div_proc,stk_div,cash_div,cash_div_tax,"
    "record_date,ex_date,pay_date,imp_ann_date,stk_co_rate,div_listdate,"
    "stk_bo_rate,base_date,base_share"
)

class DividendDownloader(BaseDownloader):
    """分红送股下载器 (终极版：按 ann_date 自然日提取，PIT存储，完美类型清洗)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('dividend', 500))
        self.page_limit = config['api']['page_limits'].get('dividend', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir('fin_dividend_dir')
        
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'chn_stock_data/calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

    def _remove_year_files(self, start_date, end_date):
        start_year = int(str(start_date)[:4])
        end_year = int(str(end_date)[:4])
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"removed old dividend parquet for rebuild: {file_path}")

    def _get_calendar_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步脚本！")
            
        df_cal = pd.read_parquet(self.cal_file)
        start_int = int(start_date)
        end_int = int(end_date)
        mask = (df_cal['cal_date'] >= start_int) & (df_cal['cal_date'] <= end_int)
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _process_and_save(self, df_chunk):
        """统一的数据清洗与落盘逻辑"""
        if df_chunk is None or df_chunk.empty: 
            return
        
        # 1. 改为以 公告日 (ann_date) 年份作为归档标准
        if 'ann_date' not in df_chunk.columns: 
            return
            
        df_chunk = df_chunk[df_chunk['ann_date'].notna()]
        df_chunk = df_chunk[df_chunk['ann_date'] != '']
        df_chunk['ann_year'] = df_chunk['ann_date'].astype(str).str[:4]
        
        # 2. 日期字段统一压缩为 int32
        date_cols = ['ann_date', 'end_date', 'ex_date', 'pay_date', 'div_listdate', 'record_date', 'imp_ann_date']
        for date_col in date_cols:
            if date_col in df_chunk.columns:
                df_chunk[date_col] = pd.to_numeric(df_chunk[date_col].str.replace('-', ''), errors='coerce').fillna(0).astype(np.int32)

        # ==========================================
        # 3. 核心清洗：消灭 None，终结 Object 毒药
        # ==========================================
        obj_cols = df_chunk.select_dtypes(include=['object']).columns
        
        # 保护名单：替换 end_year 为 ann_year
        safe_str_cols = ['ts_code', 'ann_year', 'div_proc'] 
        for col in obj_cols:
            if col not in safe_str_cols:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')

        # 4. 极致降维：float64 -> float32
        float_cols = df_chunk.select_dtypes(include=['float64']).columns
        if not float_cols.empty: 
            df_chunk[float_cols] = df_chunk[float_cols].astype(np.float32)
            
        # ==========================================
        # 5. 按公告年份 (ann_year) 归档落盘
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
                
                # PIT核心不变
                subset_cols = [c for c in ['ts_code', 'end_date', 'ann_date'] if c in df_save.columns]
                df_save.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                
            # 排序维度调整
            df_save.sort_values(by=['ts_code', 'ann_date', 'end_date'], inplace=True)
            df_save.to_parquet(file_path, index=False)

    def sync(self, start_date='20090101', target_end_date=None, rebuild=False):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
        if rebuild:
            start_date = f"{str(start_date)[:4]}0101"
            self._remove_year_files(start_date, target_end_date)

        self.logger.info(f"=== 开始同步 [分红送股] (按 ann_date: {start_date} -> {target_end_date}) ===")

        all_calendar_dates = self._get_calendar_dates(start_date, target_end_date)
        if not all_calendar_dates:
            self.logger.warning("目标区间内没有找到日历日期，请检查输入或日历文件。")
            return

        dates_by_year = {}
        for date in all_calendar_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            if len(all_calendar_dates) > 10: 
                self.logger.info(f"-> 正在处理 {year} 年分红公告 (共 {len(dates)} 天)...")
            
            yearly_new_data = []
            for date in tqdm(dates):
                try:
                    offset = 0
                    while True:
                        df = self.pro.dividend(
                            ann_date=date,
                            fields=DIVIDEND_FIELDS,
                            limit=self.page_limit,
                            offset=offset,
                        )
                        if df is None or df.empty: 
                            break
                        df = df.dropna(axis=1, how='all')
                        yearly_new_data.append(df)
                        if len(df) < self.page_limit: 
                            break
                        offset += self.page_limit
                        self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 分红数据失败: {e}")

            if yearly_new_data:
                self._process_and_save(pd.concat(yearly_new_data, ignore_index=True))
                
        self.logger.info("=== [分红送股] 同步结束 ===")
