import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class AnalystReportDownloader(BaseDownloader):
    """券商盈利预测/研报数据下载器 (按自然日横截面提取，按年归档，统一本地日历)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('report_rc', 200))
        self.page_limit = config['api']['page_limits'].get('report_rc', 5000)
        self.save_dir = self.get_full_path_and_ensure_dir('analyst_report_dir')
        
        # 严格按照基准路径合成日历文件路径
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'chn_stock_data/calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

    def _get_calendar_dates(self, start_date, end_date):
        """从本地日历获取自然日序列 (不过滤 is_open==1，因为周末也会发研报)"""
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步脚本！")
            
        df_cal = pd.read_parquet(self.cal_file)
        
        start_int = int(start_date)
        end_int = int(end_date)
        
        # 核心：完全依赖交易所官方日历时间轴，涵盖节假日
        mask = (df_cal['cal_date'] >= start_int) & (df_cal['cal_date'] <= end_int)
        
        # 筛选出来后，强制转回字符串列表，供后续 Tushare 接口使用
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        """扫描本地已下载的报告日期"""
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(os.path.join(self.save_dir, file), columns=['report_date'])
                    local_dates.update(df['report_date'].astype(str).unique().tolist())
                except Exception as e:
                    self.logger.warning(f"读取本地文件 {file} 失败: {e}")
        return local_dates

    def sync(self, start_date='20100101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [券商盈利预测研报] (区间: {start_date} -> {target_end_date}) ===")
        
        # 完美替换：使用本地数据湖的统一日历
        all_calendar_dates = self._get_calendar_dates(start_date, target_end_date)
        
        if not all_calendar_dates:
            self.logger.warning("目标区间内没有找到日历日期，请检查输入或日历文件。")
            return
        
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(all_calendar_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地研报数据已完全覆盖目标区间，无需更新。")
            return
            
        self.logger.info(f"发现 {len(missing_dates)} 个自然日缺失，开始逐日拉取...")

        # 按年分组批处理落盘，减少频繁的磁盘 I/O
        dates_by_year = {}
        for date in missing_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            self.logger.info(f"-> 正在处理 {year} 年缺失数据 (共 {len(dates)} 天)...")
            yearly_new_data = []
            
            for date in tqdm(dates):
                try:
                    offset = 0
                    day_chunks = []
                    while True:
                        df = self.pro.report_rc(report_date=date, limit=self.page_limit, offset=offset)
                        
                        if df is None or df.empty:
                            break
                        df = df.dropna(axis=1,how='all')
                        day_chunks.append(df)
                        
                        if len(df) < self.page_limit:
                            break
                            
                        offset += self.page_limit
                        self.safe_sleep()
                    
                    if day_chunks:
                        yearly_new_data.append(pd.concat(day_chunks, ignore_index=True))
                    
                except Exception as e:
                    self.logger.error(f"拉取 {date} 研报失败: {e}")
            
            if yearly_new_data:
                file_path = os.path.join(self.save_dir, f"{year}.parquet")
                df_new = pd.concat(yearly_new_data, ignore_index=True)
                
                # 读取老数据进行合并
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    
                    # 踢除新数据中全为空的列，防止 concat 类型推断警告
                    df_new_clean = df_new.dropna(axis=1, how='all')
                    df_combined = pd.concat([df_old, df_new_clean], ignore_index=True)
                    
                    # 去重：同一股票、同日期、同机构、同作者、同预测报告期
                    subset_cols = ['ts_code', 'report_date', 'org_name', 'author_name', 'quarter']
                    subset_cols = [c for c in subset_cols if c in df_combined.columns]
                    df_combined.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
                else:
                    df_combined = df_new
                    
                # ==========================================
                # 终极数据清洗：消灭 None，终结 Object 毒药
                # ==========================================
                if 'report_date' in df_combined.columns:
                    df_combined['report_date'] = pd.to_numeric(df_combined['report_date'].astype(str).str.replace('-', ''), errors='coerce').fillna(0).astype(np.int32)
                
                obj_cols = df_combined.select_dtypes(include=['object']).columns
                
                # 基于你提供的图片字典，保护所有明确的字符串列
                safe_str_cols = [
                    'ts_code', 'name', 'report_title', 'report_type', 
                    'classify', 'org_name', 'author_name', 'quarter', 'rating'
                ]
                for col in obj_cols:
                    if col not in safe_str_cols:
                        # 非保护名单的 object 列，全部强转为 float，将 None 变为 np.nan
                        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

                # 将所有的 float64 极致降维为 float32
                float_cols = df_combined.select_dtypes(include=['float64']).columns
                if not float_cols.empty:
                    df_combined[float_cols] = df_combined[float_cols].astype(np.float32)

                # ==========================================
                
                df_combined.sort_values(by=['report_date', 'ts_code'], inplace=True)
                df_combined.to_parquet(file_path, index=False)
                self.logger.info(f"✅ 成功保存 {year}.parquet，当前累计包含 {len(df_combined)} 条预测记录。")
                
        self.logger.info("=== [券商盈利预测研报] 同步结束 ===")