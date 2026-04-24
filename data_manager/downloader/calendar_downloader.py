import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class CalendarDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        rate_limit = config['api']['rate_limits'].get('trade_cal', 200)
        super().__init__(rate_limit=rate_limit)
        
        self.save_dir = self.get_full_path_and_ensure_dir('calendar_dir')
        self.exchanges = self.config['exchanges']['supported']

    def sync(self, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = f"{datetime.now().year}1231"
            
        self.logger.info(f"=== 开始同步交易日历任务 (请求区间: {start_date} -> {target_end_date}) ===")
        
        for exc in self.exchanges:
            file_name = f"trade_cal_{exc}.parquet"
            file_path = os.path.join(self.save_dir, file_name)
            
            df_local = None
            fetch_intervals = [] 
            
            if os.path.exists(file_path):
                try:
                    df_local = pd.read_parquet(file_path)
                    # 【核心修改 1】：从本地读取的 int32 转换回 str 进行运算
                    local_min = str(df_local['cal_date'].min())
                    local_max = str(df_local['cal_date'].max())
                    
                    if start_date < local_min:
                        local_min_obj = datetime.strptime(local_min, '%Y%m%d')
                        backfill_end = (local_min_obj - timedelta(days=1)).strftime('%Y%m%d')
                        fetch_intervals.append((start_date, backfill_end))
                        self.logger.info(f"[{exc}] 发现历史缺失，需回填: {start_date} -> {backfill_end}")
                        
                    if target_end_date > local_max:
                        local_max_obj = datetime.strptime(local_max, '%Y%m%d')
                        forward_start = (local_max_obj + timedelta(days=1)).strftime('%Y%m%d')
                        fetch_intervals.append((forward_start, target_end_date))
                        self.logger.info(f"[{exc}] 发现新数据缺失，需追加: {forward_start} -> {target_end_date}")
                        
                    if not fetch_intervals:
                        self.logger.info(f"[{exc}] 本地数据已完全覆盖请求区间，跳过。")
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"[{exc}] 读取本地文件失败，将执行全量同步: {e}")
                    fetch_intervals = [(start_date, target_end_date)]
            else:
                self.logger.info(f"[{exc}] 未发现本地数据，将执行全量同步: {start_date} -> {target_end_date}")
                fetch_intervals = [(start_date, target_end_date)]
                
            all_new_data = []
            for f_start, f_end in fetch_intervals:
                try:
                    df_new = self.pro.trade_cal(exchange=exc, start_date=f_start, end_date=f_end)
                    if df_new is not None and not df_new.empty:
                        all_new_data.append(df_new)
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"[{exc}] 获取区间 {f_start}->{f_end} 失败: {e}")

            if all_new_data:
                df_new_combined = pd.concat(all_new_data, ignore_index=True)
                
                if df_local is not None and not df_local.empty:
                    final_df = pd.concat([df_local, df_new_combined], ignore_index=True)
                    final_df.drop_duplicates(subset=['cal_date'], keep='last', inplace=True)
                else:
                    final_df = df_new_combined
                    
                # 【核心修改 2】：落盘前强制将 cal_date 转换为 int32
                if 'cal_date' in final_df.columns:
                    final_df['cal_date'] = final_df['cal_date'].astype(np.int32)
                    
                final_df.sort_values(by=['cal_date'], inplace=True)
                final_df.to_parquet(file_path, index=False)
                self.logger.info(f"[{exc}] 同步成功！新增 {len(df_new_combined)} 条，当前总计 {len(final_df)} 条 (已转换为 int32)。")
            else:
                self.logger.info(f"[{exc}] 请求执行完毕，但未获取到新数据。")

        self.logger.info("=== 交易日历同步任务结束 ===")