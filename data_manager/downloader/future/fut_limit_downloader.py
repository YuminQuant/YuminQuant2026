import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager

class FutureLimitDownloader(BaseDownloader):
    """期货每日涨跌停价格下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('fut_limit', 500))
        self.save_dir = self.get_full_path_and_ensure_dir('fut_limit_dir')

    def _get_trade_dates(self, start_date, end_date):
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SHFE.parquet')
        df_cal = pd.read_parquet(cal_file)
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= int(start_date)) & (df_cal['cal_date'] <= int(end_date))
        return df_cal[mask]['cal_date'].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                try:
                    df = pd.read_parquet(os.path.join(self.save_dir, file), columns=['trade_date'])
                    local_dates.update(df['trade_date'].astype(str).unique().tolist())
                except Exception as e:
                    self.logger.warning(f"读取本地文件 {file} 失败: {e}")
        return local_dates

    def sync(self, start_date='20090101', target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [期货每日涨跌停] (区间: {start_date} -> {target_end_date}) ===")
        
        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地期货涨跌停数据已完全覆盖目标区间，无需更新。")
            return
            
        self.logger.info(f"发现 {len(missing_dates)} 个交易日缺失，开始抓取...")

        # 按年分组批处理
        dates_by_year = {}
        for date in missing_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            self.logger.info(f"-> 正在处理 {year} 年缺失数据 (共 {len(dates)} 天)...")
            yearly_new_data = []
            
            for date in dates:
                try:
                    # 获取单日全市场涨跌停
                    df = self.pro.ft_limit(trade_date=date)
                    
                    if df is not None and not df.empty:
                        yearly_new_data.append(df)
                        
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取 {date} 期货涨跌停失败: {e}")
            
            if yearly_new_data:
                file_path = os.path.join(self.save_dir, f"{year}.parquet")
                df_new = pd.concat(yearly_new_data, ignore_index=True)
                
                # 读取老数据进行合并去重
                if os.path.exists(file_path):
                    df_old = pd.read_parquet(file_path)
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    df_combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                else:
                    df_combined = df_new
                    
                # ==========================================
                # 极致性能优化：强转 int32 和 float32
                # ==========================================
                if 'trade_date' in df_combined.columns:
                    df_combined['trade_date'] = df_combined['trade_date'].astype(np.int32)
                
                # 涨停价和跌停价转为 float32
                float_cols = ['up_limit', 'down_limit']
                for col in float_cols:
                    if col in df_combined.columns:
                        df_combined[col] = df_combined[col].astype(np.float32)

                df_combined.sort_values(by=['trade_date', 'ts_code'], inplace=True)
                df_combined.to_parquet(file_path, index=False)
                self.logger.info(f"✅ 成功保存 {year}.parquet，当前包含 {len(df_combined)} 条记录。")
                
        self.logger.info("=== [期货每日涨跌停] 同步结束 ===")