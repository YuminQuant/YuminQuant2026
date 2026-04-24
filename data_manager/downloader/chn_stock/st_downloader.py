# data_manager/downloaders/st_downloader.py

import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from data_manager.core import BaseDownloader, ConfigManager
from tqdm import tqdm
class StDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        
        # 1. 初始化频控限制
        rate_limit = config['api']['rate_limits'].get('stock_st', 100)
        super().__init__(rate_limit=rate_limit)
        
        # 2. 从配置中读取单次最大返回条数 (limit)，默认给个安全值 1000
        self.page_limit = config.get('api', {}).get('page_limits', {}).get('stock_st', 1000)
        
        # 3. 路径与日历依赖配置
        self.save_dir = self.get_full_path_and_ensure_dir('st_list_dir')
        cal_sub_dir = self.config['paths'].get('calendar_dir', 'calendar')
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, 'trade_cal_SSE.parquet')

    def _get_trade_dates(self, start_date, end_date):
        """从本地日历读取处于开市状态的交易日"""
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"未找到日历文件 {self.cal_file}，请先运行日历同步脚本！")
            
        df_cal = pd.read_parquet(self.cal_file)
        # 筛选开市日，并且日期在指定区间内
        mask = (df_cal['is_open'] == 1) & (df_cal['cal_date'] >= start_date) & (df_cal['cal_date'] <= end_date)
        return df_cal[mask]['cal_date'].tolist()

    def _get_local_dates(self):
        """扫描本地已存的数据，找出所有已经下载过的日期"""
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if file.endswith('.parquet'):
                file_path = os.path.join(self.save_dir, file)
                try:
                    df = pd.read_parquet(file_path, columns=['trade_date'])
                    local_dates.update(df['trade_date'].unique().tolist())
                except Exception as e:
                    self.logger.warning(f"读取本地文件 {file} 失败: {e}")
        return local_dates

    def sync(self, start_date='20160101', target_end_date=None):
        """执行同步任务"""
        # 接口最早支持到 20160101
        if start_date < '20160101':
            self.logger.warning("ST 接口数据最早始于 20160101，已自动修正 start_date。")
            start_date = '20160101'
            
        if target_end_date is None:
            # 获取北京时间当前日期作为默认截止日
            bj_tz = timezone(timedelta(hours=8))
            target_end_date = datetime.now(bj_tz).strftime('%Y%m%d')

        self.logger.info(f"=== 开始同步 [ST 股票列表] (区间: {start_date} -> {target_end_date}) ===")

        # 1. 结合交易日历计算缺失的交易日
        target_dates = self._get_trade_dates(start_date, target_end_date)
        local_dates = self._get_local_dates()
        missing_dates = sorted(list(set(target_dates) - set(local_dates)))
        
        if not missing_dates:
            self.logger.info("本地 ST 数据已完全覆盖目标区间，无需更新。")
            return
            
        self.logger.info(f"发现 {len(missing_dates)} 个交易日的数据缺失，开始抓取...")

        # 2. 按年份对缺失日期进行分组，减少文件 I/O 次数
        dates_by_year = {}
        for date in missing_dates:
            year = date[:4]
            if year not in dates_by_year:
                dates_by_year[year] = []
            dates_by_year[year].append(date)

        # 3. 逐年进行下载和覆写
        for year, dates in dates_by_year.items():
            self.logger.info(f"-> 正在处理 {year} 年缺失数据 (共 {len(dates)} 天)...")
            yearly_new_data = []
            
            for date in tqdm(dates):
                try:
                    offset = 0
                    
                    # 引入分页提取模式 (Pagination)
                    while True:
                        # 使用配置项中的 self.page_limit
                        df_chunk = self.pro.stock_st(trade_date=date, limit=self.page_limit, offset=offset)
                        
                        # 如果没有数据，结束当前日期的翻页
                        if df_chunk is None or df_chunk.empty:
                            break
                            
                        yearly_new_data.append(df_chunk)
                        
                        # 如果获取的数据少于 limit，说明到底了，结束翻页
                        if len(df_chunk) < self.page_limit:
                            break
                            
                        # 还没到底，游标推进，进入下一页
                        offset += self.page_limit
                        self.logger.info(f"[{date}] 触发分页，正在获取第 {offset//self.page_limit + 1} 页...")
                        
                        # self.safe_sleep() # 翻页同样需要被限流保护
                        
                    self.safe_sleep() # 完成一天的请求后限流休眠
                    
                except Exception as e:
                    self.logger.error(f"拉取 {date} 的 ST 数据失败: {e}")
            
            # 4. 把这一年新下到的所有数据拿去合并落盘
            if yearly_new_data:
                self._save_yearly_data(year, yearly_new_data)
                
        self.logger.info("=== [ST 股票列表] 同步结束 ===")

    def _save_yearly_data(self, year, new_data_list):
        """负责将当年新下载的数据与本地原有的当年数据进行合并和覆写"""
        file_path = os.path.join(self.save_dir, f"{year}.parquet")
        
        # 将本次内存中收集到的该年数据拼成一个大 DataFrame
        df_new = pd.concat(new_data_list, ignore_index=True)
        
        # 读取本地历史数据进行合并
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            # 严格去重，确保数据纯净
            df_combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
        else:
            df_combined = df_new
            
        # 物理排序：先按日期，再按股票代码，让最终保存的数据极其规整
        df_combined.sort_values(by=['trade_date', 'ts_code'], inplace=True)
        df_combined.to_parquet(file_path, index=False)
        
        self.logger.info(f"成功更新并保存 {year}.parquet (当前文件共 {len(df_combined)} 条记录)")