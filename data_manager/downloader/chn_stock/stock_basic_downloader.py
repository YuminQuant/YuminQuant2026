# data_manager/downloaders/stock_basic_downloader.py

import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class StockBasicDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        rate_limit = config['api']['rate_limits'].get('stock_basic', 200)
        super().__init__(rate_limit=rate_limit)
        
        # 获取路径并确保文件夹存在
        self.save_dir = self.get_full_path_and_ensure_dir('stock_info_dir')
        self.file_path = os.path.join(self.save_dir, 'stock_basic.parquet')

    def sync(self):
        self.logger.info("=== 开始同步 [股票基础信息] ===")
        
        # L:上市, D:退市, P:暂停上市
        statuses = ['L', 'D']
        all_data = []
        
        for status in statuses:
            try:
                self.logger.info(f"正在拉取状态为 '{status}' 的股票列表...")
                # fields 参数可以显式指定我们需要的最全字段
                df = self.pro.stock_basic(
                    exchange='', 
                    list_status=status, 
                    fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs'
                )
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    self.logger.info(f"状态 '{status}' 拉取成功，共 {len(df)} 只股票。")
                
                self.safe_sleep() # 触发限流保护
                
            except Exception as e:
                self.logger.error(f"拉取状态 '{status}' 的股票信息失败: {e}")
                
        # 合并并全量覆盖写入
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # 按照股票代码排序，保证每次保存的数据物理顺序一致
            final_df.sort_values(by=['ts_code'], inplace=True)
            
            # 全量覆写 Parquet
            final_df.to_parquet(self.file_path, index=False)
            self.logger.info(f"同步完成！全市场共计 {len(final_df)} 只股票，已覆写至 {self.file_path}")
        else:
            self.logger.warning("未获取到任何股票基础信息！")
            
        self.logger.info("=== [股票基础信息] 同步结束 ===")