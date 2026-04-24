import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class IndexBasicDownloader(BaseDownloader):
    """指数基础信息下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_basic', 500))
        self.save_dir = self.get_full_path_and_ensure_dir('index_basic_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [指数基础信息] ===")
        # 市场代码：MSCI, CSI(中证), SSE(上交所), SZSE(深交所), CICC(中金), SW(申万), OTH(其他)
        markets = ['MSCI', 'CSI', 'SSE', 'SZSE', 'CICC', 'SW', 'OTH']
        all_data = []
        for market in markets:
            try:
                df = self.pro.index_basic(market=market)
                if df is not None and not df.empty:
                    all_data.append(df)
                self.safe_sleep()
            except Exception as e:
                self.logger.error(f"拉取 {market} 指数基础信息失败: {e}")
                
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(subset=['ts_code'], inplace=True)
            file_path = os.path.join(self.save_dir, "index_basic.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存 指数基础信息，共 {len(df_combined)} 条。")

class IndexClassifyDownloader(BaseDownloader):
    """指数分类细则下载器"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_classify', 500))
        self.save_dir = self.get_full_path_and_ensure_dir('index_classify_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [指数分类细则] ===")
        levels = ['L1', 'L2', 'L3']
        srcs = ['SW2021', 'SW2014']
        
        for src in srcs:
            for level in levels:
                try:
                    df = self.pro.index_classify(level=level, src=src)
                    if df is not None and not df.empty:
                        file_path = os.path.join(self.save_dir, f"classify_{src}_{level}.parquet")
                        df.to_parquet(file_path, index=False)
                        self.logger.info(f"✅ 成功保存分类 {src}-{level}，共 {len(df)} 条。")
                    self.safe_sleep()
                except Exception as e:
                    self.logger.error(f"拉取分类 {src}-{level} 失败: {e}")