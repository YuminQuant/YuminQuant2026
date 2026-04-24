import os
import pandas as pd
from data_manager.core import BaseDownloader, ConfigManager

class SWMemberDownloader(BaseDownloader):
    """申万指数成分股下载器 (全量历史拉取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_member_sw', 500))
        self.page_limit = config['api']['page_limits'].get('index_member_sw', 2000)
        self.save_dir = self.get_full_path_and_ensure_dir('index_member_sw_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [申万指数历史成分股] ===")
        all_data = []
        offset = 0
        
        while True:
            try:
                # 核心：必须传入 is_new='N' 才能获取完整的历史进出记录，避免未来数据
                df = self.pro.index_member_all(is_new='N', limit=self.page_limit, offset=offset)
                
                if df is None or df.empty:
                    break
                    
                all_data.append(df)
                
                if len(df) < self.page_limit:
                    break
                    
                offset += self.page_limit
                self.safe_sleep()
                
                # 打印一下进度（按 offset 行数）
                self.logger.info(f"   已获取 {offset} 行数据...")
                
            except Exception as e:
                self.logger.error(f"拉取申万历史成分股失败: {e}")
                break
                
        # 合并落盘
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(inplace=True)
            
            # 按指数代码和进入时间排序，便于后续研究查看
            if 'index_code' in df_combined.columns and 'in_date' in df_combined.columns:
                df_combined.sort_values(by=['index_code', 'in_date'], inplace=True)
                
            file_path = os.path.join(self.save_dir, "sw_members.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存申万历史成分股全集，共 {len(df_combined)} 条记录。")


class CIMemberDownloader(BaseDownloader):
    """中信指数成分股下载器 (全量历史拉取)"""
    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config['api']['rate_limits'].get('index_member_ci', 500))
        self.page_limit = config['api']['page_limits'].get('index_member_ci', 4000)
        self.save_dir = self.get_full_path_and_ensure_dir('index_member_ci_dir')

    def sync(self):
        self.logger.info("=== 开始同步 [中信指数历史成分股] ===")
        all_data = []
        offset = 0
        
        while True:
            try:
                # 核心：必须传入 is_new='N' 以获取完整历史成分
                df = self.pro.ci_index_member(is_new='N', limit=self.page_limit, offset=offset)
                
                if df is None or df.empty:
                    break
                    
                all_data.append(df)
                
                if len(df) < self.page_limit:
                    break
                    
                offset += self.page_limit
                self.safe_sleep()
                
                self.logger.info(f"   已获取 {offset} 行数据...")
                
            except Exception as e:
                self.logger.error(f"拉取中信历史成分股失败: {e}")
                break
                
        # 合并落盘
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined.drop_duplicates(inplace=True)
            
            # 排序
            if 'index_code' in df_combined.columns and 'in_date' in df_combined.columns:
                df_combined.sort_values(by=['index_code', 'in_date'], inplace=True)
                
            file_path = os.path.join(self.save_dir, "ci_members.parquet")
            df_combined.to_parquet(file_path, index=False)
            self.logger.info(f"✅ 成功保存中信历史成分股全集，共 {len(df_combined)} 条记录。")