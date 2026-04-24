import os
import time
from abc import ABC, abstractmethod
from .tushare_client import TushareClient
from .config_manager import ConfigManager
from .logger import QuantLogger

class BaseDownloader(ABC):
    def __init__(self, rate_limit: int):
        self.pro = TushareClient().api
        self.config = ConfigManager().config
        self.logger = QuantLogger()  # 引入统一个日志模块
        
        # 计算安全休眠时间
        if rate_limit > 0:
            self.sleep_time = 60.0 / (rate_limit * 0.9)
        else:
            self.sleep_time = 0
            
        # 获取根路径
        self.base_data_dir = self.config['paths']['base_data_dir']

    def safe_sleep(self):
        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

    def get_full_path_and_ensure_dir(self, sub_dir_key):
        """
        根据配置文件中的 sub_dir 键名，拼接完整路径并确保文件夹存在
        """
        sub_dir = self.config['paths'].get(sub_dir_key, '')
        full_path = os.path.join(self.base_data_dir, sub_dir)
        
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            self.logger.info(f"创建目录: {full_path}")
            
        return full_path

    @abstractmethod
    def sync(self):
        pass