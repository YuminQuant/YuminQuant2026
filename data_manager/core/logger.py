import os
import logging
from .config_manager import ConfigManager

class QuantLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantLogger, cls).__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        config = ConfigManager().config
        base_dir = config['paths']['base_data_dir']
        
        # 确保根目录存在，日志文件存放在 base_data_dir 下
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        self.log_file = os.path.join(base_dir, 'data_sync.log')

        # 核心逻辑：检查大小，如果超过 128KB (128 * 1024 bytes)，则清空文件
        if os.path.exists(self.log_file):
            file_size_kb = os.path.getsize(self.log_file) / 1024
            if file_size_kb > 128:
                with open(self.log_file, 'w') as f:
                    f.truncate() # 清空文件内容

        self.logger = logging.getLogger("QuantDataLogger")
        self.logger.setLevel(logging.INFO)

        # 避免重复添加 Handler
        if not self.logger.handlers:
            # 文件输出
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            # 控制台输出
            ch = logging.StreamHandler()
            
            # 定义日志格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)
        
    def warning(self, message):
        self.logger.warning(message)