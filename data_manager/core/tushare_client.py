import tushare as ts
from .config_manager import ConfigManager  # 【修改后】使用相对路径导入同一个文件夹下的模块

class TushareClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TushareClient, cls).__new__(cls)
            config = ConfigManager().config
            ts.set_token(config['api']['tushare_token'])
            cls._instance.pro = ts.pro_api()
        return cls._instance

    @property
    def api(self):
        return self.pro