import tomli # 或 import tomllib (Python 3.11+)
import os

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            
            # 精确计算绝对路径，避免由于运行路径不同导致的找不到文件问题
            # 当前 __file__ = YuminQuant/data_manager/core/config_manager.py
            core_dir = os.path.dirname(__file__)                   # 指向 core/
            data_manager_dir = os.path.dirname(core_dir)           # 指向 data_manager/
            project_root = os.path.dirname(data_manager_dir)       # 指向 YuminQuant/
            
            config_path = os.path.join(project_root, 'config.toml')
            
            try:
                with open(config_path, "rb") as f:
                    cls._instance._config = tomli.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"配置文件未找到，请确保文件存在: {config_path}")
                
        return cls._instance

    @property
    def config(self):
        return self._config