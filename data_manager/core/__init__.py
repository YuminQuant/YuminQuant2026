# core/__init__.py

from .config_manager import ConfigManager
from .tushare_client import TushareClient
from .base_downloader import BaseDownloader
from .logger import QuantLogger

# 使用 __all__ 明确声明对外暴露的接口（良好的工程习惯）
__all__ = [
    'ConfigManager',
    'TushareClient',
    'BaseDownloader',
    'QuantLogger'
]