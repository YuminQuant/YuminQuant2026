# data_manager/__init__.py

from .core import QuantLogger, ConfigManager, BaseDownloader
from .downloader import * # 这里会自动拿到 CalendarDownloader 和所有的 StockXXXDownloader