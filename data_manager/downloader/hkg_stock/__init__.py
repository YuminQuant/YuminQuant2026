# data_manager/downloader/stock/__init__.py

from .hk_basic_downloader import HKBasicDownloader, HKCalendarDownloader
from .hk_daily_downloader import HKDailyDownloader,HKAdjFactorDownloader
from .hk_minute_downloader import HKMinuteDownloader
# 2. 新增的 VIP 财务数据下载器
from .hk_financial_downloader import (
    HKBalanceSheetDownloader,
    HKCashFlowDownloader,
    HKIncomeDownloader
)

__all__ = [
    'HKBasicDownloader',
    'HKCalendarDownloader',
    'HKDailyDownloader',
    'HKAdjFactorDownloader',
    'HKMinuteDownloader',
    'HKBalanceSheetDownloader',
    'HKCashFlowDownloader',
    'HKIncomeDownloader'
]
