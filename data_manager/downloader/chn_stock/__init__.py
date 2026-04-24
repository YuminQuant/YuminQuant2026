# data_manager/downloader/stock/__init__.py

from .stock_daily_pv_downloader import StockDailyPVDownloader
from .stock_basic_downloader import StockBasicDownloader
from .stock_adj_factor_downloader import StockAdjFactorDownloader
from .stock_daily_limit_downloader import StockDailyLimitDownloader
from .stock_daily_basic_downloader import StockDailyBasicDownloader
from .stock_suspend_downloader import StockSuspendDownloader
from .stock_moneyflow_downloader import StockMoneyflowDownloader
from .stock_minute_downloader import StockMinuteDownloader
from .st_downloader import StDownloader
# 2. 新增的 VIP 财务数据下载器
from .fin_statement_downloader import (
    IncomeDownloader,
    BalanceSheetDownloader,
    CashFlowDownloader,
    ForecastDownloader,
    ExpressDownloader
)

# 3. 新增的分红送股下载器
from .fin_dividend_downloader import DividendDownloader

__all__ = [
    'StockBasicDownloader',
    'StockDailyPVDownloader',
    'StockAdjFactorDownloader',
    'StockDailyLimitDownloader',
    'StockDailyBasicDownloader',
    'StockSuspendDownloader',
    'StockMoneyflowDownloader',
    'StockMinuteDownloader',
    'StDownloader',
    # 财务与分红
    'IncomeDownloader',
    'BalanceSheetDownloader',
    'CashFlowDownloader',
    'ForecastDownloader',
    'ExpressDownloader',
    'DividendDownloader'
]
