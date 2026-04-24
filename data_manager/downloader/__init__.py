from .calendar_downloader import CalendarDownloader
from .chn_stock import *
from .etf import *
from .index import * # 新增这一行
from .future import *
from .option import *
from .hkg_stock import *
from .usa_stock import *
from .alternative import *

from . import chn_stock
from . import etf
from . import index   # 新增这一行
from . import future
from . import option
from . import hkg_stock
from . import usa_stock
from . import alternative

__all__ = (
    ['CalendarDownloader'] + 
    getattr(chn_stock, '__all__', []) + 
    getattr(etf, '__all__', []) + 
    getattr(index, '__all__', []) +
    getattr(future, '__all__', []) +
    getattr(option, '__all__', []) + 
    getattr(hkg_stock, '__all__', []) +
    getattr(usa_stock, '__all__', []) + 
    getattr(alternative, '__all__', [])
)