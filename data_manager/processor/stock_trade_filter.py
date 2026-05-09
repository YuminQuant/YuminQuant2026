import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_manager.core import ConfigManager, QuantLogger
from data_manager.core.daily_storage import daily_file_path, save_daily_dataframe


class StockTradeFilterBuilder:
    """Build daily tradeability flags from close/limit/ST source tables."""

    ST_EFFECTIVE_START = "20160101"

    def __init__(self):
        self.config = ConfigManager().config
        self.logger = QuantLogger()
        self.base_data_dir = self.config["paths"]["base_data_dir"]
        paths = self.config["paths"]
        self.pv_dir = self._full_path(paths.get("stock_daily_pv_dir", "stock_data/daily/pv"))
        self.limit_dir = self._full_path(
            paths.get("stock_daily_limit_dir", "stock_data/daily/limit")
        )
        self.st_dir = self._full_path(paths.get("st_list_dir", "stock_data/daily/st_list"))
        self.save_dir = self._full_path(
            paths.get("stock_trade_filter_dir", "stock_data/daily/trade_filter")
        )
        self.cal_file = os.path.join(
            self.base_data_dir,
            paths.get("calendar_dir", "calendar"),
            "trade_cal_SSE.parquet",
        )

    def _full_path(self, value):
        return value if os.path.isabs(value) else os.path.join(self.base_data_dir, value)

    def _default_end_date(self):
        bj_tz = timezone(timedelta(hours=8))
        return datetime.now(bj_tz).strftime("%Y%m%d")

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"calendar file not found: {self.cal_file}")
        df_cal = pd.read_parquet(self.cal_file, columns=["cal_date", "is_open"])
        mask = (
            (df_cal["is_open"] == 1)
            & (df_cal["cal_date"] >= int(start_date))
            & (df_cal["cal_date"] <= int(end_date))
        )
        return df_cal.loc[mask, "cal_date"].astype(str).tolist()

    def sync(self, start_date="20090101", target_end_date=None):
        target_end_date = target_end_date or self._default_end_date()
        self.logger.info(
            f"=== build stock trade filter: {start_date} -> {target_end_date} ==="
        )
        dates = self._get_trade_dates(str(start_date), str(target_end_date))
        written = 0
        for date in tqdm(dates, desc="stock_trade_filter", mininterval=10.0, ascii=True):
            try:
                written += self._build_one_day(date)
            except Exception as exc:
                self.logger.error(f"failed to build stock trade filter {date}: {exc}")
        self.logger.info(f"=== stock trade filter complete: wrote {written} files ===")

    def _build_one_day(self, date):
        pv_path = daily_file_path(self.pv_dir, date)
        limit_path = daily_file_path(self.limit_dir, date)
        if not os.path.exists(pv_path):
            self.logger.warning(f"skip trade filter {date}: missing daily pv {pv_path}")
            return 0
        if not os.path.exists(limit_path):
            self.logger.warning(f"skip trade filter {date}: missing daily limit {limit_path}")
            return 0

        pv = pd.read_parquet(pv_path, columns=["trade_date", "ts_code", "close"])
        limit = pd.read_parquet(limit_path, columns=["trade_date", "ts_code", "up_limit", "down_limit"])
        df = pv.merge(limit, on=["trade_date", "ts_code"], how="left")

        close = pd.to_numeric(df["close"], errors="coerce").round(2)
        up_limit = pd.to_numeric(df["up_limit"], errors="coerce").round(2)
        down_limit = pd.to_numeric(df["down_limit"], errors="coerce").round(2)
        df["is_limit_up"] = (close.notna() & up_limit.notna() & (close >= up_limit)).astype(bool)
        df["is_limit_down"] = (
            close.notna() & down_limit.notna() & (close <= down_limit)
        ).astype(bool)
        df["is_limit"] = df["is_limit_up"] | df["is_limit_down"]

        st_codes = self._load_st_codes(date)
        df["is_st"] = df["ts_code"].isin(st_codes)

        output = df[
            ["trade_date", "ts_code", "is_limit_up", "is_limit_down", "is_limit", "is_st"]
        ].copy()
        output["trade_date"] = output["trade_date"].astype(np.int32)
        return save_daily_dataframe(
            self.save_dir,
            output,
            key_cols=["ts_code", "trade_date"],
            sort_cols=["trade_date", "ts_code"],
            overwrite=True,
        )

    def _load_st_codes(self, date):
        if str(date) < self.ST_EFFECTIVE_START:
            return set()
        st_path = daily_file_path(self.st_dir, date)
        if not os.path.exists(st_path):
            self.logger.warning(f"ST list missing for {date}; treating as empty")
            return set()
        st = pd.read_parquet(st_path, columns=["ts_code"])
        return set(st["ts_code"].dropna().astype(str))
