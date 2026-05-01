import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from tqdm import tqdm

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import list_local_daily_dates, save_daily_frames


class StDownloader(BaseDownloader):
    def __init__(self):
        config = ConfigManager().config
        rate_limit = config["api"]["rate_limits"].get("stock_st", 100)
        super().__init__(rate_limit=rate_limit)
        self.page_limit = config.get("api", {}).get("page_limits", {}).get("stock_st", 1000)
        self.save_dir = self.get_full_path_and_ensure_dir("st_list_dir")
        cal_sub_dir = self.config["paths"].get("calendar_dir", "calendar")
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, "trade_cal_SSE.parquet")

    def _get_trade_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"calendar file not found: {self.cal_file}")
        df_cal = pd.read_parquet(self.cal_file)
        mask = (
            (df_cal["is_open"] == 1)
            & (df_cal["cal_date"] >= int(start_date))
            & (df_cal["cal_date"] <= int(end_date))
        )
        return df_cal[mask]["cal_date"].astype(str).tolist()

    def _get_local_dates(self):
        return list_local_daily_dates(self.save_dir)

    def sync(self, start_date="20160101", target_end_date=None):
        start_date = str(start_date)
        if start_date < "20160101":
            self.logger.warning("stock_st API starts from 20160101; start_date has been adjusted")
            start_date = "20160101"
        if target_end_date is None:
            bj_tz = timezone(timedelta(hours=8))
            target_end_date = datetime.now(bj_tz).strftime("%Y%m%d")

        self.logger.info(f"=== sync stock ST list: {start_date} -> {target_end_date} ===")
        target_dates = self._get_trade_dates(start_date, target_end_date)
        missing_dates = sorted(set(target_dates) - self._get_local_dates())
        if not missing_dates:
            self.logger.info("stock ST list is already complete for target range")
            return

        self.logger.info(f"found {len(missing_dates)} missing trade dates")
        dates_by_year = {}
        for date in missing_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            daily_frames = []
            for date in tqdm(dates, desc=f"{year} stock_st", mininterval=10.0, ascii=True):
                try:
                    offset = 0
                    while True:
                        df_chunk = self.pro.stock_st(
                            trade_date=date,
                            limit=self.page_limit,
                            offset=offset,
                        )
                        if df_chunk is None or df_chunk.empty:
                            break
                        daily_frames.append(df_chunk)
                        if len(df_chunk) < self.page_limit:
                            break
                        offset += self.page_limit
                        self.safe_sleep()
                    self.safe_sleep()
                except Exception as exc:
                    self.logger.error(f"failed to fetch stock ST list {date}: {exc}")
            if daily_frames:
                self._save_yearly_data(year, daily_frames)

        self.logger.info("=== stock ST list sync complete ===")

    def _save_yearly_data(self, year, new_data_list):
        written = save_daily_frames(
            self.save_dir,
            new_data_list,
            key_cols=["ts_code", "trade_date"],
            sort_cols=["trade_date", "ts_code"],
        )
        self.logger.info(f"saved {written} daily parquet files for {year}")
