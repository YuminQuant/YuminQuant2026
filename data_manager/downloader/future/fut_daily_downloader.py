import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import list_local_daily_dates, save_daily_frames


class FutureDailyDownloader(BaseDownloader):
    """Download futures daily bars by trade date and store one parquet per day."""

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("fut_daily", 500))
        self.save_dir = self.get_full_path_and_ensure_dir("fut_daily_dir")

    def _get_trade_dates(self, start_date, end_date):
        cal_sub_dir = self.config["paths"].get("calendar_dir", "calendar")
        cal_file = os.path.join(self.base_data_dir, cal_sub_dir, "trade_cal_SHFE.parquet")
        df_cal = pd.read_parquet(cal_file)
        mask = (
            (df_cal["is_open"] == 1)
            & (df_cal["cal_date"] >= int(start_date))
            & (df_cal["cal_date"] <= int(end_date))
        )
        return df_cal[mask]["cal_date"].astype(str).tolist()

    def _get_local_dates(self):
        return list_local_daily_dates(self.save_dir)

    def sync(self, start_date="20090101", target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")

        self.logger.info(f"=== sync future daily: {start_date} -> {target_end_date} ===")
        missing_dates = sorted(
            set(self._get_trade_dates(start_date, target_end_date)) - self._get_local_dates()
        )
        if not missing_dates:
            self.logger.info("future daily is already complete for target range")
            return

        dates_by_year = {}
        for date in missing_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            daily_frames = []
            for date in tqdm(dates, desc=f"{year} future_daily", mininterval=10.0, ascii=True):
                try:
                    df = self.pro.fut_daily(trade_date=date)
                    if df is not None and not df.empty:
                        df = df[df["ts_code"].str.contains(r"\d")]
                        daily_frames.append(df)
                    self.safe_sleep()
                except Exception as exc:
                    self.logger.error(f"failed to fetch future daily {date}: {exc}")
            if daily_frames:
                self._save_yearly_data(year, daily_frames)

        self.logger.info("=== future daily sync complete ===")

    def _save_yearly_data(self, year, new_data_list):
        written = save_daily_frames(
            self.save_dir,
            new_data_list,
            key_cols=["ts_code", "trade_date"],
            sort_cols=["trade_date", "ts_code"],
            float32_cols=[
                "open",
                "high",
                "low",
                "close",
                "settle",
                "change1",
                "change2",
                "delv_settle",
                "pre_settle",
                "pre_close",
            ],
        )
        self.logger.info(f"saved {written} daily parquet files for {year}")
