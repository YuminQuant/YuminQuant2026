import concurrent.futures
import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import save_daily_dataframe


class IndexDailyDownloader(BaseDownloader):
    """Download and store index daily bars by index code and year."""

    DEFAULT_BROAD_BASE_INDEXES = [
        {"ts_code": "000300.SH", "list_date": "20050408"},
        {"ts_code": "000985.CSI", "list_date": "20110802"},
        {"ts_code": "000905.SH", "list_date": "20070115"},
        {"ts_code": "000852.SH", "list_date": "20141017"},
    ]
    REQUIRED_COLUMNS = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    FLOAT_COLUMNS = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("index_daily", 500))
        self.base_save_dir = self.get_full_path_and_ensure_dir("index_daily_dir")

    def sync(self, ts_code, start_date="19900101", target_end_date=None, list_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        if list_date:
            start_date = max(start_date, list_date)
        if start_date > target_end_date:
            self.logger.info(
                f"skip index_daily {ts_code}: start_date {start_date} > end_date {target_end_date}"
            )
            return

        self.logger.info(f"=== sync index_daily {ts_code}: {start_date} -> {target_end_date} ===")
        code_dir = os.path.join(self.base_save_dir, ts_code.replace(".", "_"))
        os.makedirs(code_dir, exist_ok=True)

        years = [str(year) for year in range(int(start_date[:4]), int(target_end_date[:4]) + 1)]
        for year in years:
            y_start = max(start_date, f"{year}0101")
            y_end = min(target_end_date, f"{year}1231")
            year_dir = os.path.join(code_dir, year)
            if (
                year < target_end_date[:4]
                and os.path.isdir(year_dir)
                and any(name.endswith(".parquet") for name in os.listdir(year_dir))
            ):
                continue

            try:
                df = self.pro.index_daily(ts_code=ts_code, start_date=y_start, end_date=y_end)
                self.safe_sleep()
                if df is None or df.empty:
                    continue
                written = self._save_year(code_dir, df)
                self.logger.info(f"-> index_daily {ts_code} {year} saved files={written}")
            except Exception as exc:
                self.logger.error(f"index_daily {ts_code} {year} failed: {exc}")

    def sync_many(self, specs=None, start_date="20090101", target_end_date=None):
        specs = specs or self.DEFAULT_BROAD_BASE_INDEXES
        for spec in specs:
            self.sync(
                spec["ts_code"],
                start_date=start_date,
                target_end_date=target_end_date,
                list_date=spec.get("list_date"),
            )

    def _save_year(self, code_dir, df):
        df = df.copy()
        for column in self.REQUIRED_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        df = df[self.REQUIRED_COLUMNS]

        df["trade_date"] = df["trade_date"].astype(np.int32)
        for column in self.FLOAT_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)
        return save_daily_dataframe(
            code_dir,
            df,
            key_cols=["trade_date"],
            sort_cols=["trade_date"],
            float32_cols=self.FLOAT_COLUMNS,
        )


class IndexWeightDownloader(BaseDownloader):
    """Download and store index weights by index code and year."""

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("index_weight", 500))
        self.base_save_dir = self.get_full_path_and_ensure_dir("index_weight_dir")

    def sync(self, index_code, start_date="20100101", target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")

        self.logger.info(f"=== sync index_weight {index_code}: {start_date} -> {target_end_date} ===")
        code_dir = os.path.join(self.base_save_dir, index_code.replace(".", "_"))
        os.makedirs(code_dir, exist_ok=True)

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(target_end_date)
        months = pd.date_range(start_dt.replace(day=1), end_dt, freq="MS")

        yearly_data = {}
        for month_start in months:
            m_start = month_start.strftime("%Y%m%d")
            next_month = month_start.replace(day=28) + timedelta(days=4)
            m_end = (next_month - timedelta(days=next_month.day)).strftime("%Y%m%d")
            year = month_start.strftime("%Y")
            file_path = os.path.join(code_dir, f"{year}.parquet")
            if year < target_end_date[:4] and os.path.exists(file_path):
                continue

            try:
                df = self.pro.index_weight(index_code=index_code, start_date=m_start, end_date=m_end)
                self.safe_sleep()
                if df is not None and not df.empty:
                    yearly_data.setdefault(year, []).append(df)
            except Exception as exc:
                self.logger.error(f"index_weight {index_code} {m_start} failed: {exc}")

        for year, frames in yearly_data.items():
            file_path = os.path.join(code_dir, f"{year}.parquet")
            df_new = pd.concat(frames, ignore_index=True)
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.drop_duplicates(subset=["con_code", "trade_date"], keep="last", inplace=True)

            if "trade_date" in df_new.columns:
                df_new["trade_date"] = df_new["trade_date"].astype(np.int32)
            if "weight" in df_new.columns:
                df_new["weight"] = pd.to_numeric(df_new["weight"], errors="coerce").astype(np.float32)

            df_new.sort_values(by=["trade_date", "con_code"], inplace=True)
            df_new.to_parquet(file_path, index=False)
            self.logger.info(f"-> index_weight {index_code} {year} saved rows={len(df_new)}")


class IndexMinuteDownloader(BaseDownloader):
    """Download and store 1-minute index bars by index code and year."""

    def __init__(self):
        config = ConfigManager().config
        super().__init__(
            rate_limit=config["api"]["rate_limits"].get(
                "idx_mins", config["api"]["rate_limits"].get("index_minute", 500)
            )
        )
        self.page_limit = config["api"]["page_limits"].get(
            "idx_mins", config["api"]["page_limits"].get("index_minute", 8000)
        )
        self.base_save_dir = self.get_full_path_and_ensure_dir("index_min1_dir")

        self.max_workers = 10
        self.api_lock = threading.Lock()
        self.last_call_time = 0.0
        self.min_interval = 60.0 / 450.0

    def _fetch_single_year(self, ts_code, year, start_time, end_time, file_path):
        yearly_chunks = []
        offset = 0

        while True:
            with self.api_lock:
                now = time.time()
                elapsed = now - self.last_call_time
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)
                self.last_call_time = time.time()

            try:
                df = self.pro.idx_mins(
                    ts_code=ts_code,
                    freq="1min",
                    start_date=start_time,
                    end_date=end_time,
                    limit=self.page_limit,
                    offset=offset,
                )
                if df is None or df.empty:
                    break
                yearly_chunks.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
            except Exception as exc:
                return False, f"API failed: {exc}"

        if not yearly_chunks:
            return True, "no data"

        try:
            df_new = pd.concat(yearly_chunks, ignore_index=True)
            df_new["trade_date"] = df_new["trade_time"].str[:10].str.replace("-", "").astype(np.int32)
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.drop_duplicates(subset=["trade_time"], keep="last", inplace=True)

            for column in ["open", "high", "low", "close", "vol", "amount"]:
                if column in df_new.columns:
                    df_new[column] = pd.to_numeric(df_new[column], errors="coerce").astype(np.float32)

            df_new.sort_values(by=["trade_time"], inplace=True)
            df_new.to_parquet(file_path, index=False)
            return True, f"saved rows={len(df_new)}"
        except Exception as exc:
            return False, f"save failed: {exc}"

    def sync(self, ts_code, start_date="20090101", target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")

        self.logger.info(f"=== sync index_minute {ts_code}: {start_date} -> {target_end_date} ===")
        code_dir = os.path.join(self.base_save_dir, ts_code.replace(".", "_"))
        os.makedirs(code_dir, exist_ok=True)

        tasks = []
        for year in range(int(start_date[:4]), int(target_end_date[:4]) + 1):
            year = str(year)
            file_path = os.path.join(code_dir, f"{year}.parquet")
            if year < target_end_date[:4] and os.path.exists(file_path):
                continue

            y_start = max(start_date, f"{year}0101")
            y_end = min(target_end_date, f"{year}1231")
            start_time = f"{y_start[:4]}-{y_start[4:6]}-{y_start[6:8]} 09:00:00"
            end_time = f"{y_end[:4]}-{y_end[4:6]}-{y_end[6:8]} 16:00:00"
            tasks.append((year, start_time, end_time, file_path))

        if not tasks:
            self.logger.info(f"{ts_code} index_minute is already up to date")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_year = {
                executor.submit(self._fetch_single_year, ts_code, year, start_time, end_time, file_path): year
                for year, start_time, end_time, file_path in tasks
            }
            for future in concurrent.futures.as_completed(future_to_year):
                year = future_to_year[future]
                try:
                    success, message = future.result()
                    if success:
                        self.logger.info(f"index_minute {ts_code} {year}: {message}")
                    else:
                        self.logger.error(f"index_minute {ts_code} {year}: {message}")
                except Exception as exc:
                    self.logger.error(f"index_minute {ts_code} {year} failed: {exc}")
