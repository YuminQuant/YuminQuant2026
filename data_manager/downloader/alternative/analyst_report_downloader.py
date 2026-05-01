import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_manager.core import BaseDownloader, ConfigManager


REPORT_RC_FIELDS = [
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "rating",
    "max_price",
    "min_price",
    "create_time",
    "imp_dg",
]

STRING_COLUMNS = {
    "ts_code",
    "name",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "rating",
    "create_time",
}

FLOAT_COLUMNS = [
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "max_price",
    "min_price",
    "imp_dg",
]

DEDUP_COLUMNS = [
    "ts_code",
    "report_date",
    "org_name",
    "author_name",
    "quarter",
    "report_title",
    "create_time",
]


class AnalystReportDownloader(BaseDownloader):
    """Download and store analyst report consensus data from report_rc."""

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("report_rc", 200))
        self.page_limit = config["api"]["page_limits"].get("report_rc", 5000)
        self.save_dir = self.get_full_path_and_ensure_dir("analyst_report_dir")

        cal_sub_dir = self.config["paths"].get("calendar_dir", "chn_stock_data/calendar")
        self.cal_file = os.path.join(self.base_data_dir, cal_sub_dir, "trade_cal_SSE.parquet")

    def _get_calendar_dates(self, start_date, end_date):
        if not os.path.exists(self.cal_file):
            raise FileNotFoundError(f"calendar file not found: {self.cal_file}")

        df_cal = pd.read_parquet(self.cal_file)
        start_int = int(start_date)
        end_int = int(end_date)
        mask = (df_cal["cal_date"] >= start_int) & (df_cal["cal_date"] <= end_int)
        return df_cal[mask]["cal_date"].astype(str).tolist()

    def _get_local_dates(self):
        local_dates = set()
        for file in os.listdir(self.save_dir):
            if not file.endswith(".parquet"):
                continue
            try:
                df = pd.read_parquet(os.path.join(self.save_dir, file), columns=["report_date"])
                local_dates.update(df["report_date"].astype(str).unique().tolist())
            except Exception as exc:
                self.logger.warning(f"failed reading local analyst report dates from {file}: {exc}")
        return local_dates

    def _fetch_date(self, date):
        chunks = []
        offset = 0
        while True:
            df = self.pro.report_rc(
                report_date=date,
                fields=REPORT_RC_FIELDS,
                limit=self.page_limit,
                offset=offset,
            )
            if df is None or df.empty:
                break
            chunks.append(self._normalize_schema(df))
            if len(df) < self.page_limit:
                break
            offset += self.page_limit
            self.safe_sleep()
        if not chunks:
            return None
        return pd.concat(chunks, ignore_index=True)

    def _normalize_schema(self, df):
        df = df.copy()
        for column in REPORT_RC_FIELDS:
            if column not in df.columns:
                df[column] = np.nan
        df = df[REPORT_RC_FIELDS]

        df["report_date"] = (
            pd.to_numeric(
                df["report_date"].astype(str).str.replace("-", "", regex=False),
                errors="coerce",
            )
            .fillna(0)
            .astype(np.int32)
        )

        for column in STRING_COLUMNS:
            if column in df.columns:
                df[column] = df[column].astype("string")

        for column in FLOAT_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype(np.float32)

        return df

    def _save_year(self, year, df_new):
        file_path = os.path.join(self.save_dir, f"{year}.parquet")
        df_new = self._normalize_schema(df_new)
        if os.path.exists(file_path):
            df_old = pd.read_parquet(file_path)
            df_old = self._normalize_schema(df_old)
            df_new = pd.concat([df_old, df_new], ignore_index=True)

        df_new.drop_duplicates(subset=DEDUP_COLUMNS, keep="last", inplace=True)
        df_new.sort_values(by=["report_date", "ts_code", "org_name"], inplace=True)
        df_new.reset_index(drop=True, inplace=True)
        df_new.to_parquet(file_path, index=False)
        self.logger.info(f"analyst_report {year} saved rows={len(df_new)}")

    def sync(self, start_date="20100101", target_end_date=None):
        if target_end_date is None:
            target_end_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")

        self.logger.info(f"=== sync analyst_report report_rc: {start_date} -> {target_end_date} ===")
        all_calendar_dates = self._get_calendar_dates(start_date, target_end_date)
        if not all_calendar_dates:
            self.logger.warning("no calendar dates found for analyst_report sync")
            return

        local_dates = self._get_local_dates()
        missing_dates = sorted(set(all_calendar_dates) - local_dates)
        if not missing_dates:
            self.logger.info("analyst_report is already up to date")
            return

        dates_by_year = {}
        for date in missing_dates:
            dates_by_year.setdefault(date[:4], []).append(date)

        for year, dates in dates_by_year.items():
            yearly_chunks = []
            self.logger.info(f"analyst_report {year}: fetching {len(dates)} dates")
            for date in tqdm(dates):
                try:
                    df = self._fetch_date(date)
                    if df is not None and not df.empty:
                        yearly_chunks.append(df)
                except Exception as exc:
                    self.logger.error(f"analyst_report {date} failed: {exc}")

            if yearly_chunks:
                self._save_year(year, pd.concat(yearly_chunks, ignore_index=True))

        self.logger.info("=== analyst_report sync complete ===")
