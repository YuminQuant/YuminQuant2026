import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.downloader.chn_stock.fin_statement_downloader import (
    QUARTER_SUFFIXES,
    _concat_preserve_schema,
    is_financial_statement_period,
)


MAINBZ_TYPES = ("P", "D", "I")
MAINBZ_VIP_PAGE_LIMIT = 10000
MAINBZ_FIELDS = (
    "ts_code,end_date,bz_item,bz_code,bz_sales,bz_profit,bz_cost,"
    "curr_type,update_flag"
)


class MainBusinessDownloader(BaseDownloader):
    """A-share main business downloader using fina_mainbz_vip by period and type."""

    def __init__(self):
        config = ConfigManager().config
        rate_limits = config.get("api", {}).get("rate_limits", {})
        page_limits = config.get("api", {}).get("page_limits", {})
        super().__init__(
            rate_limit=rate_limits.get("mainbz", rate_limits.get("financial_vip", 200))
        )
        self.page_limit = min(
            int(page_limits.get("mainbz", MAINBZ_VIP_PAGE_LIMIT)),
            MAINBZ_VIP_PAGE_LIMIT,
        )
        self.save_dir = self.get_full_path_and_ensure_dir("fin_mainbz_dir")
        self.task_name = "main business"
        self.fields = MAINBZ_FIELDS
        self.bz_types = MAINBZ_TYPES

    def _generate_periods(self, start_year, end_year):
        return [
            f"{year}{suffix}"
            for year in range(start_year, end_year + 1)
            for suffix in QUARTER_SUFFIXES
        ]

    def _fetch_period_type(self, period, bz_type, retry=3):
        if not is_financial_statement_period(period):
            self.logger.info(f"{self.task_name} skip non-quarter period: {period}")
            return pd.DataFrame()

        all_chunks = []
        offset = 0

        while True:
            df = None
            for attempt in range(retry):
                try:
                    df = self.pro.fina_mainbz_vip(
                        period=period,
                        type=bz_type,
                        fields=self.fields,
                        limit=self.page_limit,
                        offset=offset,
                    )
                    break
                except Exception as error:
                    if attempt == retry - 1:
                        self.logger.error(
                            f"{self.task_name} period={period} type={bz_type} "
                            f"offset={offset} failed: {error}"
                        )
                        return pd.DataFrame()
                    time.sleep(1)

            if df is None or df.empty:
                break

            df = df.copy()
            df["bz_type"] = bz_type
            all_chunks.append(df)

            if len(df) < self.page_limit:
                break
            offset += self.page_limit
            self.safe_sleep()

        if not all_chunks:
            return pd.DataFrame()
        return _concat_preserve_schema(all_chunks)

    def _fetch_period(self, period):
        period_chunks = []
        for bz_type in self.bz_types:
            self.logger.info(f"{self.task_name} fetch period={period} type={bz_type}")
            df = self._fetch_period_type(period, bz_type)
            if df is not None and not df.empty:
                period_chunks.append(df)
            self.safe_sleep()

        if not period_chunks:
            return pd.DataFrame()
        return _concat_preserve_schema(period_chunks)

    def _process_and_save(self, df_chunk):
        if df_chunk is None or df_chunk.empty:
            return
        if "end_date" not in df_chunk.columns:
            self.logger.warning(f"{self.task_name} response missing end_date; skip chunk")
            return

        df_chunk = df_chunk.copy()
        normalized = (
            df_chunk["end_date"].astype(str).str.replace("-", "", regex=False).str.strip()
        )
        df_chunk["end_date"] = (
            pd.to_numeric(normalized, errors="coerce").fillna(0).astype(np.int32)
        )
        df_chunk = df_chunk[df_chunk["end_date"] > 0]
        if df_chunk.empty:
            return

        if "bz_type" not in df_chunk.columns:
            df_chunk["bz_type"] = pd.NA
        df_chunk["end_year"] = (df_chunk["end_date"] // 10000).astype(str)

        if "update_flag" in df_chunk.columns:
            df_chunk["update_flag"] = (
                pd.to_numeric(df_chunk["update_flag"], errors="coerce")
                .fillna(0)
                .astype(np.int32)
            )

        safe_str_cols = {"ts_code", "bz_item", "bz_code", "curr_type", "bz_type", "end_year"}
        for col in df_chunk.select_dtypes(include=["object"]).columns:
            if col not in safe_str_cols:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce")

        float_cols = df_chunk.select_dtypes(include=["float64"]).columns
        if not float_cols.empty:
            df_chunk[float_cols] = df_chunk[float_cols].astype(np.float32)

        for year, df_year in df_chunk.groupby("end_year"):
            if not year or pd.isna(year):
                continue
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            df_save = df_year.drop(columns=["end_year"])
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_save = _concat_preserve_schema([df_old, df_save])

            subset_cols = [
                col
                for col in [
                    "ts_code",
                    "end_date",
                    "bz_type",
                    "bz_code",
                    "bz_item",
                    "curr_type",
                    "update_flag",
                ]
                if col in df_save.columns
            ]
            if subset_cols:
                df_save.drop_duplicates(subset=subset_cols, keep="last", inplace=True)

            sort_cols = [
                col
                for col in [
                    "ts_code",
                    "end_date",
                    "bz_type",
                    "bz_code",
                    "bz_item",
                    "curr_type",
                    "update_flag",
                ]
                if col in df_save.columns
            ]
            if sort_cols:
                df_save.sort_values(by=sort_cols, inplace=True)
            df_save.to_parquet(file_path, index=False)
            self.logger.info(f"{self.task_name} saved {len(df_save)} rows to {file_path}")

    def sync(self, mode="historical", start_year=2009, target_date=None):
        if mode == "historical":
            current_year = datetime.now(timezone(timedelta(hours=8))).year
            periods = self._generate_periods(start_year, current_year)
            self.logger.info(f"=== historical {self.task_name}: VIP period + type ===")
            for period in periods:
                df_period = self._fetch_period(period)
                self._process_and_save(df_period)
            self.logger.info(f"=== historical {self.task_name} complete ===")
            return

        if mode == "incremental":
            if target_date is None:
                target_date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
            if not is_financial_statement_period(target_date):
                self.logger.info(
                    f"{self.task_name} skip {target_date}: not a financial statement period"
                )
                return
            self.logger.info(f"=== incremental {self.task_name}: period={target_date} ===")
            df_period = self._fetch_period(target_date)
            self._process_and_save(df_period)
            self.logger.info(f"=== incremental {self.task_name} complete ===")
            return

        raise ValueError(f"unsupported sync mode: {mode}")
