import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager


QUARTER_SUFFIXES = ("0331", "0630", "0930", "1231")
REPORT_TYPES = tuple(range(1, 13))


def _nullable_concat_dtype(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return pd.Int32Dtype() if str(dtype) == "int32" else pd.Int64Dtype()
    if pd.api.types.is_bool_dtype(dtype):
        return pd.BooleanDtype()
    return dtype


def _concat_preserve_schema(frames):
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()

    columns = []
    seen = set()
    for df in frames:
        for col in df.columns:
            if col not in seen:
                columns.append(col)
                seen.add(col)

    dtypes = {}
    for col in columns:
        dtype = None
        for df in frames:
            if col in df.columns and df[col].notna().any():
                dtype = df[col].dtype
                break
        dtypes[col] = _nullable_concat_dtype(dtype) if dtype is not None else object

    aligned = []
    for df in frames:
        current = df.copy()
        for col in columns:
            if col not in current.columns:
                current[col] = pd.Series(pd.NA, index=current.index, dtype=dtypes[col])
        current = current.reindex(columns=columns)
        for col, dtype in dtypes.items():
            if current[col].dtype == dtype:
                continue
            try:
                current[col] = current[col].astype(dtype)
            except (TypeError, ValueError):
                current[col] = current[col].astype(object)
        aligned.append(current)

    return pd.concat(aligned, ignore_index=True)


def is_financial_statement_period(value: str) -> bool:
    value = str(value)
    return len(value) == 8 and value[:4].isdigit() and value[4:] in QUARTER_SUFFIXES


class BaseFinancialStatementDownloader(BaseDownloader):
    """A-share statement downloader using VIP period + report_type endpoints."""

    def __init__(self, vip_api_name, dir_config_key, task_name, report_types=REPORT_TYPES):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("financial_vip", 200))
        self.page_limit = config["api"]["page_limits"].get("financial_vip", 5000)
        self.save_dir = self.get_full_path_and_ensure_dir(dir_config_key)
        self.vip_api_name = vip_api_name
        self.task_name = task_name
        self.report_types = tuple(report_types)

    def _generate_periods(self, start_year, end_year):
        return [
            f"{year}{suffix}"
            for year in range(start_year, end_year + 1)
            for suffix in QUARTER_SUFFIXES
        ]

    def _fetch_period_report_type(self, period, report_type, retry=3):
        api_func = getattr(self.pro, self.vip_api_name)
        all_chunks = []
        offset = 0

        while True:
            df = None
            for attempt in range(retry):
                try:
                    df = api_func(
                        period=period,
                        report_type=report_type,
                        limit=self.page_limit,
                        offset=offset,
                    )
                    break
                except Exception as error:
                    if attempt == retry - 1:
                        self.logger.error(
                            f"{self.task_name} period={period} report_type={report_type} "
                            f"offset={offset} failed: {error}"
                        )
                        return pd.DataFrame()
                    time.sleep(1)

            if df is None or df.empty:
                break

            if "report_type" not in df.columns:
                df["report_type"] = report_type
            all_chunks.append(df)

            if len(df) < self.page_limit:
                break
            offset += self.page_limit
            self.safe_sleep()

        if not all_chunks:
            return pd.DataFrame()
        return _concat_preserve_schema(all_chunks)

    def _fetch_period(self, period):
        if not is_financial_statement_period(period):
            self.logger.info(f"{self.task_name} skip non-quarter period: {period}")
            return pd.DataFrame()

        period_chunks = []
        for report_type in self.report_types:
            self.logger.info(
                f"{self.task_name} fetch period={period} report_type={report_type}"
            )
            df = self._fetch_period_report_type(period, report_type)
            if df is not None and not df.empty:
                period_chunks.append(df)
            self.safe_sleep()

        if not period_chunks:
            return pd.DataFrame()
        return _concat_preserve_schema(period_chunks)

    def _process_and_save(self, df_chunk):
        if df_chunk is None or df_chunk.empty:
            return
        if "ann_date" not in df_chunk.columns:
            self.logger.warning(f"{self.task_name} response missing ann_date; skip chunk")
            return

        df_chunk = df_chunk.copy()
        for date_col in ["ann_date", "f_ann_date", "end_date"]:
            if date_col in df_chunk.columns:
                normalized = (
                    df_chunk[date_col]
                    .astype(str)
                    .str.replace("-", "", regex=False)
                    .str.strip()
                )
                df_chunk[date_col] = (
                    pd.to_numeric(normalized, errors="coerce").fillna(0).astype(np.int32)
                )

        df_chunk = df_chunk[df_chunk["ann_date"] > 0]
        if df_chunk.empty:
            return

        df_chunk["ann_year"] = (df_chunk["ann_date"] // 10000).astype(str)

        for int_col in ["report_type", "update_flag"]:
            if int_col in df_chunk.columns:
                df_chunk[int_col] = (
                    pd.to_numeric(df_chunk[int_col], errors="coerce")
                    .fillna(0)
                    .astype(np.int32)
                )

        obj_cols = df_chunk.select_dtypes(include=["object"]).columns
        safe_str_cols = ["ts_code", "ann_year"]
        for col in obj_cols:
            if col not in safe_str_cols:
                df_chunk[col] = pd.to_numeric(df_chunk[col], errors="coerce")

        float_cols = df_chunk.select_dtypes(include=["float64"]).columns
        if not float_cols.empty:
            df_chunk[float_cols] = df_chunk[float_cols].astype(np.float32)

        for year, df_year in df_chunk.groupby("ann_year"):
            if not year or pd.isna(year):
                continue

            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            df_save = df_year.drop(columns=["ann_year"])
            if os.path.exists(file_path):
                df_old = pd.read_parquet(file_path)
                df_save = _concat_preserve_schema([df_old, df_save])

            subset_cols = [
                col
                for col in [
                    "ts_code",
                    "end_date",
                    "report_type",
                    "f_ann_date",
                    "ann_date",
                ]
                if col in df_save.columns
            ]
            if subset_cols:
                df_save.drop_duplicates(subset=subset_cols, keep="last", inplace=True)

            sort_cols = [
                col
                for col in ["ts_code", "ann_date", "end_date", "report_type"]
                if col in df_save.columns
            ]
            if sort_cols:
                df_save.sort_values(by=sort_cols, inplace=True)
            df_save.to_parquet(file_path, index=False)
            self.logger.info(
                f"{self.task_name} saved {len(df_save)} rows to {file_path}"
            )

    def sync(self, mode="historical", start_year=2009, target_date=None):
        if mode == "historical":
            current_year = datetime.now(timezone(timedelta(hours=8))).year
            periods = self._generate_periods(start_year, current_year)
            self.logger.info(
                f"=== historical {self.task_name}: VIP period + report_type ==="
            )
            for period in periods:
                df_period = self._fetch_period(period)
                self._process_and_save(df_period)
            self.logger.info(f"=== historical {self.task_name} complete ===")
            return

        if mode == "incremental":
            target_date = target_date or datetime.now(timezone(timedelta(hours=8))).strftime(
                "%Y%m%d"
            )
            if not is_financial_statement_period(target_date):
                self.logger.info(
                    f"{self.task_name} skip {target_date}: not a financial statement period"
                )
                return
            self.logger.info(
                f"=== incremental {self.task_name}: period={target_date} ==="
            )
            df_period = self._fetch_period(target_date)
            self._process_and_save(df_period)
            self.logger.info(f"=== incremental {self.task_name} complete ===")
            return

        raise ValueError(f"unsupported sync mode: {mode}")


class IncomeDownloader(BaseFinancialStatementDownloader):
    def __init__(self):
        super().__init__("income_vip", "fin_income_dir", "income statement")


class BalanceSheetDownloader(BaseFinancialStatementDownloader):
    def __init__(self):
        super().__init__("balancesheet_vip", "fin_balance_dir", "balance sheet")


class CashFlowDownloader(BaseFinancialStatementDownloader):
    def __init__(self):
        super().__init__("cashflow_vip", "fin_cashflow_dir", "cashflow statement")


class _LegacyVipPeriodDownloader(BaseDownloader):
    """Keep forecast/express imports stable; not used by stock_financial rebuild."""

    def __init__(self, vip_api_name, dir_config_key, task_name):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("financial_vip", 200))
        self.page_limit = config["api"]["page_limits"].get("financial_vip", 5000)
        self.save_dir = self.get_full_path_and_ensure_dir(dir_config_key)
        self.vip_api_name = vip_api_name
        self.task_name = task_name

    def _generate_periods(self, start_year, end_year):
        return [
            f"{year}{suffix}"
            for year in range(start_year, end_year + 1)
            for suffix in QUARTER_SUFFIXES
        ]

    def sync(self, mode="historical", start_year=2009, target_date=None):
        if mode != "historical":
            self.logger.info(f"{self.task_name} legacy incremental is unchanged/skipped")
            return
        api_func = getattr(self.pro, self.vip_api_name)
        current_year = datetime.now(timezone(timedelta(hours=8))).year
        for period in self._generate_periods(start_year, current_year):
            offset = 0
            chunks = []
            while True:
                df = api_func(period=period, limit=self.page_limit, offset=offset)
                if df is None or df.empty:
                    break
                chunks.append(df)
                if len(df) < self.page_limit:
                    break
                offset += self.page_limit
                self.safe_sleep()
            if chunks:
                self._save_legacy(_concat_preserve_schema(chunks))

    def _save_legacy(self, df_chunk):
        if "ann_date" not in df_chunk.columns:
            return
        df_chunk = df_chunk.copy()
        df_chunk["ann_year"] = df_chunk["ann_date"].astype(str).str[:4]
        for year, df_year in df_chunk.groupby("ann_year"):
            file_path = os.path.join(self.save_dir, f"{year}.parquet")
            df_save = df_year.drop(columns=["ann_year"])
            if os.path.exists(file_path):
                df_save = _concat_preserve_schema([pd.read_parquet(file_path), df_save])
            df_save.drop_duplicates(keep="last", inplace=True)
            df_save.to_parquet(file_path, index=False)


class ForecastDownloader(_LegacyVipPeriodDownloader):
    def __init__(self):
        super().__init__("forecast_vip", "fin_forecast_dir", "forecast")


class ExpressDownloader(_LegacyVipPeriodDownloader):
    def __init__(self):
        super().__init__("express_vip", "fin_express_dir", "express")
