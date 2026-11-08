from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import daily_file_path


class FutureTradeParamDownloader(BaseDownloader):
    """Download one complete trade-parameter snapshot per exchange and date."""

    ENDPOINTS = {
        "CZCE": "fut_trade_param",
        "CFFEX": "fut_trade_param_cfx",
        "GFEX": "fut_trade_param_gfe",
        "DCE": "fut_trade_param_dce",
    }
    PAGE_LIMIT = 3000

    def __init__(self):
        config = ConfigManager().config
        rate_limit = int(config["api"]["rate_limits"].get("fut_trade_param", 500))
        if not 1 <= rate_limit <= 500:
            raise ValueError("fut_trade_param rate limit must be between 1 and 500")
        super().__init__(rate_limit=rate_limit)
        paths = self.config["paths"]
        self.save_dir = Path(self.base_data_dir) / paths.get(
            "fut_trade_param_dir", "future_data/basic/trade_param"
        )
        self.calendar_dir = Path(self.base_data_dir) / paths.get("calendar_dir", "calendar")

    def _request_page(self, endpoint, trade_date, offset):
        request = getattr(self.pro, endpoint)
        for attempt in range(1, 4):
            self.safe_sleep()
            try:
                return request(trade_date=trade_date, limit=self.PAGE_LIMIT, offset=offset)
            except Exception as exc:
                self.logger.warning(
                    f"{endpoint} {trade_date} offset={offset} attempt={attempt}/3: {exc}"
                )
                if attempt == 3:
                    raise

    def _fetch_day(self, endpoint, trade_date):
        pages = []
        seen_rows = set()
        offset = 0
        while True:
            page = self._request_page(endpoint, trade_date, offset)
            if not isinstance(page, pd.DataFrame) or len(page) > self.PAGE_LIMIT:
                raise ValueError(f"{endpoint} {trade_date}: invalid page at offset={offset}")
            if page.empty:
                break
            if not page["trade_date"].astype(str).eq(trade_date).all():
                raise ValueError(f"{endpoint} {trade_date}: unexpected trade_date at offset={offset}")
            if pages and not page.columns.equals(pages[0].columns):
                raise ValueError(f"{endpoint} {trade_date}: schema changed at offset={offset}")
            row_hashes = set(pd.util.hash_pandas_object(page, index=False))
            if seen_rows.intersection(row_hashes):
                raise ValueError(f"{endpoint} {trade_date}: overlapping pages at offset={offset}")
            seen_rows.update(row_hashes)
            pages.append(page)
            if len(page) < self.PAGE_LIMIT:
                break
            offset += len(page)
        return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()

    def sync(self, start_date=None, target_end_date=None):
        end_date = target_end_date or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        start_date = start_date or end_date
        if datetime.strptime(start_date, "%Y%m%d") > datetime.strptime(end_date, "%Y%m%d"):
            raise ValueError("fut_trade_param start_date must be on or before end_date")

        for exchange, endpoint in self.ENDPOINTS.items():
            calendar = pd.read_parquet(
                self.calendar_dir / f"trade_cal_{exchange}.parquet",
                columns=["cal_date", "is_open"],
            )
            dates = calendar.loc[
                calendar["is_open"].eq(1)
                & calendar["cal_date"].between(int(start_date), int(end_date)),
                "cal_date",
            ]
            for trade_date in sorted(dates.astype(str).unique()):
                path = Path(daily_file_path(self.save_dir / exchange, trade_date))
                if path.is_file():
                    continue
                frame = self._fetch_day(endpoint, trade_date)
                if frame.empty:
                    self.logger.warning(f"{endpoint} {trade_date}: no rows returned")
                    continue

                path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    frame.to_parquet(temp_path, index=False)
                    temp_path.replace(path)
                finally:
                    temp_path.unlink(missing_ok=True)
                self.logger.info(f"{endpoint} {trade_date}: saved {len(frame)} rows to {path}")
