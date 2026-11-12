from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import daily_file_path


class FutureSettleDownloader(BaseDownloader):
    """Store daily futures settlement parameters by exchange."""

    EXCHANGES = ("SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX")
    VALUE_COLUMNS = (
        "settle",
        "trading_fee_rate",
        "trading_fee",
        "delivery_fee",
        "b_hedging_margin_rate",
        "s_hedging_margin_rate",
        "long_margin_rate",
        "short_margin_rate",
        "offset_today_fee",
    )
    COLUMNS = ("ts_code", "trade_date", *VALUE_COLUMNS, "exchange")

    def __init__(self):
        config = ConfigManager().config
        api = config["api"]
        rate_limit = int(api["rate_limits"].get("fut_settle", 480))
        self.page_limit = int(api.get("page_limits", {}).get("fut_settle", 1600))
        if rate_limit <= 0 or self.page_limit <= 0:
            raise ValueError("fut_settle rate and page limits must be positive")
        super().__init__(rate_limit=rate_limit)
        paths = self.config["paths"]
        root = Path(self.base_data_dir)
        self.save_dir = root / paths.get("fut_settle_dir", "future_data/basic/settle")
        self.calendar_dir = root / paths.get("calendar_dir", "calendar")

    def _request_page(self, exchange, trade_date, offset):
        for attempt in range(1, 4):
            self.safe_sleep()
            try:
                return self.pro.fut_settle(
                    exchange=exchange,
                    trade_date=trade_date,
                    fields=",".join(self.COLUMNS),
                    limit=self.page_limit,
                    offset=offset,
                )
            except Exception as exc:
                self.logger.warning(
                    f"fut_settle {exchange} {trade_date} offset={offset} attempt={attempt}/3: {exc}"
                )
                if attempt == 3:
                    raise

    def _fetch_day(self, exchange, trade_date):
        pages = []
        seen_codes = set()
        offset = 0
        scope = f"fut_settle {exchange} {trade_date}"
        # A server-side cap can produce a short page even if there are more rows.
        while True:
            page = self._request_page(exchange, trade_date, offset)
            if not isinstance(page, pd.DataFrame) or len(page) > self.page_limit:
                raise ValueError(f"{scope}: invalid page at offset={offset}")
            if page.empty:
                break
            missing = set(self.COLUMNS).difference(page.columns)
            if missing:
                raise ValueError(f"{scope}: missing fields {sorted(missing)}")
            valid_date = page["trade_date"].astype(str).eq(trade_date).all()
            valid_exchange = page["exchange"].eq(exchange).all()
            if not valid_date or not valid_exchange:
                raise ValueError(f"{scope}: unexpected date or exchange at offset={offset}")
            codes = page["ts_code"]
            if codes.isna().any() or codes.astype(str).str.strip().eq("").any():
                raise ValueError(f"{scope}: missing contract code at offset={offset}")
            if codes.duplicated().any() or seen_codes.intersection(codes):
                raise ValueError(f"{scope}: repeated contract at offset={offset}")
            seen_codes.update(codes)
            pages.append(page.loc[:, list(self.COLUMNS)])
            offset += len(page)

        if not pages:
            return pd.DataFrame()
        frame = pd.concat(pages, ignore_index=True)
        frame["trade_date"] = frame["trade_date"].astype("int32")
        for column in self.VALUE_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        return frame.sort_values("ts_code", ignore_index=True)

    def sync(self, start_date=None, target_end_date=None):
        end = target_end_date or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        start = start_date or end
        if datetime.strptime(start, "%Y%m%d") > datetime.strptime(end, "%Y%m%d"):
            raise ValueError("fut_settle start_date must be on or before end_date")

        for exchange in self.EXCHANGES:
            calendar = pd.read_parquet(
                self.calendar_dir / f"trade_cal_{exchange}.parquet",
                columns=["cal_date", "is_open"],
            )
            dates = calendar.loc[
                calendar["is_open"].eq(1) & calendar["cal_date"].between(int(start), int(end)),
                "cal_date",
            ]
            for trade_date in sorted(dates.astype(str).unique()):
                path = Path(daily_file_path(self.save_dir / exchange, trade_date))
                if path.is_file():
                    continue
                frame = self._fetch_day(exchange, trade_date)
                if frame.empty:
                    self.logger.warning(f"fut_settle {exchange} {trade_date}: no rows returned")
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                try:
                    frame.to_parquet(temp, index=False)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
                self.logger.info(f"fut_settle {exchange} {trade_date}: saved {len(frame)} rows to {path}")
