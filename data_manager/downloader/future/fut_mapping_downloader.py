from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager
from data_manager.core.daily_storage import daily_file_path


class FutureMappingDownloader(BaseDownloader):
    """Store the provider's full main/continuous mapping snapshot by trade date."""

    def __init__(self):
        config = ConfigManager().config
        super().__init__(rate_limit=config["api"]["rate_limits"].get("fut_mapping", 200))
        self.root = Path(self.base_data_dir)
        self.calendar_dir = self.root / self.config["paths"].get("calendar_dir", "calendar")

    def sync(self, start_date=None, target_end_date=None):
        end = target_end_date or datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
        start = start_date or end
        if datetime.strptime(start, "%Y%m%d") > datetime.strptime(end, "%Y%m%d"):
            raise ValueError("fut_mapping start_date must be on or before end_date")
        calendar = pd.read_parquet(
            self.calendar_dir / "trade_cal_SHFE.parquet", columns=["cal_date", "is_open"]
        )
        dates = calendar.loc[
            calendar["is_open"].eq(1) & calendar["cal_date"].between(int(start), int(end)),
            "cal_date",
        ]
        for date in sorted(dates.astype(str).unique()):
            path = Path(daily_file_path(self.root / "future_data/basic/mapping", date))
            if path.is_file():
                continue
            self.safe_sleep()
            frame = self.pro.fut_mapping(trade_date=date)
            if frame is None or frame.empty:
                self.logger.warning(f"fut_mapping {date}: no rows returned")
                continue
            if len(frame) >= 2000 or not frame["trade_date"].astype(str).eq(date).all():
                raise ValueError(f"fut_mapping {date}: truncated or invalid date snapshot")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                frame.to_parquet(temp, index=False)
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            self.logger.info(f"fut_mapping {date}: saved {len(frame)} rows to {path}")
