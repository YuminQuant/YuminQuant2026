from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from uuid import uuid4

import pandas as pd

from data_manager.core import BaseDownloader, ConfigManager


class FutureMappingDownloader(BaseDownloader):
    """Store complete daily mapping snapshots in one Parquet file per year."""

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
        directory = self.root / "future_data/basic/mapping"
        for year, year_dates in groupby(sorted(dates.astype(str).unique()), key=lambda date: date[:4]):
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{year}.parquet"
            lock = path.with_suffix(".lock")
            with lock.open("x"):
                pass
            try:
                self._sync_year(path, year_dates)
            finally:
                lock.unlink(missing_ok=True)

    def _sync_year(self, path, dates):
        frames = [pd.read_parquet(path)] if path.is_file() else []
        covered = set(frames[0]["trade_date"].astype(str)) if frames else set()
        existing = len(frames)
        for legacy in sorted(path.with_suffix("").glob("????????.parquet")):
            if legacy.stem not in covered:
                frames.append(pd.read_parquet(legacy))
                covered.add(legacy.stem)
        for date in dates:
            if date in covered:
                continue
            self.safe_sleep()
            frame = self.pro.fut_mapping(trade_date=date)
            if frame is None or frame.empty:
                self.logger.warning(f"fut_mapping {date}: no rows returned")
                continue
            if len(frame) >= 2000 or not frame["trade_date"].astype(str).eq(date).all():
                raise ValueError(f"fut_mapping {date}: truncated or invalid date snapshot")
            frames.append(frame)
        if len(frames) == existing:
            return
        merged = pd.concat(frames, ignore_index=True)
        merged["trade_date"] = merged["trade_date"].astype(str)
        merged.sort_values(["trade_date", "ts_code"], inplace=True)
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            merged.to_parquet(temp, index=False)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        self.logger.info(f"fut_mapping {path.stem}: saved {len(merged)} rows to {path}")
