from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TradingCalendar:
    dates: list[int]

    @classmethod
    def load(cls, data_root: str | Path = "data", exchange: str = "SSE") -> "TradingCalendar":
        path = Path(data_root) / "calendar" / f"trade_cal_{exchange}.parquet"
        table = pd.read_parquet(path, columns=["cal_date", "is_open"])
        dates = table.loc[table["is_open"].astype(int) == 1, "cal_date"].astype(int).tolist()
        return cls(sorted(dates))

    def between(self, start: int, end: int) -> list[int]:
        return [date for date in self.dates if start <= date <= end]

    def previous_open(self, date: int) -> int | None:
        before = [item for item in self.dates if item < date]
        return before[-1] if before else None

    def offset(self, date: int, offset: int) -> int | None:
        if date not in self.dates:
            return None
        idx = self.dates.index(date) + offset
        if 0 <= idx < len(self.dates):
            return self.dates[idx]
        return None
