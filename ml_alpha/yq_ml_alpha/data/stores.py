from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


KEY_COLUMNS = ["trade_date", "ts_code"]


def daily_path(root: str | Path, trade_date: int) -> Path:
    root = Path(root)
    return root / str(trade_date // 10000) / f"{trade_date}.parquet"


def read_daily(root: str | Path, trade_date: int, columns: Iterable[str]) -> pd.DataFrame:
    path = daily_path(root, trade_date)
    requested = _dedupe([*KEY_COLUMNS, *columns])
    if not path.exists():
        return pd.DataFrame(columns=requested)
    try:
        frame = pd.read_parquet(path, columns=requested)
    except Exception:
        frame = pd.read_parquet(path)
        for column in requested:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[requested]
    for column in requested:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[requested]


def read_daily_range(root: str | Path, dates: Iterable[int], columns: Iterable[str]) -> pd.DataFrame:
    frames = [read_daily(root, int(date), columns) for date in dates]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_dedupe([*KEY_COLUMNS, *columns]))
    return pd.concat(frames, ignore_index=True)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
