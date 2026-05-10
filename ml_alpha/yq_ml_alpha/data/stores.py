from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


KEY_COLUMNS = ["trade_date", "ts_code"]
NON_FEATURE_COLUMNS = {"trade_date", "ts_code", "trade_time"}


def daily_path(root: str | Path, trade_date: int) -> Path:
    root = Path(root)
    return root / str(trade_date // 10000) / f"{trade_date}.parquet"


def read_daily(root: str | Path, trade_date: int, columns: Iterable[str]) -> pd.DataFrame:
    path = daily_path(root, trade_date)
    requested = _dedupe([*KEY_COLUMNS, *columns])
    if not path.exists():
        return pd.DataFrame(columns=requested)
    try:
        available = set(parquet_columns(path))
        existing = [column for column in requested if column in available]
        frame = pd.read_parquet(path, columns=existing)
    except Exception:
        frame = pd.read_parquet(path)
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


def parquet_columns(path: str | Path) -> list[str]:
    return list(pq.read_schema(path).names)


def discover_value_columns(root: str | Path) -> list[str]:
    output = []
    seen = set(NON_FEATURE_COLUMNS)
    for path in sorted(Path(root).glob("*/*.parquet")):
        for column in parquet_columns(path):
            if column not in seen:
                output.append(column)
                seen.add(column)
    if not output:
        raise ValueError(f"no feature columns discovered under {root}")
    return output


def is_all_column_request(columns: list[str] | str) -> bool:
    if isinstance(columns, str):
        return columns.strip().lower() in {"__all__", "all", "*"}
    return len(columns) == 1 and str(columns[0]).strip().lower() in {"__all__", "all", "*"}
