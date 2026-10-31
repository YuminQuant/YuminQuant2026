from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

from pathlib import Path
from typing import Iterable

import pandas as pd


KEY_COLUMNS = ["trade_date", "ts_code"]
NON_FEATURE_COLUMNS = {"trade_date", "ts_code", "trade_time"}


def daily_path(root: str | Path, trade_date: int) -> Path:
    root = Path(root)
    return root / str(trade_date // 10000) / f"{trade_date}.parquet"


def read_daily(root: str | Path, trade_date: int, columns: Iterable[str], *, required=()) -> pd.DataFrame:
    path = daily_path(root, trade_date)
    requested = _dedupe([*KEY_COLUMNS, *columns])
    try:
        frame = data_io.read_projected_frame(path, requested, required=required)
    except FileNotFoundError:
        if required:
            raise
        return pd.DataFrame(columns=requested)
    for column in requested:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[requested]


def read_daily_range(root: str | Path, dates: Iterable[int], columns: Iterable[str]) -> pd.DataFrame:
    columns = list(columns)
    frames = list(data_io.ordered_io_map(lambda date: read_daily(root, int(date), columns), dates,
        root=Path(root), path_for=lambda date: daily_path(root, int(date))))
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
    return list(data_io.parquet_schema(path).names)


def discover_value_columns(root: str | Path, end_date: int | None = None, excluded=()) -> list[str]:
    output = []
    seen = set(NON_FEATURE_COLUMNS) | set(excluded)
    paths = []
    for path in sorted(data_io.glob(Path(root), "*/*.parquet")):
        from yq_ml_alpha.data.factor_files import factor_file_date
        date = factor_file_date(path)
        if date is None or (end_date is not None and date > end_date):
            continue
        paths.append(path)
    for columns in data_io.ordered_io_map(parquet_columns, paths, root=Path(root)):
        for column in columns:
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
