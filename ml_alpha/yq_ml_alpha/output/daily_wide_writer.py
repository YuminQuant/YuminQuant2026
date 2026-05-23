from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from yq_ml_alpha.data.stores import daily_path, read_daily


KEY_COLUMNS = ["trade_date", "ts_code"]
Layout = Literal["direct", "standard"]


class DailyWideWriter:
    """Write one daily wide parquet column with stable coverage and overwrite semantics."""

    def __init__(
        self,
        output_root: str | Path,
        output_id: str,
        *,
        layout: Layout = "direct",
        asset: str = "stock",
        frequency: str = "daily",
        base_root: str | Path | None = None,
        write_workers: int = 4,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_id = output_id
        self.layout = layout
        self.asset = asset
        self.frequency = frequency
        self.base_root = Path(base_root) if base_root is not None else None
        self.write_workers = max(1, int(write_workers))

    def write(
        self,
        predictions: pd.DataFrame,
        *,
        coverage_dates: Iterable[int] | None = None,
        schema_dates: Iterable[int] | None = None,
    ) -> list[Path]:
        required = {"trade_date", "ts_code", "score"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")

        dates = _coverage_dates(predictions, coverage_dates)
        if not dates:
            return []
        grouped = _prediction_groups(predictions)
        schema_source_dates = _coverage_dates(predictions, schema_dates if schema_dates is not None else dates)
        schema_columns = self._schema_columns(schema_source_dates)
        tasks = [(date, grouped.get(date), schema_columns) for date in dates]
        if self.write_workers == 1 or len(tasks) <= 1:
            return [self._write_one(date, daily, columns) for date, daily, columns in tasks]
        workers = min(self.write_workers, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda item: self._write_one(item[0], item[1], item[2]), tasks))

    def dates_missing_output_column(self, coverage_dates: Iterable[int]) -> list[int]:
        dates = sorted({int(date) for date in coverage_dates})
        missing = []
        for trade_date in dates:
            path = self._path(trade_date)
            if not path.exists():
                missing.append(trade_date)
                continue
            if self.output_id not in parquet_columns(path):
                missing.append(trade_date)
        return missing

    def ensure_output_column(self, coverage_dates: Iterable[int]) -> list[Path]:
        dates = sorted({int(date) for date in coverage_dates})
        if not dates:
            return []
        missing_dates = self.dates_missing_output_column(dates)
        if not missing_dates:
            return []
        print(
            f"daily-wide ensure_output_column id={self.output_id} missing={len(missing_dates)} "
            f"dates={_date_span(missing_dates)}",
            flush=True,
        )
        schema_columns = self._schema_columns(dates)
        tasks = [(date, schema_columns) for date in missing_dates]
        if self.write_workers == 1 or len(tasks) <= 1:
            paths = [self._ensure_output_column_one(date, columns) for date, columns in tasks]
        else:
            workers = min(self.write_workers, len(tasks))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                paths = list(pool.map(lambda item: self._ensure_output_column_one(item[0], item[1]), tasks))
        return [path for path in paths if path is not None]

    def _write_one(self, trade_date: int, daily: pd.DataFrame | None, columns: list[str]) -> Path:
        path = self._path(trade_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        base = self._load_base(path, trade_date, daily)
        values = _daily_values(trade_date, daily, self.output_id)
        merged = _merge_values(base, values, self.output_id, columns)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    def _ensure_output_column_one(self, trade_date: int, columns: list[str]) -> Path | None:
        path = self._path(trade_date)
        if path.exists() and self.output_id in parquet_columns(path):
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        base = self._load_base(path, trade_date, None)
        values = pd.DataFrame(columns=[*KEY_COLUMNS, self.output_id])
        merged = _merge_values(base, values, self.output_id, columns)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        merged.to_parquet(tmp, index=False)
        tmp.replace(path)
        return path

    def _schema_columns(self, dates: list[int]) -> list[str]:
        columns = set(KEY_COLUMNS)
        columns.add(self.output_id)
        for trade_date in dates:
            path = self._path(trade_date)
            if path.exists():
                columns.update(parquet_columns(path))
        value_columns = sorted(column for column in columns if column not in KEY_COLUMNS)
        return KEY_COLUMNS + value_columns

    def _load_base(self, path: Path, trade_date: int, daily: pd.DataFrame | None) -> pd.DataFrame:
        if path.exists():
            return pd.read_parquet(path)
        if self.base_root is not None:
            base = read_daily(self.base_root, trade_date, [])
            if not base.empty:
                return base[KEY_COLUMNS]
            raise FileNotFoundError(
                f"missing base daily pv file for {trade_date}: {daily_path(self.base_root, trade_date)}"
            )
        if daily is not None and not daily.empty:
            return daily[KEY_COLUMNS].copy()
        return pd.DataFrame(columns=KEY_COLUMNS)

    def _path(self, trade_date: int) -> Path:
        year = str(trade_date // 10000)
        if self.layout == "standard":
            return self.output_root / self.asset / self.frequency / year / f"{trade_date}.parquet"
        return self.output_root / year / f"{trade_date}.parquet"


def parquet_columns(path: str | Path) -> list[str]:
    return list(pq.read_schema(path).names)


def _coverage_dates(predictions: pd.DataFrame, coverage_dates: Iterable[int] | None) -> list[int]:
    dates = set()
    if coverage_dates is not None:
        dates.update(int(date) for date in coverage_dates)
    if not predictions.empty:
        dates.update(int(date) for date in predictions["trade_date"].dropna().unique())
    return sorted(dates)


def _date_span(dates: list[int]) -> str:
    if not dates:
        return "none"
    if len(dates) == 1:
        return str(dates[0])
    return f"{dates[0]}..{dates[-1]}"


def _prediction_groups(predictions: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if predictions.empty:
        return {}
    frame = predictions[["trade_date", "ts_code", "score"]].copy()
    frame["trade_date"] = frame["trade_date"].astype("int32")
    frame["score"] = frame["score"].astype("float32")
    output = {}
    for trade_date, daily in frame.groupby("trade_date", sort=False):
        output[int(trade_date)] = daily.drop_duplicates(KEY_COLUMNS, keep="last")
    return output


def _daily_values(trade_date: int, daily: pd.DataFrame | None, output_id: str) -> pd.DataFrame:
    if daily is None or daily.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, output_id])
    values = daily[KEY_COLUMNS + ["score"]].rename(columns={"score": output_id}).copy()
    values["trade_date"] = np.int32(trade_date)
    values[output_id] = values[output_id].astype("float32")
    return values


def _merge_values(
    base: pd.DataFrame,
    values: pd.DataFrame,
    output_id: str,
    columns: list[str],
) -> pd.DataFrame:
    base = base.copy()
    for column in KEY_COLUMNS:
        if column not in base.columns:
            base[column] = pd.Series(dtype="int32" if column == "trade_date" else "object")
    if output_id in base.columns:
        base = base.drop(columns=[output_id])
    if values.empty:
        merged = base.copy()
        merged[output_id] = np.nan
    else:
        merged = base.merge(values, on=KEY_COLUMNS, how="outer", sort=False)
    merged = merged.sort_values(KEY_COLUMNS).reset_index(drop=True)
    merged["trade_date"] = merged["trade_date"].astype("int32")
    for column in columns:
        if column not in merged.columns:
            merged[column] = np.nan
    value_columns = [column for column in columns if column not in KEY_COLUMNS]
    for column in value_columns:
        merged[column] = merged[column].astype("float32")
    return merged[columns]
