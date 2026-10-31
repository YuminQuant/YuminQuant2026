from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from yq_ml_alpha.data.stores import daily_path, read_daily
from yq_ml_alpha.output.file_lock import daily_file_lock
from yq_ml_alpha.output import artifact_io


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
        file_suffix: str = "",
    ) -> None:
        self.output_root = Path(output_root)
        self.output_id = output_id
        self.layout = layout
        self.asset = asset
        self.frequency = frequency
        self.base_root = Path(base_root) if base_root is not None else None
        if file_suffix not in {"", "_ml"}:
            raise ValueError("daily output file_suffix must be empty or _ml")
        self.file_suffix = file_suffix
        self._owners = {}
        if file_suffix == "_ml":
            from yq_ml_alpha.data.factor_files import FactorFiles
            root = self.output_root / asset / frequency if layout == "standard" else self.output_root
            self._owners = FactorFiles(root).owners
            self._check_owner(output_id)
        self.write_workers = max(1, int(write_workers))
        if data_io.backend(self.output_root) is not None:
            self.write_workers = data_io.io_options(self.output_root)["workers"]
        self._schema_dates_seen: set[int] = set()
        self._known_columns = set(KEY_COLUMNS) | {output_id}

    def _map_writes(self, function, items):
        if data_io.backend(self.output_root) is not None:
            return list(data_io.ordered_io_map(function, items, root=self.output_root,
                path_for=lambda item: self._path(item[0]), workers=self.write_workers))
        with ThreadPoolExecutor(max_workers=min(self.write_workers, len(items))) as pool:
            return list(pool.map(function, items))

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
        return self._map_writes(lambda item: self._write_one(item[0], item[1], item[2]), tasks)

    def dates_missing_output_column(self, coverage_dates: Iterable[int]) -> list[int]:
        dates = sorted({int(date) for date in coverage_dates})
        missing = []
        for trade_date in dates:
            path = self._path(trade_date)
            if not artifact_io.exists(path):
                missing.append(trade_date)
                continue
            if self.output_id not in parquet_columns(path):
                missing.append(trade_date)
        return missing

    def write_many(self, predictions: dict[str, pd.DataFrame], *,
                   coverage_dates: dict[str, Iterable[int]] | None = None,
                   schema_dates: Iterable[int] | None = None) -> list[Path]:
        """Merge several factor columns with one read/compression/write per day.

        Coverage is per column: a factor not covering a date is left untouched.
        Empty predictions on a covered date intentionally replace it with NaN.
        """
        tasks = {}
        for output_id, frame in predictions.items():
            self._check_owner(output_id)
            if not output_id or output_id in KEY_COLUMNS:
                raise ValueError(f"invalid output column: {output_id}")
            if not set([*KEY_COLUMNS, "score"]).issubset(frame.columns):
                raise ValueError(f"{output_id}: missing prediction keys/score")
            coverage = None if coverage_dates is None else coverage_dates.get(output_id, [])
            grouped = _prediction_groups(frame)
            for date in _coverage_dates(frame, coverage):
                tasks.setdefault(date, {})[output_id] = grouped.get(date)
        if not tasks:
            return []
        self._known_columns.update(predictions)
        columns = self._schema_columns(sorted(tasks) if schema_dates is None else list(schema_dates))
        items = sorted(tasks.items())
        def write(item):
            date, values = item
            return self._write_columns_one(date, values, columns)
        return self._map_writes(write, items)

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
            paths = self._map_writes(lambda item: self._ensure_output_column_one(item[0], item[1]), tasks)
        return [path for path in paths if path is not None]

    def _write_one(self, trade_date: int, daily: pd.DataFrame | None, columns: list[str]) -> Path:
        return self._write_columns_one(trade_date, {self.output_id: daily}, columns)

    def _write_columns_one(self, trade_date, daily_by_column, columns):
        path = self._path(trade_date)
        with daily_file_lock(path):
            daily = next((value for value in daily_by_column.values() if value is not None and not value.empty), None)
            base = self._load_base(path, trade_date, daily)
            values = [
                _daily_values(trade_date, frame, name).set_index(KEY_COLUMNS)
                for name, frame in daily_by_column.items()
            ]
            joined = pd.concat(values, axis=1).reset_index()
            base = base.drop(columns=list(daily_by_column), errors="ignore")
            if joined.empty:
                merged = base.copy()
                for name in daily_by_column:
                    merged[name] = np.nan
            else:
                joined["trade_date"] = joined["trade_date"].astype("int32")
                merged = base.merge(joined, on=KEY_COLUMNS, how="outer", sort=False)
            # An absent ML column may still live in a legacy mixed file. Do not
            # pad it with nulls merely because another date/model has that column.
            names = set(merged.columns) if self.file_suffix == "_ml" else set(columns) | set(merged.columns)
            ordered = KEY_COLUMNS + sorted(names - set(KEY_COLUMNS))
            merged = merged.reindex(columns=ordered).sort_values(KEY_COLUMNS).reset_index(drop=True)
            merged["trade_date"] = merged["trade_date"].astype("int32")
            merged[ordered[2:]] = merged[ordered[2:]].astype("float32")
            self._atomic_write(path, merged)
        return path

    @staticmethod
    def _atomic_write(path, frame):
        if artifact_io.backend(path):
            artifact_io.write_parquet(frame, path, index=False)
            return
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            data_io.write_parquet(frame, tmp, index=False)
            data_io.replace(tmp, path)
        finally:
            if data_io.exists(tmp):
                data_io.unlink(tmp)

    def _ensure_output_column_one(self, trade_date: int, columns: list[str]) -> Path | None:
        path = self._path(trade_date)
        with daily_file_lock(path):
            if artifact_io.exists(path) and self.output_id in parquet_columns(path):
                return None
            base = self._load_base(path, trade_date, None)
            values = pd.DataFrame(columns=[*KEY_COLUMNS, self.output_id])
            local_columns = [*base.columns, self.output_id] if self.file_suffix == "_ml" else columns
            merged = _merge_values(base, values, self.output_id, local_columns)
            self._atomic_write(path, merged)
        return path

    def _schema_columns(self, dates: list[int]) -> list[str]:
        if self.file_suffix == "_ml":
            return KEY_COLUMNS + sorted(self._known_columns - set(KEY_COLUMNS))
        for trade_date in dates:
            if trade_date in self._schema_dates_seen:
                continue
            path = self._path(trade_date)
            if artifact_io.exists(path):
                self._known_columns.update(parquet_columns(path))
            self._schema_dates_seen.add(trade_date)
        value_columns = sorted(column for column in self._known_columns if column not in KEY_COLUMNS)
        return KEY_COLUMNS + value_columns

    def _load_base(self, path: Path, trade_date: int, daily: pd.DataFrame | None) -> pd.DataFrame:
        if artifact_io.exists(path):
            frame = artifact_io.read_parquet(path)
            if self.file_suffix == "_ml" and any(self._owners.get(c) == "manual" for c in frame.columns):
                raise ValueError(f"ML output contains manual-owned columns: {path}")
            return frame
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
            return self.output_root / self.asset / self.frequency / year / f"{trade_date}{self.file_suffix}.parquet"
        return self.output_root / year / f"{trade_date}{self.file_suffix}.parquet"

    def _check_owner(self, output_id):
        if self.file_suffix == "_ml" and self._owners.get(output_id) == "manual":
            raise ValueError(f"ML writer cannot update manual-owned factor: {output_id}")


def parquet_columns(path: str | Path) -> list[str]:
    return artifact_io.parquet_columns(path)


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
    columns = KEY_COLUMNS + sorted((set(columns) | set(merged.columns)) - set(KEY_COLUMNS))
    for column in columns:
        if column not in merged.columns:
            merged[column] = np.nan
    value_columns = [column for column in columns if column not in KEY_COLUMNS]
    for column in value_columns:
        merged[column] = merged[column].astype("float32")
    return merged[columns]
