from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq


DEFAULT_COLUMNS = [
    "WQAlpha001",
    "WQAlpha005",
    "WQAlpha009",
    "WQAlpha010",
    "WQAlpha017",
    "WQAlpha025",
    "WQAlpha028",
    "WQAlpha030",
    "WQAlpha032",
    "WQAlpha034",
    "WQAlpha041",
    "WQAlpha046",
    "WQAlpha047",
    "WQAlpha048",
    "WQAlpha049",
    "WQAlpha051",
    "WQAlpha053",
    "WQAlpha056",
    "WQAlpha060",
    "WQAlpha100",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_date(value: str, name: str) -> int:
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError(f"{name} must be YYYYMMDD, got {value!r}")
    return int(value)


def parse_columns(value: str | None) -> list[str]:
    if value is None:
        return DEFAULT_COLUMNS.copy()
    columns = [item.strip() for item in value.split(",") if item.strip()]
    if not columns:
        raise argparse.ArgumentTypeError("--columns cannot be empty")
    return columns


def iter_factor_files(root: Path, start_date: int, end_date: int):
    start_year = start_date // 10_000
    end_year = end_date // 10_000
    for year in range(start_year, end_year + 1):
        year_dir = root / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("*.parquet")):
            if not path.stem.isdigit():
                continue
            trade_date = int(path.stem)
            if start_date <= trade_date <= end_date:
                yield path, trade_date


def remove_columns(path: Path, columns: list[str], dry_run: bool) -> list[str]:
    column_set = set(columns)
    schema = pq.read_schema(path)
    schema_columns = list(schema.names)
    existing = [column for column in columns if column in schema_columns]
    if not existing:
        return []
    if not dry_run:
        keep_columns = [column for column in schema_columns if column not in column_set]
        table = pq.read_table(path, columns=keep_columns)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp_remove_columns")
        try:
            pq.write_table(table, tmp, compression="snappy")
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove factor columns from daily factor parquet files."
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End date in YYYYMMDD.")
    parser.add_argument(
        "--factor-root",
        default=str(project_root() / "data" / "factors" / "stock" / "daily"),
        help="Daily factor root. Defaults to data/factors/stock/daily.",
    )
    parser.add_argument(
        "--columns",
        help="Comma-separated columns to remove. Defaults to deprecated WQAlpha columns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report files and columns that would be rewritten.",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date, "start-date")
    end_date = parse_date(args.end_date, "end-date")
    if start_date > end_date:
        parser.error("start-date must be <= end-date")

    root = Path(args.factor_root)
    columns = parse_columns(args.columns)
    scanned = 0
    touched = 0
    removed_counts = {column: 0 for column in columns}

    for path, _trade_date in iter_factor_files(root, start_date, end_date):
        scanned += 1
        existing = remove_columns(path, columns, args.dry_run)
        if not existing:
            continue
        touched += 1
        for column in existing:
            removed_counts[column] += 1
        action = "would update" if args.dry_run else "updated"
        print(f"{action}: {path} columns={','.join(existing)}")

    mode = "dry-run" if args.dry_run else "write"
    print(f"mode: {mode}")
    print(f"factor_root: {root}")
    print(f"date_range: {start_date}..{end_date}")
    print(f"files_scanned: {scanned}")
    print(f"files_touched: {touched}")
    for column, count in removed_counts.items():
        if count:
            print(f"{column}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
