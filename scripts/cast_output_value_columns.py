from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


KEY_COLUMNS = {"trade_date", "ts_code", "trade_time"}
KEY_COLUMN_TYPES = {
    "trade_date": pa.int32(),
    "ts_code": pa.string(),
    "trade_time": pa.string(),
}


def parse_date(value: str, name: str) -> int:
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError(f"{name} must be YYYYMMDD, got {value!r}")
    return int(value)


def iter_date_parquet_files(root: Path, start_date: int | None, end_date: int | None):
    for path in sorted(root.rglob("*.parquet")):
        if not path.stem.isdigit():
            continue
        trade_date = int(path.stem)
        if start_date is not None and trade_date < start_date:
            continue
        if end_date is not None and trade_date > end_date:
            continue
        yield path, trade_date


def column_needs_cast(source_type: pa.DataType, target_type: pa.DataType) -> bool:
    return not source_type.equals(target_type)


def cast_array(array: pa.ChunkedArray, target_type: pa.DataType, column: str) -> pa.ChunkedArray:
    try:
        return pc.cast(array, target_type, safe=False)
    except pa.ArrowInvalid as exc:
        raise ValueError(f"cannot cast column {column!r} to {target_type}: {exc}") from exc


def cast_file(path: Path, dry_run: bool) -> tuple[list[str], list[str]]:
    table = pq.read_table(path)
    value_columns = [column for column in table.column_names if column not in KEY_COLUMNS]
    changed: list[str] = []
    arrays: list[pa.ChunkedArray] = []
    fields: list[pa.Field] = []

    for field in table.schema:
        column = field.name
        source_type = field.type
        array = table[column]
        target_type: pa.DataType | None = None

        if column in KEY_COLUMN_TYPES:
            target_type = KEY_COLUMN_TYPES[column]
        elif column in value_columns:
            if not (
                pa.types.is_integer(source_type)
                or pa.types.is_floating(source_type)
                or pa.types.is_decimal(source_type)
            ):
                raise ValueError(
                    f"{path}: non-numeric value column cannot be cast: {column}"
                )
            target_type = pa.float32()

        if target_type is not None and column_needs_cast(source_type, target_type):
            changed.append(column)
            if not dry_run:
                array = cast_array(array, target_type, column)
                field = pa.field(column, target_type, nullable=field.nullable)

        arrays.append(array)
        fields.append(field)

    if not dry_run and changed:
        output = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp_cast_columns")
        try:
            pq.write_table(output, tmp, compression="snappy")
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
    return value_columns, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cast formal output parquet value columns to float32. "
            "Key columns are normalized to Rust-compatible Arrow types: "
            "trade_date=int32 and ts_code/trade_time=utf8."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        help=(
            "Root to scan, for example data/factors/stock/daily, "
            "data/label/stock/daily, or data/barra/stock/daily/CNE6."
        ),
    )
    parser.add_argument("--start-date", help="Optional start date in YYYYMMDD.")
    parser.add_argument("--end-date", help="Optional end date in YYYYMMDD.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files and columns that would be rewritten without writing.",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date, "start-date") if args.start_date else None
    end_date = parse_date(args.end_date, "end-date") if args.end_date else None
    if start_date is not None and end_date is not None and start_date > end_date:
        parser.error("start-date must be <= end-date")

    root = Path(args.root)
    if not root.exists():
        parser.error(f"root does not exist: {root}")

    scanned = 0
    touched = 0
    value_column_count = 0
    changed_column_count = 0
    mode = "dry-run" if args.dry_run else "write"

    for path, _trade_date in iter_date_parquet_files(root, start_date, end_date):
        scanned += 1
        value_columns, changed = cast_file(path, args.dry_run)
        value_column_count += len(value_columns)
        changed_column_count += len(changed)
        if changed:
            touched += 1
            action = "would update" if args.dry_run else "updated"
            print(f"{action}: {path} columns={','.join(changed)}")

    print(f"mode: {mode}")
    print(f"root: {root}")
    print(
        "date_range: "
        f"{start_date if start_date is not None else 'all'}.."
        f"{end_date if end_date is not None else 'all'}"
    )
    print(f"files_scanned: {scanned}")
    print(f"files_touched: {touched}")
    print(f"value_columns_seen: {value_column_count}")
    print(f"columns_cast: {changed_column_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
