import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype


KEY_COLUMNS = {"trade_date", "ts_code", "trade_time"}


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


def cast_file(path: Path, dry_run: bool) -> tuple[list[str], list[str]]:
    df = pd.read_parquet(path)
    value_columns = [column for column in df.columns if column not in KEY_COLUMNS]
    invalid = [
        column for column in value_columns if not is_numeric_dtype(df[column].dtype)
    ]
    if invalid:
        columns = ", ".join(invalid)
        raise ValueError(f"{path}: non-numeric value columns cannot be cast: {columns}")

    changed = [
        column
        for column in value_columns
        if str(df[column].dtype).lower() != "float32"
    ]
    if not dry_run and changed:
        for column in changed:
            df[column] = df[column].astype("float32")
        df.to_parquet(path, index=False)
    return value_columns, changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cast formal output parquet value columns to float32. "
            "Key columns trade_date/ts_code/trade_time are preserved."
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
    print(f"value_columns_cast: {changed_column_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
