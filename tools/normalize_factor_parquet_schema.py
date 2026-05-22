from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


KEY_COLUMNS = {"trade_date", "ts_code"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite external factor parquet files to the Rust backtest-compatible key schema.",
    )
    parser.add_argument("--src", required=True, help="Source factor root containing {year}/{YYYYMMDD}.parquet files.")
    parser.add_argument("--dst", required=True, help="Destination root to write normalized parquet files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination parquet files.",
    )
    return parser.parse_args()


def _normalize_table(table: pa.Table) -> pa.Table:
    arrays = []
    fields = []
    for field in table.schema:
        column = table[field.name]
        if field.name == "trade_date":
            column = pc.cast(column, pa.int32(), safe=True)
            field = pa.field(field.name, pa.int32(), nullable=field.nullable)
        elif field.name == "ts_code":
            column = pc.cast(column, pa.string(), safe=True)
            field = pa.field(field.name, pa.string(), nullable=field.nullable)
        arrays.append(column)
        fields.append(field)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _iter_parquet_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*.parquet") if path.is_file())


def main() -> None:
    args = _parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(f"source root does not exist: {src}")
    if src == dst:
        raise ValueError("destination must be different from source")

    files = _iter_parquet_files(src)
    if not files:
        raise FileNotFoundError(f"no parquet files found under {src}")

    written = 0
    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(src)
        out = dst / rel
        if out.exists() and not args.overwrite:
            continue
        table = pq.read_table(path)
        missing = KEY_COLUMNS.difference(table.column_names)
        if missing:
            raise ValueError(f"{path} missing required key columns: {sorted(missing)}")
        normalized = _normalize_table(table)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(normalized, out)
        written += 1
        if idx == 1 or idx == len(files) or idx % 250 == 0:
            print(f"normalized {idx}/{len(files)} written={written} latest={rel}", flush=True)

    print(f"done source={src} destination={dst} files={len(files)} written={written}", flush=True)


if __name__ == "__main__":
    main()
