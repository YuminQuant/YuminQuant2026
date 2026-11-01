from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib
import json
import math
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml_alpha"))
from yq_ml_alpha.data.factor_files import factor_file_date
from yq_ml_alpha.output.file_lock import daily_file_lock
from yq_ml_alpha.output import artifact_io


ROW_KEYS = {"trade_date", "ts_code", "trade_time"}


class IoRate:
    """Pace filesystem operations across all workers, not per file."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate
        self.next_call = time.monotonic()
        self.lock = threading.Lock()

    def pace(self):
        with self.lock:
            start = max(time.monotonic(), self.next_call)
            self.next_call = start + self.interval
        time.sleep(max(0.0, start - time.monotonic()))


def initialize_hdfs(bootstrap: str, workers: int, rate: float, root: Path):
    # The deployment owns credentials, mounts, distributed locks and atomic rename.
    # Configure its native process-wide budget before initializing its client.
    config = json.loads(os.environ["YQ_HDFS_CONFIG"])
    config["io_concurrency"] = workers
    config["read_concurrency"] = workers
    config["read_opens_per_second"] = rate
    os.environ["YQ_HDFS_CONFIG"] = json.dumps(config)
    module, function = bootstrap.split(":", 1)
    getattr(importlib.import_module(module), function)()
    store = artifact_io.backend(root)
    if store is None:
        raise ValueError("HDFS root is not mounted; local fallback is forbidden")
    for name in ("open", "glob", "stat", "exists", "remove", "replace", "lock"):
        if not callable(getattr(store, name, None)):
            raise ValueError(f"HDFS backend is missing {name}")


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
    return list(dict.fromkeys(columns))


def iter_factor_files(root: Path, start_date: int, end_date: int, remote=False, rate=None):
    start_year = start_date // 10_000
    end_year = end_date // 10_000
    for year in range(start_year, end_year + 1):
        year_dir = root / str(year)
        if remote:
            rate.pace()
            paths = artifact_io.glob(year_dir, "*.parquet")
        else:
            if not year_dir.exists():
                continue
            paths = year_dir.glob("*.parquet")
        for path in sorted(paths):
            if path.parent != year_dir:
                continue
            trade_date = factor_file_date(path)
            if trade_date is None:
                continue
            if start_date <= trade_date <= end_date:
                yield path, trade_date


def remove_columns(path: Path, columns: list[str], dry_run: bool) -> list[str]:
    if ROW_KEYS.intersection(columns):
        raise ValueError("cannot remove factor row keys")
    with daily_file_lock(path):
        return _remove_columns_locked(path, columns, dry_run)


def _remove_columns_locked(path: Path, columns: list[str], dry_run: bool) -> list[str]:
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


def remove_hdfs_columns(path: Path, columns: list[str], dry_run: bool,
                        rate: IoRate, batch_rows: int) -> list[str]:
    if ROW_KEYS.intersection(columns):
        raise ValueError("cannot remove factor row keys")
    if dry_run:
        rate.pace()
        with artifact_io.open_file(path, "rb") as source:
            names = pq.ParquetFile(source).schema_arrow.names
        return [column for column in columns if column in names]
    rate.pace()
    with daily_file_lock(path):
        rate.pace()
        original = artifact_io.stat(path)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp_remove_columns")
        try:
            rate.pace()
            with artifact_io.open_file(path, "rb") as source:
                parquet = pq.ParquetFile(source)
                schema = parquet.schema_arrow
                existing = [column for column in columns if column in schema.names]
                if not existing:
                    return existing
                keep = [name for name in schema.names if name not in set(columns)]
                if not ROW_KEYS.intersection(keep):
                    raise ValueError(f"factor file has no row keys: {path}")
                retained_schema = pa.schema([schema.field(name) for name in keep],
                                            metadata=schema.metadata)
                # Keep pandas metadata consistent with the projected Arrow schema.
                metadata = dict(retained_schema.metadata or {})
                if b"pandas" in metadata:
                    pandas = json.loads(metadata[b"pandas"])
                    pandas["columns"] = [item for item in pandas["columns"]
                                         if item.get("field_name") in keep]
                    metadata[b"pandas"] = json.dumps(pandas).encode()
                    retained_schema = retained_schema.with_metadata(metadata)
                rows = parquet.metadata.num_rows
                rate.pace()
                with artifact_io.open_file(temp, "wb") as target:
                    with pq.ParquetWriter(target, retained_schema, compression="snappy") as writer:
                        for batch in parquet.iter_batches(batch_size=batch_rows, columns=keep,
                                                          use_threads=False):
                            writer.write_batch(batch.replace_schema_metadata(retained_schema.metadata))
            # Validate the closed temp file before exposing it to any reader.
            rate.pace()
            with artifact_io.open_file(temp, "rb") as check:
                rewritten = pq.ParquetFile(check)
                if (rewritten.metadata.num_rows != rows or
                        not rewritten.schema_arrow.equals(retained_schema, check_metadata=True)):
                    raise ValueError(f"temporary parquet validation failed: {path}")
            rate.pace()
            current = artifact_io.stat(path)
            if (current.st_size, current.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
                raise RuntimeError(f"factor file changed while locked; refusing replacement: {path}")
            rate.pace()
            artifact_io.replace(temp, path)
            return existing
        finally:
            rate.pace()
            if artifact_io.exists(temp):
                rate.pace()
                artifact_io.unlink(temp)


def bounded_results(function, files, workers):
    """Bound queued work as well as active files; surface every worker failure."""
    iterator = iter(files)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        try:
            while True:
                while len(pending) < workers:
                    item = next(iterator, None)
                    if item is None:
                        break
                    path, date = item
                    pending[pool.submit(function, path)] = (path, date)
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    path, _date = pending.pop(future)
                    try:
                        yield path, future.result()
                    except Exception as error:
                        raise RuntimeError(f"failed to remove columns from {path}: {error}") from error
        finally:
            for future in pending:
                future.cancel()


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
    parser.add_argument("--backend", choices=["local", "hdfs"], default="local",
                        help="HDFS mode never reads or modifies local factor files.")
    parser.add_argument("--hdfs-bootstrap", help="Deployment module:function registering the artifact backend. "
                        "Requires YQ_HDFS_CONFIG and the deployment's mount/auth environment.")
    parser.add_argument("--workers", type=int, help="Parallel files; defaults to local=1, HDFS=50 (max 50).")
    parser.add_argument("--io-requests-per-second", type=float, default=1000.0,
                        help="Process-wide filesystem-call limit in HDFS mode (max 1000).")
    parser.add_argument("--batch-rows", type=int, default=1024,
                        help="Rows per projected HDFS rewrite batch; bounds per-worker table memory.")
    args = parser.parse_args()

    start_date = parse_date(args.start_date, "start-date")
    end_date = parse_date(args.end_date, "end-date")
    if start_date > end_date:
        parser.error("start-date must be <= end-date")

    root = Path(args.factor_root)
    columns = parse_columns(args.columns)
    if ROW_KEYS.intersection(columns):
        parser.error("cannot remove factor row keys")
    workers = args.workers if args.workers is not None else (50 if args.backend == "hdfs" else 1)
    if not 1 <= workers <= 50:
        parser.error("workers must be between 1 and 50")
    if (not math.isfinite(args.io_requests_per_second) or
            not 0 < args.io_requests_per_second <= 1000):
        parser.error("io-requests-per-second must be in (0, 1000]")
    if args.batch_rows <= 0:
        parser.error("batch-rows must be positive")
    rate = IoRate(args.io_requests_per_second)
    remote = args.backend == "hdfs"
    if remote:
        if (not args.hdfs_bootstrap or not args.columns or not root.is_absolute()
                or args.factor_root == parser.get_default("factor_root")):
            parser.error("HDFS mode requires --hdfs-bootstrap, --columns and an explicit absolute --factor-root")
        initialize_hdfs(args.hdfs_bootstrap, workers, args.io_requests_per_second, root)
    scanned = 0
    touched = 0
    removed_counts = {column: 0 for column in columns}

    files = iter_factor_files(root, start_date, end_date, remote, rate)
    def process(path):
        if remote:
            return remove_hdfs_columns(path, columns, args.dry_run, rate, args.batch_rows)
        return remove_columns(path, columns, args.dry_run)

    for path, existing in bounded_results(process, files, workers):
        scanned += 1
        if not existing:
            if scanned % 100 == 0:
                print(f"progress: scanned={scanned} touched={touched}", flush=True)
            continue
        touched += 1
        for column in existing:
            removed_counts[column] += 1
        action = "would update" if args.dry_run else "updated"
        print(f"{action}: {path} columns={','.join(existing)}", flush=True)

    mode = "dry-run" if args.dry_run else "write"
    print(f"mode: {mode}")
    print(f"backend: {args.backend}")
    print(f"workers: {workers}")
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
