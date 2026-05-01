import argparse
import os
import sys
from pathlib import Path

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_manager.core import ConfigManager
from data_manager.core.daily_storage import save_daily_dataframe


DATASETS = {
    "stock_daily_pv": {
        "config_key": "stock_daily_pv_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_daily_basic": {
        "config_key": "stock_daily_basic_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_adj_factor": {
        "config_key": "stock_adj_factor_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_daily_limit": {
        "config_key": "stock_daily_limit_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_suspend": {
        "config_key": "stock_suspend_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_moneyflow": {
        "config_key": "stock_moneyflow_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "stock_st": {
        "config_key": "st_list_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
    "future_daily": {
        "config_key": "fut_daily_dir",
        "key_cols": ["ts_code", "trade_date"],
        "sort_cols": ["trade_date", "ts_code"],
    },
}


def resolve_data_dir(config, config_key):
    base_data_dir = Path(config["paths"]["base_data_dir"])
    configured = Path(config["paths"][config_key])
    return configured if configured.is_absolute() else base_data_dir / configured


def yearly_files(base_dir, start_date=None, end_date=None):
    start_year = int(str(start_date)[:4]) if start_date else None
    end_year = int(str(end_date)[:4]) if end_date else None
    paths = []
    for path in Path(base_dir).glob("*.parquet"):
        if not path.stem.isdigit() or len(path.stem) != 4:
            continue
        year = int(path.stem)
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        paths.append(path)
    return sorted(paths)


def in_range(df, start_date, end_date):
    if start_date is not None:
        df = df[df["trade_date"] >= int(start_date)]
    if end_date is not None:
        df = df[df["trade_date"] <= int(end_date)]
    return df


def migrate_year_file(path, base_dir, key_cols, sort_cols, start_date, end_date, dry_run, overwrite):
    df = pd.read_parquet(path)
    if "trade_date" not in df.columns:
        raise ValueError(f"{path} missing trade_date")
    df["trade_date"] = df["trade_date"].astype("int32")
    df = in_range(df, start_date, end_date)
    if df.empty:
        return {"source": str(path), "rows": 0, "files": 0}
    dates = sorted(df["trade_date"].unique().tolist())
    if not dry_run:
        save_daily_dataframe(
            base_dir,
            df,
            key_cols=key_cols,
            sort_cols=sort_cols,
            overwrite=overwrite,
        )
    return {"source": str(path), "rows": len(df), "files": len(dates)}


def migrate_regular_dataset(name, spec, config, args):
    base_dir = resolve_data_dir(config, spec["config_key"])
    total_rows = 0
    total_files = 0
    for path in yearly_files(base_dir, args.start_date, args.end_date):
        result = migrate_year_file(
            path,
            base_dir,
            spec["key_cols"],
            spec["sort_cols"],
            args.start_date,
            args.end_date,
            args.dry_run,
            args.overwrite,
        )
        total_rows += result["rows"]
        total_files += result["files"]
        print(f"{name}: {result['source']} rows={result['rows']} daily_files={result['files']}")
        delete_yearly_file(path, args)
    print(f"{name}: total rows={total_rows} daily_files={total_files}")


def migrate_index_daily(config, args):
    base_dir = resolve_data_dir(config, "index_daily_dir")
    total_rows = 0
    total_files = 0
    for code_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        for path in yearly_files(code_dir, args.start_date, args.end_date):
            result = migrate_year_file(
                path,
                code_dir,
                ["trade_date"],
                ["trade_date"],
                args.start_date,
                args.end_date,
                args.dry_run,
                args.overwrite,
            )
            total_rows += result["rows"]
            total_files += result["files"]
            print(
                f"index_daily/{code_dir.name}: {result['source']} "
                f"rows={result['rows']} daily_files={result['files']}"
            )
            delete_yearly_file(path, args)
    print(f"index_daily: total rows={total_rows} daily_files={total_files}")


def delete_yearly_file(path, args):
    if not args.delete_yearly:
        return
    if args.dry_run:
        print(f"dry-run delete yearly: {path}")
        return
    path.unlink()
    print(f"deleted yearly file: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate yearly daily parquet files to date files.")
    parser.add_argument(
        "--datasets",
        default=(
            "stock_daily_pv,stock_daily_basic,stock_adj_factor,stock_daily_limit,"
            "stock_suspend,stock_moneyflow,stock_st,future_daily,index_daily"
        ),
        help=(
            "Comma-separated datasets. Supported: "
            "stock_daily_pv,stock_daily_basic,stock_adj_factor,stock_daily_limit,"
            "stock_suspend,stock_moneyflow,stock_st,future_daily,index_daily"
        ),
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Accepted for CLI compatibility; existing daily files are merged by key.",
    )
    parser.add_argument(
        "--delete-yearly",
        action="store_true",
        help="Delete migrated YYYY.parquet files after a full-range migration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.delete_yearly and (args.start_date or args.end_date):
        raise ValueError("--delete-yearly is only allowed for full-range migrations")
    config = ConfigManager().config
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    for name in selected:
        if name == "index_daily":
            migrate_index_daily(config, args)
        elif name in DATASETS:
            migrate_regular_dataset(name, DATASETS[name], config, args)
        else:
            raise ValueError(f"unknown dataset: {name}")


if __name__ == "__main__":
    main()
