import os
from pathlib import Path

import numpy as np
import pandas as pd


def daily_file_path(base_dir, trade_date):
    trade_date = str(int(trade_date))
    return os.path.join(base_dir, trade_date[:4], f"{trade_date}.parquet")


def list_local_daily_dates(base_dir):
    dates = set()
    if not os.path.exists(base_dir):
        return dates
    for year_dir in os.listdir(base_dir):
        year_path = os.path.join(base_dir, year_dir)
        if not os.path.isdir(year_path) or not year_dir.isdigit():
            continue
        for file_name in os.listdir(year_path):
            if file_name.endswith(".parquet"):
                dates.add(file_name[:-8])
    return dates


def save_daily_frames(
    base_dir,
    frames,
    key_cols,
    sort_cols=None,
    float32_cols=None,
    int32_cols=None,
    overwrite=False,
):
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    return save_daily_dataframe(
        base_dir,
        df,
        key_cols=key_cols,
        sort_cols=sort_cols,
        float32_cols=float32_cols,
        int32_cols=int32_cols,
        overwrite=overwrite,
    )


def save_daily_dataframe(
    base_dir,
    df,
    key_cols,
    sort_cols=None,
    float32_cols=None,
    int32_cols=None,
    overwrite=False,
):
    if df is None or df.empty:
        return 0
    df = df.copy()
    if "trade_date" not in df.columns:
        raise ValueError("daily dataframe must contain trade_date")

    df["trade_date"] = df["trade_date"].astype(np.int32)
    for col in int32_cols or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32").astype(np.int32)
    for col in float32_cols or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    written = 0
    sort_cols = sort_cols or key_cols
    for trade_date, df_day in df.groupby("trade_date", sort=True):
        file_path = Path(daily_file_path(base_dir, trade_date))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.exists() and not overwrite:
            df_old = pd.read_parquet(file_path)
            df_day = pd.concat([df_old, df_day], ignore_index=True)
        df_day.drop_duplicates(subset=key_cols, keep="last", inplace=True)
        df_day.sort_values(by=sort_cols, inplace=True)
        df_day.reset_index(drop=True, inplace=True)
        df_day.to_parquet(file_path, index=False)
        written += 1
    return written
