"""Resolve local and xz336/HDFS inputs for a CTA intraday report."""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from yq_analysis.strategy import futures_products


_MAPPING_COLUMNS = ["trade_date", "ts_code", "mapping_ts_code"]


def prior_trading_days(calendar_path, dates, count):
    calendar = pd.read_parquet(calendar_path, columns=["cal_date", "is_open"])
    open_dates = sorted(set(map(int, calendar.loc[calendar.is_open.eq(1), "cal_date"])))
    previous = {}
    for date in dates:
        index = bisect_left(open_dates, date)
        if index == len(open_dates) or open_dates[index] != date:
            raise ValueError(f"Trading calendar {calendar_path} has no open session for {date}")
        previous[date] = open_dates[max(0, index - count):index]
    return previous


def _mapping_for_days(folder, dates, product):
    folder = Path(folder)
    matches = {}
    for year in sorted({date // 10000 for date in dates}):
        year_path = folder / f"{year}.parquet"
        years = pd.read_parquet(year_path, columns=_MAPPING_COLUMNS) if year_path.is_file() else None
        if years is not None:
            matches.update(_mapping_rows(years, product, dates))
        for date in (day for day in dates if day // 10000 == year and day not in matches):
            path = folder / str(year) / f"{date}.parquet"
            if path.is_file():
                matches.update(_mapping_rows(pd.read_parquet(path, columns=_MAPPING_COLUMNS), product, [date]))
    return matches


def _mapping_rows(table, product, dates):
    rows = table.loc[table.ts_code.str.split(".").str[0].eq(product)].dropna().drop_duplicates()
    days = pd.to_numeric(rows.trade_date.astype(str).str.replace("-", "").str[:8], errors="raise")
    rows = rows.assign(trade_date=days).loc[lambda p: p.trade_date.isin(dates)]
    rows = rows.loc[rows.mapping_ts_code.str.fullmatch(re.escape(product) + r"[0-9]{3,4}[.][A-Z]+")]
    if rows.trade_date.duplicated().any():
        raise ValueError(f"Multiple main contracts for {product} on a trading day")
    return dict(zip(rows.trade_date.astype(int), rows.mapping_ts_code))


def _local_signals(root, signal, product, dates):
    columns = ["ts_code", "trade_time", signal]
    results = {}
    for date in dates:
        path = Path(root) / str(date // 10000) / f"{date}.parquet"
        if path.is_file():
            rows = pd.read_parquet(path, columns=columns)
            codes = futures_products(pd.Index(rows.ts_code))
            results[date] = rows.loc[codes == product]
    return results


_HDFS_SCRIPT = r"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, sys, tomllib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from yq_cluster.auth import renewing_kerberos
from yq_cluster.paths import resolve
from yq_cluster.storage import Hdfs

p = Path.cwd()
cfg = tomllib.loads((p / "cluster.toml").read_text())
routes = resolve(cfg)
with renewing_kerberos(cfg["hdfs"], p / "runtime") as (env, ticket):
    store = Hdfs(cfg["hdfs"], env)
    def read(path, columns):
        if not store.exists(path):
            return None
        with store.open_input(path) as source:
            return pq.ParquetFile(source).read(columns=columns).to_pandas()

    fields = ["trade_date", "ts_code", "mapping_ts_code"]
    base = routes["input_hdfs_prefix"] + "/future_data/basic/mapping"
    yearly = {}
    for year in sorted({date // 10000 for date in request["mapping_dates"]}):
        rows = read(f"{base}/{year}.parquet", fields)
        if rows is not None:
            rows["trade_date"] = pd.to_numeric(rows.trade_date.astype(str).str.replace("-", "").str[:8])
        yearly[year] = rows

    def mapping(date):
        rows = yearly[date // 10000]
        if rows is not None:
            found = rows.loc[rows.trade_date.eq(date)]
            if not found.empty:
                rows = found
            else:
                rows = None
        if rows is None:
            rows = read(f"{base}/{date//10000}/{date}.parquet", fields)
        if rows is None:
            return date, []
        return date, rows.loc[rows.ts_code.str.split(".").str[0].eq(request["product"])].values.tolist()

    columns = ["ts_code", "trade_time", request["signal"]]
    model_root = routes["outputs"]["signals"] + "/future/minute_1m"
    def predictions(date):
        rows = read(f"{model_root}/{date//10000}/{date}.parquet", columns)
        if rows is None:
            return date, []
        products = rows.ts_code.str.extract(r"^([A-Za-z]+)", expand=False).str.upper()
        chosen = rows.loc[products.eq(request["product"]), columns].copy()
        chosen = chosen.astype(object).where(pd.notna(chosen), None)
        return date, chosen.values.tolist()

    workers = max(1, min(50, cfg["hdfs"].get("read_concurrency", 50)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        output = {"mapping": dict(pool.map(mapping, request["mapping_dates"])),
                  "predictions": dict(pool.map(predictions, request["signal_dates"]))}
    print(json.dumps(output, default=str))
"""


def _read_hdfs(host, request):
    command = 'cd "$HOME/YuminQuant/YuminQuant2026_private" && "$HOME/miniforge3/bin/python" -B -'
    code = "import json\nrequest = json.loads(" + repr(json.dumps(request)) + ")\n" + _HDFS_SCRIPT
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15", "--", host, command],
        input=code.encode(), stdout=subprocess.PIPE, check=True,
    )
    return json.loads(result.stdout)


def load_intraday_inputs(signal, product, dates, *, signal_root, mapping_root, hdfs_host):
    """Read main-contract mappings and raw scores for display and warm-up dates."""
    dates = sorted(set(map(int, dates)))
    mapping = _mapping_for_days(mapping_root, dates, product)
    frames = _local_signals(signal_root, signal, product, dates)
    request = {
        "signal": signal, "product": product,
        "mapping_dates": [date for date in dates if date not in mapping],
        "signal_dates": [date for date in dates if date not in frames],
    }
    if hdfs_host and (request["mapping_dates"] or request["signal_dates"]):
        remote = _read_hdfs(hdfs_host, request)
        for date, rows in remote["mapping"].items():
            mapping.update(_mapping_rows(pd.DataFrame(rows, columns=_MAPPING_COLUMNS), product, [int(date)]))
        for date, rows in remote["predictions"].items():
            frames[int(date)] = pd.DataFrame(rows, columns=["ts_code", "trade_time", signal])
    scores = {}
    for date, frame in frames.items():
        if frame.duplicated(["ts_code", "trade_time"]).any():
            raise ValueError(f"Duplicate prediction key for {signal} on {date}")
        values = pd.to_numeric(frame[signal], errors="coerce")
        scores[date] = dict(zip(zip(frame.ts_code, frame.trade_time), values.where(np.isfinite(values))))
    return mapping, scores
