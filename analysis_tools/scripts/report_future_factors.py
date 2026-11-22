"""Build futures factor reports from local files or authenticated direct HDFS reads."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis_tools"
sys.path.insert(0, str(ANALYSIS))

from yq_analysis.future_io import load_future_backtest_result, load_future_sectors
from yq_analysis.future_plots import plot_future_return_summary
from yq_analysis.future_report import make_future_backtest_report
from yq_analysis.metrics import clean_series, sharpe
from yq_analysis.future_diagnostics import distribution_diagnostics
from yq_analysis.report_source import initialize_hdfs, hdfs_config, source_path, read_parquet

SUMMARY_ROOT = ROOT / "data/backtest/future/minute_1m"
PLOTS_ROOT = ANALYSIS / "plots/future"
REPORT_PATH = ROOT / "data/reports/future_factor_review.html"
METADATA = ROOT / "data/factors/factor_metadata.parquet"
SECTORS = ROOT / "factor_engine/docs/FUTURE_SECTORS.md"
TEMPLATE = ANALYSIS / "templates/future_factor_review.html"
FACTOR_ID = re.compile(r"yq_\d{3,}")
PAYLOAD = re.compile(r'<script type="application/json" id="report-data">([\s\S]*?)</script>')
# Specialized tags take precedence over broad input or activity categories.
FAMILY_TAGS = {
    "cci": "CCI",
    "wait_and_see": "观望",
    "reversal_continuation": "反转延续",
    "runs": "游程",
    "hurst": "长记忆",
    "ljung_box": "Ljung-Box",
    "variance_ratio": "方差比",
    "snr": "信噪比",
    "skewness": "偏度",
    "aroon": "Aroon",
    "rsi": "RSI",
    "volatility": "波动率",
    "volume_open_interest": "量仓",
    "open_interest": "持仓量",
    "volume": "成交量",
    "price_volume": "量价",
}
GENERIC_TAGS = {"minute", "minute_1m", "future", "pv", "yq", "timing"}
TRADING_DAYS = 240
MIN_COMMON = 120
MIN_CORRELATION = 0.90


def factor_number(factor):
    return int(factor.split("_", 1)[1])


def family(tags):
    tags = {tag.strip().lower() for tag in tags if tag.strip()}
    for tag, name in FAMILY_TAGS.items():
        if tag in tags:
            return name
    return "其他" if tags - GENERIC_TAGS else "未分类"


def read_factor_metadata(path):
    if not path.is_file():
        return {}
    columns = ["asset_class", "frequency", "factor_id", "description", "aliases_json", "tags_json"]
    frame = read_parquet(path, columns=columns)
    futures = frame[frame.asset_class.eq("future") & frame.frequency.eq("minute_1m")]
    return {
        row.factor_id: {
            "description": row.description or "",
            "aliases": json.loads(row.aliases_json or "[]"),
            "tags": json.loads(row.tags_json or "[]"),
        }
        for row in futures.itertuples(index=False)
    }


def factor_catalog(path, factor_ids, work):
    catalog = read_factor_metadata(path)
    missing = factor_ids - catalog.keys()
    if missing and path == METADATA:
        executable = "yq-factor-engine.exe" if os.name == "nt" else "yq-factor-engine"
        targets = ROOT / "factor_engine/target"
        candidates = [targets / mode / executable for mode in ("release", "debug")]
        binary = max((p for p in candidates if p.is_file()), key=lambda p: p.stat().st_mtime_ns, default=None)
        if binary is not None:
            staged = work / "factor-catalog"
            staged.mkdir()
            config = staged / "config.toml"
            config.write_text("base_data_dir = " + json.dumps(staged.as_posix(), ensure_ascii=False), encoding="utf-8")
            try:
                subprocess.run([str(binary), "metadata", "--config", str(config)], check=True,
                               cwd=staged, capture_output=True, timeout=30,
                               env={**os.environ, "YQ_STORAGE_BACKEND": "local"})
                fallback = read_factor_metadata(staged / "factors/factor_metadata.rust.parquet")
                catalog.update({factor: fallback[factor] for factor in missing & fallback.keys()})
            except (OSError, subprocess.SubprocessError) as error:
                print(f"无法从本地引擎补全因子元数据：{error}", file=sys.stderr)
    missing = factor_ids - catalog.keys()
    if missing:
        print(f"{len(missing)} 个因子缺少元数据，已归为未分类。可用 --metadata 指定最新元数据。",
              file=sys.stderr)
    return catalog


def finite(value):
    return round(float(value), 8) if math.isfinite(value) else None


def records(frame):
    return json.loads(frame.reset_index().to_json(orient="records", double_precision=8, force_ascii=False))


def sector_catalog(sectors):
    """Embed the canonical bilingual mapping, independently of backtest coverage."""
    return [
        {"abbr": str(abbr), "chinese": group["name"].iloc[0],
         "english": group["sector"].iloc[0],
         "products": group[["product", "chinese", "english", "exchange"]].to_dict("records")}
        for abbr, group in sectors.groupby("abbr", observed=True, sort=False)
    ]


def return_metrics(daily_bp):
    returns = daily_bp / 10_000
    nav = 1 + returns.cumsum()
    peak = nav.cummax().clip(lower=1.0)
    return {
        "annual_return": finite(100 * returns.mean() * TRADING_DAYS),
        "sharpe": finite(sharpe(returns, periods_per_year=TRADING_DAYS)),
        "max_drawdown": finite(100 * (nav / peak - 1).min()),
        "days": len(returns),
    }


def fingerprint(path):
    stat = path.stat()
    stats = path.with_name("factor_stats.parquet")
    detail = stats.stat() if stats.is_file() else None
    return [stat.st_size, stat.st_mtime_ns, detail.st_size if detail else None, detail.st_mtime_ns if detail else None]


def settle(factor, path, sectors, plots, work, plot_fresh):
    result = load_future_backtest_result(path.parent.parent, factor)
    report = make_future_backtest_report(result["summary"], sectors)
    market = report["market"]
    if market.empty:
        raise ValueError("摘要缺少全品种日汇总")
    # Use the same All IC as the panel in the existing plot.
    all_metrics = report["sector_ic"].loc["All"]
    ic = all_metrics["ic"]
    market_net = clean_series(report["market_net"])
    sector_returns = report["sector"].rename_axis("sector").join(report["trade_num"])
    directed = report["directed"]
    product_returns = directed.assign(net=directed.sum(axis=1, min_count=2)).join(report["product_trade_num"])
    if report.get("holding_returns"):
        product_returns["net"] = report["daily_product"].groupby("ts_code").mean()
    stage = None
    plot = plots / f"{factor}.png"
    if not plot_fresh:
        stage = work / plot.name
        plot_future_return_summary(report, title=f"{factor} | {result['quantile_mode']}", save=stage)
    tags = ["label_id", "quantile_mode", "tail_quantile", "pool_id"]
    entry = {
        "id": factor, "fingerprint": fingerprint(path), "configuration": [result[k] for k in tags],
        "first": int(market.index.min()), "last": int(market.index.max()),
        "ic": finite(ic), "abs_ic": finite(abs(ic)), "cosic": finite(all_metrics["cosic"]),
        "total_days": len(market), "coverage": finite(100 * len(market_net) / len(market)),
        "daily": [[int(day), finite(value)] for day, value in market_net.items()],
        "sectors": records(sector_returns),
        "products": records(product_returns),
        "holding_label": result["holding_label_id"],
        "distribution": distribution_diagnostics(path.with_name("factor_stats.parquet")),
        **return_metrics(market_net),
    }
    return entry, stage


def correlations(entries):
    if len(entries) < 2:
        return {"pairs": [], "count": 0}
    returns = pd.DataFrame({
        x["id"]: pd.Series(dict(x["daily"]), dtype=float)
        for x in entries
    })
    matrix = returns.corr(min_periods=MIN_COMMON).to_numpy()
    pairs = np.argwhere(np.triu(np.abs(matrix) >= MIN_CORRELATION, k=1))
    ranked = sorted(pairs.tolist(), key=lambda p: -abs(matrix[p[0], p[1]]))
    available = returns.notna().to_numpy()
    return {
        "count": len(ranked),
        "pairs": [
            {"left": returns.columns[i], "right": returns.columns[j], "correlation": finite(matrix[i, j]),
             "days": int(np.count_nonzero(available[:, i] & available[:, j]))}
            for i, j in ranked[:40]
        ],
    }


def cache_key(summaries):
    paths = [Path(__file__), SECTORS, *(ANALYSIS / "yq_analysis" / f"{name}.py" for name in
             ("future_report", "future_io", "metrics", "future_plots", "future_diagnostics"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return [str(summaries), digest.hexdigest()]


def prior_entries(output, key):
    if output.is_file():
        match = PAYLOAD.search(output.read_text(encoding="utf-8"))
        if match:
            try:
                payload = json.loads(match.group(1))
                if payload.get("cache_key") == key:
                    return {entry["id"]: entry for entry in payload["factors"]}
            except (ValueError, KeyError, TypeError):
                pass
    return {}


def publish(output, html):
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                prefix=".future-review-", suffix=".tmp", delete=False) as f:
            pending = Path(f.name)
            f.write(html)
        os.replace(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def plot_link(plot, report):
    try:
        return Path(os.path.relpath(plot, report.parent)).as_posix()
    except ValueError:
        return plot.as_uri()


def generate(summaries, output, plots, metadata, rebuild, workers=4):
    files = {p.parent.name: p for p in summaries.glob("yq_*/summary.parquet")
             if FACTOR_ID.fullmatch(p.parent.name)}
    if not files:
        raise ValueError(f"没有找到因子回测摘要：{summaries}")
    with ThreadPoolExecutor(max_workers=50) as reader:
        fingerprints = dict(zip(files, reader.map(fingerprint, files.values())))
    key = cache_key(summaries)
    previous = {} if rebuild else prior_entries(output, key)
    sectors = load_future_sectors(SECTORS)
    tmp = ROOT / "tmp"
    tmp.mkdir(exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(tmp / "future-factor-review-mpl"))
    failed, entries = [], []
    reused, updated = 0, 0
    plots.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="future-factor-review-", dir=tmp) as directory, ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"), initializer=initialize_hdfs, initargs=(hdfs_config(),)) as executor:
        work = Path(directory)
        catalog = factor_catalog(metadata, set(files), work)
        jobs = {}
        for factor, path in sorted(files.items(), key=lambda p: factor_number(p[0])):
            current = fingerprints[factor]
            plot = plots / f"{factor}.png"
            plot_fresh = not rebuild and plot.is_file() and plot.stat().st_mtime_ns >= current[1]
            saved = previous.get(factor)
            if not (saved and saved["fingerprint"] == current and plot_fresh):
                jobs[factor] = executor.submit(settle, factor, path, sectors, plots, work, plot_fresh)
        for factor, path in sorted(files.items(), key=lambda p: factor_number(p[0])):
            plot = plots / f"{factor}.png"
            saved = previous.get(factor)
            current = fingerprints[factor]
            plot_fresh = (not rebuild and plot.is_file() and plot.stat().st_mtime_ns >= current[1])
            if saved and saved["fingerprint"] == current and plot_fresh:
                entry = saved
                reused += 1
            else:
                try:
                    entry, stage = jobs[factor].result()
                    if entry["fingerprint"] != current:
                        raise ValueError("摘要在读取期间发生变化")
                    if stage is not None:
                        pending = plot.with_suffix(".png.tmp")
                        try:
                            pending.write_bytes(stage.read_bytes())
                            os.replace(pending, plot)
                        finally:
                            pending.unlink(missing_ok=True)
                    updated += 1
                    if updated % 50 == 0:
                        print(f"已结算 {updated} 个新增或变化的因子", flush=True)
                except (OSError, ValueError, KeyError) as error:
                    failed.append({"id": factor, "reason": str(error)})
                    print(f"{factor}: {error}", file=sys.stderr)
                    continue
            meta = catalog.get(factor, {"description": "", "aliases": [], "tags": []})
            entry.update(family=family(meta["tags"]), plot=plot_link(plot, output), **meta)
            scope = next((row for row in entry.get("distribution", {}).get("scopes", []) if row["ts_code"] == "all"), {})
            coverage = scope.get("coverage")
            entry["minute_coverage"] = finite(100 * coverage) if coverage is not None else None
            entries.append(entry)
        if not entries:
            raise ValueError("没有可结算的期货因子")
        groups = {}
        for entry in entries:
            group = " / ".join(str(x) if x else "未记录" for x in entry["configuration"])
            entry["group"] = group
            groups.setdefault(group, []).append(entry)
        payload = {
            "generated": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"),
            "cache_key": key, "factors": entries, "failed": failed,
            "sector_catalog": sector_catalog(sectors),
            "groups": [{"name": name, "size": len(rows), **correlations(rows)}
                       for name, rows in sorted(groups.items(), key=lambda p: (-len(p[1]), p[0]))],
            "min_common": MIN_COMMON, "min_correlation": MIN_CORRELATION,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        html = TEMPLATE.read_text(encoding="utf-8").replace("__REPORT_PAYLOAD__", encoded.replace("<", "\\u003c"))
        publish(output, html)
    print(f"{output} | 共 {len(entries)} 个因子，更新 {updated}，复用 {reused}，跳过 {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--plots-root", type=Path, default=PLOTS_ROOT)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--hdfs-config", type=Path, help="Authenticated deployment config; read inputs directly from HDFS")
    parser.add_argument("--rebuild", action="store_true", help="重新结算全部摘要并重画全部图")
    parser.add_argument("--workers", type=int, default=4, help="结算和绘图进程数，默认4，可按机器资源调整")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers 必须大于零")
    initialize_hdfs(args.hdfs_config)
    summaries = source_path(args.summary_root if args.summary_root != SUMMARY_ROOT else "data/backtest/future/minute_1m") if args.hdfs_config else args.summary_root.resolve()
    metadata = source_path(args.metadata if args.metadata != METADATA else "data/factors/factor_metadata.parquet") if args.hdfs_config else args.metadata.resolve()
    generate(summaries, args.output.resolve(), args.plots_root.resolve(), metadata, args.rebuild, args.workers)


if __name__ == "__main__":
    main()
