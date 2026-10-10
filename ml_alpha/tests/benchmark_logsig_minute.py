"""Small, opt-in real-data comparison. All outputs go to a new scratch directory."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import numpy as np
    from yq_ml_alpha.calendar import TradingCalendar
    from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider

    data = Path(args.data).resolve()
    scratch = Path(args.scratch).resolve()
    dates = TradingCalendar.load(data).dates
    targets = [d for d in dates if args.start <= d <= args.end]
    source = args.worker
    root = data / "stock_data/minute" if source == "minute" else scratch / "data/derived/stock/bar/5m"
    provider = LogsigSignatureProvider(root, "__all__", dict(source=source, lookback_days=20,
        bar_size=5, order=10, feature_cache_dir=scratch / args.output, read_only=False))
    provider.set_calendar_dates(dates)
    provider.set_cache_days_for_target_dates(targets)
    summary = []
    start = time.perf_counter()
    for date in targets:
        frame = provider.load(date)
        assert len(frame) and np.isfinite(frame.iloc[:, 2:].to_numpy()).all()
        summary.append(dict(date=date, rows=len(frame), path=str(provider._feature_cache_path(date))))
        del frame
    (scratch / f"{args.output}.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start, dates=summary)))


def measure(command, log):
    import psutil
    start = time.perf_counter()
    peak = 0
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                info = monitor.memory_info()
                peak = max(peak, getattr(info, "peak_wset", info.rss))
            except psutil.NoSuchProcess:
                pass
            time.sleep(0.02)
        if process.returncode:
            raise RuntimeError(f"benchmark command failed: {log.read_text()}")
    return dict(seconds=time.perf_counter()-start, peak_rss_mib=peak/1024**2)


def benchmark(args):
    import numpy as np
    import pandas as pd
    from yq_ml_alpha.calendar import TradingCalendar

    scratch = Path(args.scratch).resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    data = Path(args.data).resolve()
    calendar = TradingCalendar.load(data)
    targets = calendar.between(args.start, args.end)
    if not 1 <= len(targets) <= 20:
        raise ValueError("benchmark requires 1..20 target trading days")
    warmup = calendar.offset(targets[0], -19)
    sources = calendar.between(warmup, targets[-1])
    # Hard links permit a genuinely isolated Rust derive-bar run without copying large raw files.
    for date in sources:
        relative = Path(f"stock_data/minute/{date//10000}/{date}.parquet")
        target = scratch / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(data / relative, target)
    cal = scratch / "data/calendar/trade_cal_SSE.parquet"
    cal.parent.mkdir(parents=True)
    os.link(data / "calendar/trade_cal_SSE.parquet", cal)
    config = scratch / "config.toml"
    config.write_text('[paths]\nbase_data_dir = "' + (scratch / "data").as_posix() + '"\n')
    result = dict(target_dates=targets, source_days=len(sources))
    result["derive_bar"] = measure([args.engine, "derive-bar", "--asset", "stock", "--source", "minute",
        "--bar-size", "5", "--columns", "volume", "--start-date", str(warmup), "--end-date", str(targets[-1]),
        "--date-batch-size", "1", "--project-config", str(config)], scratch / "derive.log")
    # Reverse order in the second round to reduce file-cache/order bias.
    for source, output in [("bar", "old1"), ("minute", "new1"), ("minute", "new2"), ("bar", "old2")]:
        result[output] = measure([sys.executable, __file__, "--worker", source, "--output", output,
            "--data", str(data), "--scratch", str(scratch), "--start", str(args.start), "--end", str(args.end)], scratch / f"{output}.log")
    reference = json.loads((scratch / "old1.json").read_text())["dates"]
    max_diff = 0.
    for name in ["new1", "new2", "old2"]:
        candidate = json.loads((scratch / f"{name}.json").read_text())["dates"]
        for old, new in zip(reference, candidate):
            a, b = pd.read_parquet(old["path"]), pd.read_parquet(new["path"])
            pd.testing.assert_frame_equal(a.iloc[:, :2], b.iloc[:, :2])
            x, y = a.iloc[:, 2:].to_numpy(), b.iloc[:, 2:].to_numpy()
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)
            max_diff = max(max_diff, float(np.max(np.abs(x-y))))
    result.update(max_absolute_difference=max_diff, rows=[(v["date"], v["rows"]) for v in reference])
    (scratch / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--engine", default="factor_engine/target/release/yq-factor-engine.exe")
    parser.add_argument("--start", type=int, default=20110104)
    parser.add_argument("--end", type=int, default=20110110)
    parser.add_argument("--worker", choices=["bar", "minute"])
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        benchmark(args)
