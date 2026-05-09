from __future__ import annotations

import subprocess
from pathlib import Path


def run_rust_backtest(
    factor_id: str,
    start_date: int,
    end_date: int,
    factor_root: str | Path = "data/models",
    groups: int = 10,
    rebalance: str = "5",
) -> subprocess.CompletedProcess:
    command = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        "factor_engine/Cargo.toml",
        "--",
        "backtest",
        "--asset",
        "stock",
        "--frequency",
        "daily",
        "--start-date",
        str(start_date),
        "--end-date",
        str(end_date),
        "--factors",
        factor_id,
        "--factor-root",
        str(factor_root),
        "--groups",
        str(groups),
        "--rebalance",
        rebalance,
    ]
    return subprocess.run(command, check=True)
