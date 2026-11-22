from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from yq_analysis.report_source import source_path, read_many


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_backtest_result(root: str | Path, factor_id: str) -> dict[str, pd.DataFrame | None]:
    """Load the standard Rust backtest outputs for one factor.

    Missing files are returned as None so callers can analyze just returns or IC.
    """

    root = source_path(root)
    factor_root = root / factor_id
    names = ["returns", "ic", "factor_stats", "barra_exposure", "index_group_returns"]
    paths = [factor_root / f"{name}.parquet" for name in names]
    frames = read_many(paths)
    return {name: frames[path] for name, path in zip(names, paths)}


def require_frame(result: dict[str, Any], key: str) -> pd.DataFrame:
    frame = result.get(key)
    if frame is None:
        raise FileNotFoundError(f"backtest result does not contain {key!r}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"backtest result {key!r} is not a DataFrame")
    return frame


def load_strategy_result(root: str | Path, strategy_id: str, *, storage=None) -> pd.DataFrame:
    """Read one strategy's account/holding snapshots; missing output is an error."""
    path = Path(root) / strategy_id / "holdings.parquet"
    exists = storage.exists if storage is not None else lambda p: p.exists()
    if not exists(path):
        raise FileNotFoundError(f"Strategy output not found: {path}")
    return storage.read_parquet(path) if storage is not None else pd.read_parquet(path)


def load_index_returns(
    root: str | Path, index_id: str, start_date: int, end_date: int,
) -> pd.DataFrame:
    """Local index close/pre_close returns, including the first selected day."""
    parts = []
    for year in range(start_date // 10000, end_date // 10000 + 1):
        path = Path(root) / index_id.replace(".", "_") / f"{year}.parquet"
        if path.exists():
            parts.append(pd.read_parquet(path, columns=["trade_date", "close", "pre_close"]))
    if not parts:
        raise FileNotFoundError(f"No local index data for {index_id}")
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame.trade_date.between(start_date, end_date)].sort_values("trade_date")
    if frame.trade_date.duplicated().any():
        raise ValueError(f"Duplicate benchmark dates: {index_id}")
    frame["return"] = frame["close"] / frame["pre_close"].where(frame["pre_close"] > 0) - 1
    return frame[["trade_date", "return"]].reset_index(drop=True)
