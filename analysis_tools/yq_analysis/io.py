from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _read_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_backtest_result(root: str | Path, factor_id: str) -> dict[str, pd.DataFrame | None]:
    """Load the standard Rust backtest outputs for one factor.

    Missing files are returned as None so callers can analyze just returns or IC.
    """

    root = Path(root)
    factor_root = root / factor_id
    return {
        "returns": _read_optional(factor_root / "returns.parquet"),
        "ic": _read_optional(factor_root / "ic.parquet"),
        "factor_stats": _read_optional(factor_root / "factor_stats.parquet"),
        "barra_exposure": _read_optional(factor_root / "barra_exposure.parquet"),
        "index_group_returns": _read_optional(factor_root / "index_group_returns.parquet"),
    }


def require_frame(result: dict[str, Any], key: str) -> pd.DataFrame:
    frame = result.get(key)
    if frame is None:
        raise FileNotFoundError(f"backtest result does not contain {key!r}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"backtest result {key!r} is not a DataFrame")
    return frame


def load_strategy_result(root: str | Path, strategy_id: str) -> pd.DataFrame:
    """Read one strategy's account/holding snapshots; missing output is an error."""
    path = Path(root) / strategy_id / "holdings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Strategy output not found: {path}")
    return pd.read_parquet(path)


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
