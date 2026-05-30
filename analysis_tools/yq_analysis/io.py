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
