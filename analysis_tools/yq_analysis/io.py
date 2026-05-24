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
    return {
        "returns": _read_optional(root / "returns" / f"{factor_id}.parquet"),
        "ic": _read_optional(root / "ic" / f"{factor_id}.parquet"),
        "factor_stats": _read_optional(root / "factor_stats" / f"{factor_id}.parquet"),
        "barra_exposure": _read_optional(root / "barra_exposure" / f"{factor_id}.parquet"),
    }


def require_frame(result: dict[str, Any], key: str) -> pd.DataFrame:
    frame = result.get(key)
    if frame is None:
        raise FileNotFoundError(f"backtest result does not contain {key!r}")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"backtest result {key!r} is not a DataFrame")
    return frame
