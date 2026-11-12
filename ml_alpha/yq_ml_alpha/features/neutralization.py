"""Cross-sectional neutralization backed by the factor engine."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def apply_rust_neutralization(data: pd.DataFrame, score: pd.Series, spec: str) -> pd.Series:
    spec = str(spec or "none").strip()
    if not spec or spec.lower() == "none" or data.empty:
        return score.astype("float32")
    rust = _import_rust_neutralizer()
    trade_dates = pd.to_numeric(data["trade_date"], errors="raise").astype("int32")
    ts_codes = data["ts_code"].astype(str)
    score_values = score.astype("float64").to_numpy()
    payload = [
        float(value) if np.isfinite(value) else None
        for value in score_values
    ]
    result = rust.neutralize_daily(
        trade_dates.tolist(),
        ts_codes.tolist(),
        payload,
        spec,
        "stock",
        "daily",
        int(trade_dates.min()),
        int(trade_dates.max()),
        None,
    )
    if len(result) != len(score):
        raise ValueError(
            f"Rust neutralize_daily returned {len(result)} rows, expected {len(score)}"
        )
    values = [np.nan if value is None else float(value) for value in result]
    return pd.Series(values, index=score.index, dtype="float32")


def _import_rust_neutralizer():
    try:
        return importlib.import_module("yq_factor_engine_py")
    except ImportError as exc:
        raise ImportError(
            "Cross-sectional neutralization requires the factor_engine Rust "
            "Python extension `yq_factor_engine_py`. Build it from the repository root with:\n"
            "python -m pip install maturin\n"
            "python -m maturin develop --manifest-path "
            "factor_engine/python/yq_factor_engine_py/Cargo.toml"
        ) from exc
