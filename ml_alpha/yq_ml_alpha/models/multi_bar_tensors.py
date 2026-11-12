"""Tensor access for models combining daily and intraday bars."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _multi_bar_tensors(
    frame: pd.DataFrame,
    columns: list[str],
    rows: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    daily_columns = _prefixed_columns(columns, "daily")
    minute_columns = _prefixed_columns(columns, "minute")
    daily = _panel_tensor(
        frame,
        daily_columns,
        rows,
        int(params["daily_sequence_length"]),
        int(params["input_size"]),
        "daily",
    )
    minute = _panel_tensor(
        frame,
        minute_columns,
        rows,
        int(params["minute_sequence_length"]),
        int(params["input_size"]),
        "minute",
    )
    return daily, minute


def _multi_bar_batch(
    frame: pd.DataFrame,
    columns: list[str],
    rows: np.ndarray,
    params: dict[str, Any],
    tensors: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if tensors is None:
        return _multi_bar_tensors(frame, columns, rows, params)
    return (
        _branch_tensor_from_array(tensors["daily"], rows, int(params["daily_sequence_length"]), int(params["input_size"]), "daily"),
        _branch_tensor_from_array(tensors["minute"], rows, int(params["minute_sequence_length"]), int(params["input_size"]), "minute"),
    )


def _branch_tensor_from_array(
    tensor: np.ndarray,
    rows: np.ndarray,
    sequence_length: int,
    input_size: int,
    label: str,
) -> np.ndarray:
    if tensor.ndim != 3 or tensor.shape[1] != sequence_length or tensor.shape[2] != input_size:
        raise ValueError(f"{label} branch expects tensor shape [N,{sequence_length},{input_size}], got {tensor.shape}")
    output = np.ascontiguousarray(tensor[np.asarray(rows, dtype=np.int64)], dtype="float32")
    if not np.isfinite(output).all():
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    return output


def _require_multi_bundle_tensors(bundle) -> dict[str, np.ndarray]:
    tensors = getattr(bundle, "tensors", None)
    if not tensors or "daily" not in tensors or "minute" not in tensors:
        raise ValueError("MultiBarGRUAlphaModel requires DatasetBundle.tensors['daily'] and ['minute']")
    return tensors


def _optional_multi_bundle_tensors(bundle) -> dict[str, np.ndarray] | None:
    if bundle is None:
        return None
    tensors = getattr(bundle, "tensors", None)
    if not tensors:
        return None
    if "daily" not in tensors or "minute" not in tensors:
        return None
    return tensors


def _prefixed_columns(columns: list[str], prefix: str) -> list[str]:
    wanted = f"{prefix}__"
    return [column for column in columns if column.startswith(wanted)]


def _panel_tensor(
    frame: pd.DataFrame,
    columns: list[str],
    rows: np.ndarray,
    sequence_length: int,
    input_size: int,
    label: str,
) -> np.ndarray:
    expected = sequence_length * input_size
    if len(columns) != expected:
        raise ValueError(f"{label} branch expects {expected} feature columns, got {len(columns)}")
    flat = (
        frame.loc[rows, columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    return flat.reshape(flat.shape[0], sequence_length, input_size).astype("float32", copy=False)
