"""Parameter validation, labels and loss weighting for multi-task models."""
from __future__ import annotations

import numpy as np


def normalize_multi_task_params(raw):
    params = dict(hidden_size=256, weighting="dwa", negative_slope=0.01,
                  rank_weight_half_life=0.5, dwa_temperature=2.0,
                  lr=0.0001, weight_decay=0.0, epochs=200, patience=20,
                  batch_size=3000, seed=42, min_delta=1e-12, device="auto")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown multi-task parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("hidden_size", "epochs", "patience", "batch_size", "seed"):
        value = params[key]
        params[key] = int(value)
        if isinstance(value, bool) or params[key] != value or (key != "seed" and params[key] <= 0):
            raise ValueError(f"invalid integer multi-task parameter {key}")
    for key in ("negative_slope", "rank_weight_half_life", "dwa_temperature", "lr", "weight_decay", "min_delta"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if any(params[key] <= 0 for key in ("rank_weight_half_life", "dwa_temperature", "lr")):
        raise ValueError("rank_weight_half_life, dwa_temperature and lr must be positive")
    if params["weighting"] not in {"uw", "dwa"}:
        raise ValueError("multi-task weighting must be uw or dwa")
    return params


def _label_columns(context):
    columns = list(context.label_columns or [context.label_column])
    if (len(columns) < 2 or len(set(columns)) != len(columns)
            or columns[0] != context.label_column):
        raise ValueError("multi-task training requires distinct ordered primary and auxiliary labels")
    return columns


def _dwa_weights(history, task_count, temperature):
    # Initial two EPOCHS use equal weights; ratios use unweighted per-task losses.
    if len(history) < 2:
        return np.ones(task_count, dtype="float32")
    ratio = history[-1] / np.maximum(history[-2], 1e-12)
    logits = ratio / temperature
    scaled = np.exp(logits - logits.max())
    return (task_count * scaled / scaled.sum()).astype("float32")
