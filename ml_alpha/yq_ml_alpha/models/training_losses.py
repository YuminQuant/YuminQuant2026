"""Loss components shared by independent model architectures."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rank_weights(values: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if params["loss"] == "mse":
        return np.ones(len(values), dtype="float32")
    ranks = pd.Series(values).rank(method="average", ascending=False).to_numpy()
    percentile = (ranks - 1.0) / max(1, len(values) - 1)
    # Subtract the best rank before exponentiation to avoid all-zero underflow.
    weights = np.exp2(-(percentile - percentile.min()) / params["rank_weight_half_life"])
    return (weights / weights.mean()).astype("float32")


def _correlation_penalty(torch, base):
    """RMS off-diagonal correlation, computed within the current date batch."""
    centered = base - base.mean(dim=0, keepdim=True)
    normalized = centered / centered.square().mean(dim=0, keepdim=True).clamp_min(1e-12).sqrt()
    corr = normalized.T @ normalized / len(base)
    mask = ~torch.eye(base.shape[1], device=base.device, dtype=torch.bool)
    return (corr[mask].square().mean() + 1e-12).sqrt()
