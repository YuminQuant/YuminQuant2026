"""Targets derived from observed returns; never used by prediction loaders."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_tail_labels(
    frame: pd.DataFrame, label_column: str, tail_fraction: float = 0.3,
) -> pd.Series:
    """Bottom/top return tails become 0/1; the middle and missing labels stay NaN.

    Midpoint percentiles keep equal returns in one class and avoid ordering
    ties by stock code. Consequently tied returns can change the tail count.
    """
    if not 0 < float(tail_fraction) < 0.5:
        raise ValueError("tail_fraction must be strictly between zero and 0.5")
    values = pd.to_numeric(frame[label_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = values.groupby(frame.trade_date)
    percentile = (grouped.rank(method="average") - 0.5) / grouped.transform("count")
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    result.loc[percentile <= tail_fraction] = 0.0
    result.loc[percentile >= 1.0 - tail_fraction] = 1.0
    return result
