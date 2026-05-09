from __future__ import annotations

import pandas as pd


def coverage(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return float(frame[column].notna().mean())
