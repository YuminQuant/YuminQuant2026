from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section_rank(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = output.groupby("trade_date")[column].rank(pct=True)
    return output


def fill_feature_nan(frame: pd.DataFrame, columns: list[str], value: float = 0.0) -> pd.DataFrame:
    output = frame.copy()
    output[columns] = output[columns].replace([np.inf, -np.inf], np.nan).fillna(value)
    return output
