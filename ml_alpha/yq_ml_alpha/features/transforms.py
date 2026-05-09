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


def cross_section_zscore_log_rank(
    frame: pd.DataFrame,
    columns: list[str],
    fill_columns: list[str] | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            continue
        output[column] = output.groupby("trade_date", group_keys=False)[column].transform(_zscore_log_rank_series)
    if fill_columns:
        existing = [column for column in fill_columns if column in output.columns]
        output[existing] = output[existing].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return output


def _zscore_log_rank_series(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", na_option="keep", ascending=True)
    transformed = np.log(ranks)
    valid = transformed.notna()
    if not valid.any():
        return transformed
    mean = transformed.loc[valid].mean()
    std = transformed.loc[valid].std(ddof=0)
    if not np.isfinite(std) or std <= 0.0:
        transformed.loc[valid] = 0.0
        return transformed
    transformed.loc[valid] = (transformed.loc[valid] - mean) / std
    return transformed
