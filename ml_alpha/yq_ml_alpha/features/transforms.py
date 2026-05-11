from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd


TransformFn = Callable[[pd.DataFrame, List[str], List[str], float], pd.DataFrame]
TRANSFORM_REGISTRY: dict[str, TransformFn] = {}


def _normalize_transform_name(name: str) -> str:
    return str(name).strip().lower()


def register_transform(*names: str) -> Callable[[TransformFn], TransformFn]:
    def decorator(func: TransformFn) -> TransformFn:
        for name in names:
            TRANSFORM_REGISTRY[_normalize_transform_name(name)] = func
        return func

    return decorator


def apply_cross_section_transform(
    frame: pd.DataFrame,
    transform_name: str,
    feature_columns: list[str],
    label_columns: list[str] | None = None,
    feature_fill_value: float = 0.0,
) -> pd.DataFrame:
    name = _normalize_transform_name(transform_name)
    transform = TRANSFORM_REGISTRY.get(name)
    if transform is None:
        supported = ", ".join(available_transforms())
        raise ValueError(f"unsupported preprocess.cross_section_transform: {transform_name}; supported: {supported}")
    return transform(frame, feature_columns, label_columns or [], feature_fill_value)


def available_transforms() -> list[str]:
    return sorted(TRANSFORM_REGISTRY)


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


@register_transform("", "none")
def _transform_none(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    feature_fill_value: float,
) -> pd.DataFrame:
    return fill_feature_nan(frame, feature_columns, feature_fill_value)


@register_transform("zscore_log_rank", "log_rank_zscore", "cs_zscore_log_rank")
def _transform_zscore_log_rank(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    feature_fill_value: float,
) -> pd.DataFrame:
    return cross_section_zscore_log_rank(
        frame,
        [*feature_columns, *label_columns],
        fill_columns=feature_columns,
        fill_value=feature_fill_value,
    )


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
