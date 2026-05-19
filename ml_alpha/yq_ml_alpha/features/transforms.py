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


def cross_section_zscore_erfinv_rank(
    frame: pd.DataFrame,
    columns: list[str],
    fill_columns: list[str] | None = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            continue
        output[column] = output.groupby("trade_date", group_keys=False)[column].transform(
            _zscore_erfinv_rank_series
        )
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


@register_transform("zscore", "cs_zscore")
def _transform_zscore(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    feature_fill_value: float,
) -> pd.DataFrame:
    output = frame.copy()
    for column in [*feature_columns, *label_columns]:
        if column not in output.columns:
            continue
        output[column] = output.groupby("trade_date", group_keys=False)[column].transform(_zscore_series)
    return fill_feature_nan(output, feature_columns, feature_fill_value)


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


@register_transform("zscore_erfinv_rank", "zscore_inverf_rank", "erfinv_rank_zscore", "rank_gauss")
def _transform_zscore_erfinv_rank(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    feature_fill_value: float,
) -> pd.DataFrame:
    return cross_section_zscore_erfinv_rank(
        frame,
        [*feature_columns, *label_columns],
        fill_columns=feature_columns,
        fill_value=feature_fill_value,
    )


def _zscore_log_rank_series(values: pd.Series) -> pd.Series:
    values = values.replace([np.inf, -np.inf], np.nan)
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


def _zscore_series(values: pd.Series) -> pd.Series:
    transformed = values.replace([np.inf, -np.inf], np.nan).astype("float64")
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


def _zscore_erfinv_rank_series(values: pd.Series) -> pd.Series:
    values = values.replace([np.inf, -np.inf], np.nan)
    valid = values.notna()
    transformed = pd.Series(np.nan, index=values.index, dtype="float64")
    n = int(valid.sum())
    if n == 0:
        return transformed
    ranks = values.loc[valid].rank(method="average", ascending=True)
    percentile = (ranks.to_numpy(dtype="float64") - 0.5) / n
    transformed.loc[valid] = _standard_normal_ppf(percentile)
    mean = transformed.loc[valid].mean()
    std = transformed.loc[valid].std(ddof=0)
    if not np.isfinite(std) or std <= 0.0:
        transformed.loc[valid] = 0.0
        return transformed
    transformed.loc[valid] = (transformed.loc[valid] - mean) / std
    return transformed


def _standard_normal_ppf(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype="float64")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("probabilities must be strictly inside (0, 1)")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    lower = 0.02425
    upper = 1.0 - lower
    output = np.empty_like(p)

    mask = p < lower
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(p[mask]))
        output[mask] = _polyval(c, q) / _polyval(np.r_[d, 1.0], q)

    mask = (p >= lower) & (p <= upper)
    if np.any(mask):
        q = p[mask] - 0.5
        r = q * q
        output[mask] = (_polyval(a, r) / _polyval(np.r_[b, 1.0], r)) * q

    mask = p > upper
    if np.any(mask):
        q = np.sqrt(-2.0 * np.log(1.0 - p[mask]))
        output[mask] = -_polyval(c, q) / _polyval(np.r_[d, 1.0], q)

    return output


def _polyval(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype="float64")
    for coefficient in coefficients:
        result = result * x + coefficient
    return result
