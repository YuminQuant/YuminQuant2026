from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.config import MlAlphaConfig
from yq_ml_alpha.data.stores import read_daily


_SW_CACHE: dict[Path, pd.DataFrame] = {}


def apply_prediction_postprocess(config: MlAlphaConfig, source: pd.DataFrame, score: pd.Series) -> pd.Series:
    neutralize = config.postprocess.neutralize
    if not neutralize.enabled:
        return score.astype("float32")
    if source.empty:
        return score.astype("float32")
    frame = pd.DataFrame(
        {
            "trade_date": source["trade_date"].astype("int32").to_numpy(),
            "ts_code": source["ts_code"].astype(str).to_numpy(),
            "score": score.astype("float32").to_numpy(),
        },
        index=source.index,
    )
    output = []
    for trade_date, daily in frame.groupby("trade_date", sort=False):
        output.append(_neutralize_one_date(config, int(trade_date), daily))
    if not output:
        return score.astype("float32")
    values = pd.concat(output).reindex(source.index)
    return values.astype("float32")


def _neutralize_one_date(config: MlAlphaConfig, trade_date: int, daily: pd.DataFrame) -> pd.Series:
    neutralize = config.postprocess.neutralize
    work = daily[["ts_code", "score"]].copy()
    if neutralize.industry in {"sw_l1", "sw", "shenwan_l1"}:
        industries = _active_sw_l1(neutralize.sw_classification_path, trade_date)
        work = work.merge(industries, on="ts_code", how="left")
    else:
        work["industry"] = "__all__"

    if neutralize.size in {"barra_cne6_size", "barra:size", "size"}:
        size = read_daily(neutralize.barra_root, trade_date, ["SIZE"])
        if not size.empty and "SIZE" in size.columns:
            size = size[["ts_code", "SIZE"]].copy()
            size["ts_code"] = size["ts_code"].astype(str)
            work = work.merge(size, on="ts_code", how="left")
    if "SIZE" not in work.columns:
        work["SIZE"] = np.nan

    residual = _cross_section_residual(work["score"], work["SIZE"], work["industry"])
    return pd.Series(residual, index=daily.index, dtype="float32")


def _active_sw_l1(path: Path, trade_date: int) -> pd.DataFrame:
    members = _load_sw_members(path)
    if members.empty:
        return pd.DataFrame(columns=["ts_code", "industry"])
    in_date = members["in_date_num"].fillna(0)
    out_date = members["out_date_num"].fillna(99991231)
    active = members.loc[(in_date <= trade_date) & (out_date >= trade_date), ["ts_code", "industry"]].copy()
    return active.drop_duplicates("ts_code", keep="last")


def _load_sw_members(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path in _SW_CACHE:
        return _SW_CACHE[path]
    if not path.exists():
        raise FileNotFoundError(f"missing SW classification file for neutralization: {path}")
    frame = pd.read_parquet(path)
    industry_column = _first_existing(frame, ["l1_code", "industry_code", "index_code", "level1_code"])
    required = {"ts_code", "in_date", "out_date", industry_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"SW classification file {path} missing columns: {sorted(missing)}")
    output = frame[["ts_code", "in_date", "out_date", industry_column]].copy()
    output = output.rename(columns={industry_column: "industry"})
    output["ts_code"] = output["ts_code"].astype(str)
    output["industry"] = output["industry"].astype(str).replace({"nan": "__missing__", "None": "__missing__"})
    output["in_date_num"] = pd.to_numeric(output["in_date"], errors="coerce")
    output["out_date_num"] = pd.to_numeric(output["out_date"], errors="coerce")
    _SW_CACHE[path] = output
    return output


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"missing industry column; tried {candidates}")


def _cross_section_residual(score: pd.Series, size: pd.Series, industry: pd.Series) -> np.ndarray:
    y = pd.to_numeric(score, errors="coerce").astype("float64").to_numpy()
    valid = np.isfinite(y)
    residual = np.full(len(y), np.nan, dtype="float64")
    if valid.sum() < 2:
        residual[valid] = 0.0
        return residual

    columns = [np.ones(int(valid.sum()), dtype="float64")]
    size_values = pd.to_numeric(size, errors="coerce").astype("float64").to_numpy()[valid]
    if np.isfinite(size_values).any():
        fill = float(np.nanmean(size_values))
        size_values = np.where(np.isfinite(size_values), size_values, fill)
        std = float(np.std(size_values))
        if std > 1e-12:
            columns.append((size_values - float(np.mean(size_values))) / std)

    industry_values = industry.fillna("__missing__").astype(str).to_numpy()[valid]
    dummies = pd.get_dummies(industry_values, dtype="float64")
    if dummies.shape[1] > 1:
        for column in dummies.columns[1:]:
            columns.append(dummies[column].to_numpy(dtype="float64"))

    x = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(x, y[valid], rcond=None)
    residual[valid] = y[valid] - x @ beta
    return residual
