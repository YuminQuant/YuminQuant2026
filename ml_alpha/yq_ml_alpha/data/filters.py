from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.config import FiltersConfig


class TradeFilters:
    def __init__(self, config: FiltersConfig, data_root: str | Path = "data") -> None:
        self.config = config
        self.data_root = Path(data_root)

    def apply(self, frame: pd.DataFrame, trade_date: int) -> pd.DataFrame:
        output = frame
        if self.config.exclude_bj:
            output = output.loc[~output["ts_code"].str.upper().str.endswith(".BJ")]
        if not self.config.exclude_limit and not self.config.exclude_st:
            return output.copy()

        path = (
            self.data_root
            / "stock_data"
            / "daily"
            / "trade_filter"
            / str(trade_date // 10000)
            / f"{trade_date}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(f"missing trade_filter for {trade_date}: {path}")
        filter_columns = ["ts_code"]
        if self.config.exclude_limit:
            filter_columns.append("is_limit")
        if self.config.exclude_st:
            filter_columns.append("is_st")
        mask = pd.read_parquet(path, columns=filter_columns)
        output = output.merge(mask, on="ts_code", how="left")
        keep = pd.Series(True, index=output.index)
        if self.config.exclude_limit:
            keep &= ~output["is_limit"].fillna(False).astype(bool)
        if self.config.exclude_st and trade_date >= 20160101:
            keep &= ~output["is_st"].fillna(False).astype(bool)
        drop_cols = [column for column in ["is_limit", "is_st"] if column in output.columns]
        return output.loc[keep].drop(columns=drop_cols).copy()
