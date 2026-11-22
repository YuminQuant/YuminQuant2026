from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import require_hdfs_input
from yq_ml_alpha.output import artifact_io


def future_products(contracts: pd.Series) -> pd.Series:
    return contracts.str.replace(r"\d+(?=\.[^.]+$)", "", regex=True)


class FutureLiquidityPool:
    def __init__(self, data_root: Path, pool_id: str, *, require_hdfs: bool) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pool_id):
            raise ValueError("future pool ID must be a nonempty identifier")
        self.root = data_root / "future_data" / "basic" / "mapping_pool" / pool_id
        self.require_hdfs = require_hdfs
        self.members: dict[int, frozenset[str]] = {}
        self.loaded_years: set[int] = set()

    def prepare(self, dates: list[int]) -> None:
        for year in sorted({date // 10000 for date in dates} - self.loaded_years):
            path = self.root / f"{year}.parquet"
            if self.require_hdfs:
                require_hdfs_input(path)
            try:
                frame = artifact_io.read_parquet(path, columns=["trade_date", "ts_code", "mapping_ts_code"])
            except FileNotFoundError as exc:
                first = min(date for date in dates if date // 10000 == year)
                raise FileNotFoundError(f"missing futures liquidity pool for {first}: {path}") from exc
            trade_date = pd.to_numeric(frame["trade_date"], errors="raise").astype("Int32")
            contracts = frame["mapping_ts_code"].astype("string")
            products = frame["ts_code"].astype("string")
            actual_contract = contracts.str.fullmatch(
                r"[A-Za-z]+[0-9]{1,2}(?:0[1-9]|1[0-2])\.[^.]+", na=False,
            )
            valid = (trade_date // 10000 == year).fillna(False) & actual_contract
            valid &= future_products(contracts).eq(products).fillna(False)
            if not valid.all():
                raise ValueError(f"invalid futures liquidity pool mapping in {path}")
            mappings = frame.assign(trade_date=trade_date, mapping_ts_code=contracts).drop_duplicates()
            if mappings.duplicated(["trade_date", "ts_code"]).any():
                raise ValueError(f"conflicting futures liquidity pool mappings in {path}")
            self.members.update({
                int(date): frozenset(values)
                for date, values in mappings.groupby("trade_date")["mapping_ts_code"]
            })
            self.loaded_years.add(year)
        for date in dates:
            if date not in self.members:
                raise ValueError(f"missing futures liquidity pool for {date}: {self.root}")

    def filter(self, frame: pd.DataFrame, date: int) -> pd.DataFrame:
        return frame.loc[frame["ts_code"].isin(self.members[date])]
