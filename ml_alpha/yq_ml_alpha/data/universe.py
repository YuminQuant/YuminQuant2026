from __future__ import annotations

from pathlib import Path

import pandas as pd


BUILTIN_INDEX_UNIVERSES = {"000300.SH", "000905.SH", "000852.SH", "000985.CSI"}


class Universe:
    def __init__(self, universe_id: str, data_root: str | Path = "data") -> None:
        self.universe_id = universe_id
        self.data_root = Path(data_root)
        self._records: pd.DataFrame | None = None

    def filter(self, frame: pd.DataFrame, trade_date: int) -> pd.DataFrame:
        if self.universe_id.lower() in {"mkt_all", "all"} or self.universe_id == "000985.CSI":
            return frame
        members = self._members(trade_date)
        return frame.loc[frame["ts_code"].isin(members)].copy()

    def _members(self, trade_date: int) -> set[str]:
        records = self._load_records()
        records = records.loc[records["trade_date"] <= trade_date]
        if records.empty:
            return set()
        effective_date = records["trade_date"].max()
        return set(records.loc[records["trade_date"] == effective_date, "ts_code"])

    def _load_records(self) -> pd.DataFrame:
        if self._records is not None:
            return self._records
        if self.universe_id in BUILTIN_INDEX_UNIVERSES:
            self._records = _load_index_weight(self.data_root, self.universe_id)
        else:
            self._records = _load_custom_universe(self.data_root, self.universe_id)
        return self._records


def _load_index_weight(data_root: Path, index_code: str) -> pd.DataFrame:
    root = data_root / "index_data" / "monthly_weight" / index_code.replace(".", "_")
    if not root.exists():
        raise FileNotFoundError(f"missing index universe weights: {root}")
    frames = []
    for path in sorted(root.rglob("*.parquet")):
        table = pd.read_parquet(path, columns=["trade_date", "con_code", "weight"])
        table = table.rename(columns={"con_code": "ts_code"})
        table = table.loc[table["weight"].astype(float) > 0, ["trade_date", "ts_code"]]
        frames.append(table)
    if not frames:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    return pd.concat(frames, ignore_index=True).astype({"trade_date": "int32"})


def _load_custom_universe(data_root: Path, universe_id: str) -> pd.DataFrame:
    path = data_root / "universe" / f"{universe_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing custom universe: {path}")
    table = pd.read_parquet(path)
    return table[["trade_date", "ts_code"]].astype({"trade_date": "int32"})
