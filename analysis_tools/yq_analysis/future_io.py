from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def load_future_backtest_result(root: str | Path, factor_id: str) -> dict:
    """Load one factor's daily summary and its backtest settings."""
    source = pq.ParquetFile(Path(root) / factor_id / "summary.parquet")
    metadata = source.metadata.metadata or {}
    return {
        "summary": source.read().to_pandas(),
        "label_id": metadata.get(b"label_id", b"").decode(),
        "quantile_mode": metadata.get(b"quantile_mode", b"daily").decode(),
        "tail_quantile": int(metadata.get(b"tail_quantile", b"90")),
    }


def load_future_sectors(path: str | Path) -> pd.DataFrame:
    """Read the sector and contract tables in FUTURE_SECTORS.md."""
    rows = [
        [part.strip() for part in line.strip("|").split("|")]
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.startswith("| ")
    ]
    sectors = pd.DataFrame(
        [row for row in rows if len(row) == 5 and row[3] in {"商品", "金融", "运价"}],
        columns=["abbr", "sector", "name", "kind", "product"],
    )
    contracts = pd.DataFrame(
        [row for row in rows if len(row) == 4 and row[0].isascii() and row[0].isupper()],
        columns=["product", "english", "chinese", "exchange"],
    )
    members = sectors.assign(product=sectors["product"].str.split(", ")).explode("product")
    mapping = members.merge(contracts, on="product", how="outer", validate="one_to_one", indicator=True)
    if mapping.empty or mapping["_merge"].ne("both").any():
        raise ValueError(f"Sector and product tables do not match in {path}")
    order = sectors["abbr"].drop_duplicates()
    mapping["abbr"] = pd.Categorical(mapping["abbr"], categories=order, ordered=True)
    mapping["ts_code"] = mapping["product"] + "." + mapping["exchange"]
    return mapping.drop(columns="_merge").sort_values("abbr", kind="stable").set_index("ts_code")
