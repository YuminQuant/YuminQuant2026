from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.config import PROJECT_ROOT
from yq_ml_alpha.data.future_pool import future_products
from yq_ml_alpha.data.stores import require_hdfs_input
from yq_ml_alpha.output import artifact_io


def _load_sectors(config) -> dict[str, str]:
    relative = config.preprocess.future_sector_map
    if relative is None:
        path = PROJECT_ROOT / "factor_engine" / "docs" / "FUTURE_SECTORS.md"
        document = path.read_text(encoding="utf-8")
    else:
        path = config.data_root / relative
        document = artifact_io.read_text(path, encoding="utf-8")

    sectors = {}
    exchanges = {}
    for line in document.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) == 5 and cells[3] in {"商品", "金融", "运价"}:
            for code in cells[4].split(","):
                code = code.strip()
                if code in sectors:
                    raise ValueError(f"duplicate futures sector for {code} in {path}")
                sectors[code] = cells[0]
        if len(cells) == 4 and cells[0].isascii() and cells[0].isupper():
            code = cells[0]
            if code in exchanges:
                raise ValueError(f"duplicate futures exchange for {code} in {path}")
            exchanges[code] = cells[3]
    if not sectors or sectors.keys() != exchanges.keys():
        raise ValueError(f"futures sector and exchange tables do not match in {path}")
    return {f"{code}.{exchanges[code]}": sector for code, sector in sectors.items()}


def fit_future_sector_ic(config, feature_columns: list[str], train_dates: list[int]) -> dict:
    if not train_dates:
        raise ValueError("sector IC alignment requires training dates")
    root = config.data_root / config.preprocess.future_sector_ic_summary_root
    product_sectors = _load_sectors(config)
    sectors = sorted(set(product_sectors.values()))
    dates = set(train_dates)
    pool_id = config.filters.future_pool_id if config.filters.future_liquidity_enabled else "none"
    require_hdfs = config.features.params.get("require_hdfs", False)

    def read_direction(column):
        path = root / column / "summary.parquet"
        if require_hdfs:
            require_hdfs_input(path)
        with artifact_io.open_parquet(path) as parquet:
            metadata = parquet.metadata.metadata or {}
            if (metadata.get(b"label_id") != config.label.id.encode()
                    or metadata.get(b"pool_id") != pool_id.encode()):
                raise ValueError(f"sector IC summary has a different label or liquidity pool: {path}")
            summary = parquet.read(columns=["trade_date", "ts_code", "ic"]).to_pandas()
        summary = summary.loc[summary["trade_date"].isin(dates)]
        missing = sorted(dates - set(summary.loc[summary["ts_code"].eq("all"), "trade_date"]))
        if missing:
            raise ValueError(f"sector IC summary {path} misses {len(missing)} training dates, "
                             f"from {missing[0]} through {missing[-1]}")
        summary = summary.loc[summary["ts_code"].str.startswith("sector:", na=False)]
        if summary.empty or summary.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"missing or duplicated daily sector IC rows in {path}")
        summary = summary.loc[np.isfinite(summary["ic"])]
        stats = summary.groupby(summary["ts_code"].str.removeprefix("sector:"))["ic"].agg(["mean", "count"])
        if set(stats.index) - set(sectors):
            raise ValueError(f"unknown sector in IC summary {path}")
        return column, {
            sector: {
                "sign": -1 if sector in stats.index and stats.at[sector, "mean"] < 0 else 1,
                "days": int(stats.at[sector, "count"]) if sector in stats.index else 0,
            }
            for sector in sectors
        }

    return {
        "product_sectors": product_sectors,
        "directions": dict(artifact_io.ordered_io_map(read_direction, feature_columns, root=root)),
    }


def apply_future_sector_ic(frame: pd.DataFrame, state: dict) -> pd.DataFrame:
    sectors = future_products(frame["ts_code"]).map(state["product_sectors"])
    for column, directions in state["directions"].items():
        negative = [sector for sector, stats in directions.items() if stats["sign"] == -1]
        flip = sectors.isin(negative)
        if flip.any():
            frame.loc[flip, column] = -frame.loc[flip, column]
    return frame
