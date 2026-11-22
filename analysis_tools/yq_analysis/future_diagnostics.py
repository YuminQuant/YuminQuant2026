"""Distribution diagnostics from engine-produced daily statistics (no raw rereads)."""
from pathlib import Path
import json
import pandas as pd
from yq_analysis.report_source import read_parquet


def distribution_diagnostics(path: Path) -> dict:
    if not path.is_file():
        return {"available": False, "scopes": [], "monthly": []}
    frame = read_parquet(path)
    required = {"trade_date", "ts_code", "observations", "mean", "std", "min", "p25", "median", "p75", "max", "coverage", "inf_rate"}
    if missing := required - set(frame):
        raise ValueError(f"Distribution statistics missing columns: {sorted(missing)}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("Duplicate daily distribution keys")
    frame["constant"] = (frame["observations"].gt(0) & frame["std"].eq(0)).astype(float)
    frame["flat_iqr"] = (frame["observations"].gt(0) & frame["p25"].eq(frame["p75"])).astype(float)
    frame.loc[frame.observations.eq(0), ["constant", "flat_iqr"]] = None
    means = ["mean", "std", "p25", "median", "p75", "coverage", "inf_rate", "constant", "flat_iqr"]
    grouped = frame.groupby("ts_code", sort=True)
    scopes = grouped[means].mean()
    scopes["min"] = grouped["min"].min()
    scopes["max"] = grouped["max"].max()
    scopes["observations"] = grouped["observations"].sum()
    scopes["days"] = grouped.size()
    market = frame[frame.ts_code.eq("all")].copy()
    market["month"] = market.trade_date.astype(str).str[:6]
    monthly = market.groupby("month")[means].mean().reset_index()
    def rows(value):
        return json.loads(value.to_json(orient="records", double_precision=8))
    return {"available": True, "scopes": rows(scopes.reset_index()), "monthly": rows(monthly)}
