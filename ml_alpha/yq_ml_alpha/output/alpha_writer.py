from __future__ import annotations

from pathlib import Path

import pandas as pd


class AlphaWriter:
    def __init__(self, output_root: str | Path, alpha_id: str) -> None:
        self.output_root = Path(output_root)
        self.alpha_id = alpha_id

    def write(self, predictions: pd.DataFrame) -> list[Path]:
        required = {"trade_date", "ts_code", "score"}
        missing = required.difference(predictions.columns)
        if missing:
            raise ValueError(f"predictions missing columns: {sorted(missing)}")
        written = []
        for trade_date, daily in predictions.groupby("trade_date", sort=True):
            path = self._path(int(trade_date))
            path.parent.mkdir(parents=True, exist_ok=True)
            new_values = daily[["trade_date", "ts_code", "score"]].rename(columns={"score": self.alpha_id})
            new_values[self.alpha_id] = new_values[self.alpha_id].astype("float32")
            if path.exists():
                existing = pd.read_parquet(path)
                merged = existing.merge(new_values, on=["trade_date", "ts_code"], how="outer", suffixes=("", "__new"))
                new_col = f"{self.alpha_id}__new"
                if new_col in merged.columns:
                    merged[self.alpha_id] = merged[new_col].combine_first(merged.get(self.alpha_id))
                    merged = merged.drop(columns=[new_col])
            else:
                merged = new_values
            for column in merged.columns:
                if column not in {"trade_date", "ts_code"}:
                    merged[column] = merged[column].astype("float32")
            tmp = path.with_suffix(".tmp.parquet")
            merged.to_parquet(tmp, index=False)
            tmp.replace(path)
            written.append(path)
        return written

    def _path(self, trade_date: int) -> Path:
        return self.output_root / str(trade_date // 10000) / f"{trade_date}.parquet"
