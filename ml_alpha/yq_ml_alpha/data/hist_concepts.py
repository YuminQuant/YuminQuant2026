"""Read sparse, dated HIST memberships and current market capitalization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path


class HistConcepts:
    def __init__(self, root, basic_root=None):
        self.root = Path(root)
        self.basic_root = Path(basic_root) if basic_root is not None else None
        self.manifest = None

    def identity(self):
        if self.manifest is None:
            path = self.root / "manifest.json"
            if not path.exists():
                raise FileNotFoundError(f"missing HIST concepts: {path}; run derive-hist-concepts")
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if (metadata.get("version") != "hist_mainbz_i_v1"
                    or metadata.get("source_type") != "I"
                    or metadata.get("availability") != "annual_next_may01_interim_sep01"):
                raise ValueError("incompatible HIST concept dataset")
            self.manifest = metadata
        return self.manifest

    def _membership(self, date, symbols):
        self.identity()
        frame = pd.read_parquet(daily_path(self.root, date))
        keys = ["trade_date", "ts_code", "concept_id", "end_date", "available_date"]
        if (frame[keys].isna().any().any() or not frame.trade_date.eq(date).all()
                or frame.duplicated(["ts_code", "concept_id"]).any()
                or not frame.available_date.le(date).all()
                or not frame.end_date.lt(frame.available_date).all()):
            raise ValueError(f"invalid/as-of-violating HIST concepts on {date}")
        positions = {symbol: i for i, symbol in enumerate(symbols)}
        if len(positions) != len(symbols):
            raise ValueError("duplicate HIST stock nodes")
        frame = frame.loc[frame.ts_code.isin(positions)]
        group_index, groups = pd.factorize(frame.concept_id, sort=True)
        stock_index = frame.ts_code.map(positions).to_numpy(dtype=np.int64)
        return stock_index, group_index, len(groups)

    def load_binary(self, date, symbols, device):
        """As-of 0/1 incidence, without a market-cap dependency."""
        import torch

        stocks, concepts, count = self._membership(date, symbols)
        return (torch.as_tensor(stocks, dtype=torch.long, device=device),
                torch.as_tensor(concepts, dtype=torch.long, device=device), count)

    def load(self, date, symbols, device):
        import torch

        stock_index, group_index, count = self._membership(date, symbols)
        if self.basic_root is None:
            raise ValueError("HIST market-cap weighting requires basic_root")
        basic = pd.read_parquet(daily_path(self.basic_root, date), columns=["ts_code", "total_mv"])
        if basic.ts_code.duplicated().any():
            raise ValueError(f"duplicate market capitalization rows on {date}")
        caps = pd.to_numeric(basic.set_index("ts_code").total_mv.reindex(symbols),
                             errors="coerce").to_numpy(dtype=np.float64)
        if not (np.isfinite(caps) & (caps > 0)).all():
            raise ValueError(f"missing/nonpositive total_mv for HIST nodes on {date}")
        # Dividing by a common scale preserves relative market-cap weights.
        caps = (caps / caps.max()).astype(np.float32)
        return (
            torch.as_tensor(stock_index, dtype=torch.long, device=device),
            torch.as_tensor(group_index, dtype=torch.long, device=device),
            count,
            torch.as_tensor(caps, device=device),
        )
