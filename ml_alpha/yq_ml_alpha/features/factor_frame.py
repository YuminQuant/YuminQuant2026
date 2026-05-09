from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.features.base import FeatureProvider


class FactorFrameProvider(FeatureProvider):
    def __init__(self, root: str | Path, columns: list[str]) -> None:
        self.root = Path(root)
        self.feature_columns = list(columns)

    def load(self, trade_date: int) -> pd.DataFrame:
        frame = read_daily(self.root, trade_date, self.feature_columns)
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[["trade_date", "ts_code", *self.feature_columns]]
