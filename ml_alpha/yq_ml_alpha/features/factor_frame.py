from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import discover_value_columns, is_all_column_request, read_daily
from yq_ml_alpha.features.base import FeatureProvider


class FactorFrameProvider(FeatureProvider):
    def __init__(self, root: str | Path, columns: list[str] | str) -> None:
        self.root = Path(root)
        self.feature_columns = discover_value_columns(self.root) if is_all_column_request(columns) else list(columns)

    def load(self, trade_date: int) -> pd.DataFrame:
        frame = read_daily(self.root, trade_date, self.feature_columns)
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[["trade_date", "ts_code", *self.feature_columns]]
