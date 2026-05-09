from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.features.base import FeatureProvider


class RawPanelProvider(FeatureProvider):
    """Minimal v1.5 raw-panel provider.

    The provider reads configured raw/OHLCV columns and returns a tabular frame.
    Sequence/tensor reshaping is intentionally left to model code so end-to-end
    models can choose their own windowing and tensor layout.
    """

    def __init__(self, root: str | Path, columns: list[str]) -> None:
        self.root = Path(root)
        self.feature_columns = list(columns)

    def load(self, trade_date: int) -> pd.DataFrame:
        frame = read_daily(self.root, trade_date, self.feature_columns)
        return frame[["trade_date", "ts_code", *self.feature_columns]]
