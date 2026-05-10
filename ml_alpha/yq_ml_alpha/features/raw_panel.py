from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import discover_value_columns, is_all_column_request, read_daily
from yq_ml_alpha.features.base import FeatureProvider


class RawPanelProvider(FeatureProvider):
    """Minimal v1.5 raw-panel provider.

    The provider reads configured raw/OHLCV columns and returns a tabular frame.
    Sequence/tensor reshaping is intentionally left to model code so end-to-end
    models can choose their own windowing and tensor layout.
    """

    def __init__(self, root: str | Path, columns: list[str] | str) -> None:
        self.root = Path(root)
        self.feature_columns = discover_value_columns(self.root) if is_all_column_request(columns) else list(columns)

    def load(self, trade_date: int) -> pd.DataFrame:
        frame = read_daily(self.root, trade_date, self.feature_columns)
        return frame[["trade_date", "ts_code", *self.feature_columns]]
