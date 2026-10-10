from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.features.base import FeatureProvider


class DerivedLogsigProvider(FeatureProvider):
    """Read the fixed Rust derive-logsig dataset; never compute during training."""

    def __init__(self, root):
        self.root = Path(root)
        self.feature_columns = [f"logsig_{i:04}" for i in range(1, 227)]

    def load(self, trade_date):
        path = daily_path(self.root, trade_date)
        if not path.exists():
            raise FileNotFoundError(f"Missing derived logsignature: {path}; run Rust derive-logsig first")
        return pd.read_parquet(path, columns=["trade_date", "ts_code", *self.feature_columns])
