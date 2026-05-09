from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureProvider(ABC):
    feature_columns: list[str]

    @abstractmethod
    def load(self, trade_date: int) -> pd.DataFrame:
        """Return a frame with trade_date, ts_code and feature columns."""
