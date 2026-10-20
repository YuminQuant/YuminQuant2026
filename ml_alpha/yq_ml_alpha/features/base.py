from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class FeatureProvider(ABC):
    feature_columns: list[str]
    # Graph models need unlabeled nodes too; labels mask the loss, not the graph.
    requires_full_cross_section: bool = False

    @abstractmethod
    def load(self, trade_date: int) -> pd.DataFrame:
        """Return a frame with trade_date, ts_code and feature columns."""
