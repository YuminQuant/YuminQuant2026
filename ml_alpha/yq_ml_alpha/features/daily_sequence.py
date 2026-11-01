"""Full date cross-sections backed by a shared, lazy daily RAM reader."""
from __future__ import annotations

import pandas as pd

from yq_ml_alpha.features.base import FeatureProvider


class DailySequenceProvider(FeatureProvider):
    requires_full_cross_section = True

    def __init__(self, sequence):
        self.sequence = sequence
        self.feature_columns = list(sequence.columns)

    def set_calendar_dates(self, dates):
        self.sequence.set_calendar_dates(dates)

    def load(self, trade_date):
        symbols = self.sequence.symbols(trade_date)
        return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "ts_code": symbols})
