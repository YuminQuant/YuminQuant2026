"""Read materialized chip-age tensors; never estimate chips during fitting."""
from yq_ml_alpha.output import artifact_io as data_io
import json
from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.features.base import FeatureProvider


class DerivedChipAgeProvider(FeatureProvider):
    def __init__(self, root, params):
        self.root = Path(root)
        self.feature_columns = [f"chip_age_{age}_{bucket:02d}" for age in range(4) for bucket in range(32)]
        self.sequence = DailyTensorSequence(
            root, self.feature_columns, (4, 32),
            length=int(params.get("sequence_length", 30)),
            memory_budget_mb=int(params.get("memory_budget_mb", 128)),
        )
        self.expected_version = params.get("dataset_version", "chip_age_rolling250_v2")
        self._manifest = None

    def set_calendar_dates(self, dates):
        self.sequence.set_calendar_dates(dates)

    def load(self, trade_date):
        self.dataset_identity()
        symbols = self.sequence.symbols(trade_date)
        return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "ts_code": symbols})

    def dataset_identity(self):
        if self._manifest is None:
            path = self.root / "manifest.json"
            if not data_io.exists(path):
                raise FileNotFoundError(f"Missing chip-age manifest: {path}; run derive-chip-age first")
            self._manifest = json.loads(data_io.read_text(path, encoding="utf-8"))
            if self._manifest.get("version") != self.expected_version:
                raise ValueError("chip-age dataset version mismatch")
        return self._manifest
