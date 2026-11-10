from __future__ import annotations

from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.stores import (
    MINUTE_KEY_COLUMNS, discover_value_columns, is_all_column_request, read_minute,
    require_hdfs_input,
)
from yq_ml_alpha.data.factor_files import FactorFiles
from yq_ml_alpha.features.base import FeatureProvider


class FactorFrameProvider(FeatureProvider):
    def __init__(self, root: str | Path, columns: list[str] | str, end_date: int | None = None, excluded=()) -> None:
        self.root = Path(root)
        self.files = FactorFiles(root)
        self.feature_columns = discover_value_columns(self.root, end_date, excluded) if is_all_column_request(columns) else list(columns)

    def load(self, trade_date: int) -> pd.DataFrame:
        frame = self.files.read(trade_date, self.feature_columns)
        for column in self.feature_columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        return frame[["trade_date", "ts_code", *self.feature_columns]]


class MinuteFactorFrameProvider(FeatureProvider):
    def __init__(self, root: Path, columns: list[str] | str, *, require_hdfs: bool) -> None:
        if (not isinstance(columns, list) or not columns or is_all_column_request(columns)
                or len(set(columns)) != len(columns)
                or set(columns).intersection(MINUTE_KEY_COLUMNS)):
            raise ValueError("minute_factor_frame requires explicit, unique feature columns excluding the minute keys")
        if not isinstance(require_hdfs, bool):
            raise ValueError("features.params.require_hdfs must be boolean")
        if require_hdfs:
            require_hdfs_input(root)
        self.root = root
        self.feature_columns = columns
        self.require_hdfs = require_hdfs

    def load(self, trade_date: int) -> pd.DataFrame:
        return read_minute(self.root, trade_date, self.feature_columns, require_hdfs=self.require_hdfs)
