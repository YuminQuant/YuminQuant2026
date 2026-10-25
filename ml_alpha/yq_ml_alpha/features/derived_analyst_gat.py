"""Read the three strict rolling inputs produced by Rust analyst GAT derivation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path, is_all_column_request
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.output import artifact_io as data_io


ANALYST_GAT_COLUMNS = ["return_40d", "turnover_mean_40d", "volatility_40d"]
ANALYST_GAT_IDENTITY = {
    "version": "analyst_gat_inputs_v1", "lookback_days": 40,
    "return": "adjusted_close_t_over_t_minus_40_minus_one",
    "adjustment": "close_times_adj_factor",
    "turnover": "mean_turnover_rate_f_percent_last_40",
    "volatility": "std_simple_adjusted_close_returns_40_ddof0",
    "missing": "strict_41_closes_and_last_40_turnover",
}


class DerivedAnalystGATProvider(FeatureProvider):
    def __init__(self, root, columns):
        self.root = Path(root)
        requested = list(ANALYST_GAT_COLUMNS) if is_all_column_request(columns) else list(columns)
        if len(requested) != len(ANALYST_GAT_COLUMNS) or set(requested) != set(ANALYST_GAT_COLUMNS):
            raise ValueError(f"analyst_raw40_sequence requires exactly {ANALYST_GAT_COLUMNS}")
        self.feature_columns = requested
        self._manifest = None
        self.dataset_identity()
        if not any(data_io.glob(self.root, "*/*.parquet")):
            raise FileNotFoundError(
                f"missing analyst GAT daily inputs under {self.root}; run derive-analyst-gat-inputs"
            )

    def dataset_identity(self):
        if self._manifest is None:
            path = self.root / "manifest.json"
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing analyst GAT input manifest: {path}; run derive-analyst-gat-inputs")
            manifest = json.loads(data_io.read_text(path, encoding="utf-8"))
            if any(manifest.get(key) != value for key, value in ANALYST_GAT_IDENTITY.items()):
                raise ValueError("analyst GAT input dataset version/window/formula mismatch")
            self._manifest = manifest
        return self._manifest

    def require_date(self, trade_date):
        self.dataset_identity()
        path = daily_path(self.root, trade_date)
        if not data_io.exists(path):
            raise FileNotFoundError(f"missing analyst GAT inputs: {path}; run derive-analyst-gat-inputs")
        return path

    def load(self, trade_date):
        path = self.require_date(trade_date)
        columns = ["trade_date", "ts_code", *self.feature_columns]
        frame = data_io.read_projected_frame(path, columns, required=columns)
        if (frame["trade_date"].isna().any() or not frame["trade_date"].eq(trade_date).all()
                or not pd.api.types.is_integer_dtype(frame["trade_date"].dtype)
                or frame["ts_code"].isna().any() or frame["ts_code"].duplicated().any()
                or not frame["ts_code"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
            raise ValueError(f"invalid analyst GAT input date/stock keys: {path}")
        for column in self.feature_columns:
            if (not pd.api.types.is_numeric_dtype(frame[column].dtype)
                    or pd.api.types.is_bool_dtype(frame[column].dtype)):
                raise ValueError(f"analyst GAT input {column} must be numeric: {path}")
        # Incomplete raw windows are null, not valid zero-valued features. Drop
        # them before shared feature normalization can apply its fill-zero rule.
        complete = frame[self.feature_columns].notna().all(axis=1).to_numpy()
        rows = np.flatnonzero(complete)
        values = frame.loc[complete, self.feature_columns].to_numpy(dtype="float64")
        return frame.iloc[rows[np.isfinite(values).all(axis=1)]][columns].reset_index(drop=True)
