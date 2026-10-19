from __future__ import annotations

from pathlib import Path

import numpy as np

from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.features.bar_panel import BAR_FEATURES, BarPanelProvider, _feature_column


class RawBarProvider(BarPanelProvider):
    """Adjusted daily prices normalized by current close; volume by current volume."""

    def __init__(self, root, columns, params=None):
        params = dict(params or {})
        params.update(source_frequency="daily", bar_size=1, time_series_scale="none", strict=True)
        super().__init__(root, columns, params)
        if self.output_features != BAR_FEATURES:
            raise ValueError("rawbar requires open, high, low, close, vwap, volume in that order")
        self.adj_root = Path(params.get("adj_root", self.root.parent / "adj_factor"))

    def _read_daily_session(self, trade_date):
        cached = self._cache_get(trade_date)
        if cached is not None:
            return cached
        session = super()._read_daily_session(trade_date).copy()
        if not session.empty:
            adjustment = read_daily(self.adj_root, trade_date, ["adj_factor"])
            if adjustment.empty:
                raise FileNotFoundError(f"rawbar missing adjustment factors: {trade_date} under {self.adj_root}")
            session = session.merge(adjustment[["ts_code", "adj_factor"]], on="ts_code", how="left", validate="one_to_one")
            factor = session["adj_factor"].astype("float64")
            valid = np.isfinite(factor) & (factor > 0)
            session = session.loc[valid].copy()
            for column in ["open", "high", "low", "close", "amount"]:
                # Scaling amount also scales the VWAP subsequently computed by the base provider.
                session[column] = session[column].astype("float64") * factor.loc[valid]
            session = session.drop(columns="adj_factor")
        self._cache_put(trade_date, session)
        return session

    def _finalize_window(self, trade_date, wide):
        frame = super()._finalize_window(trade_date, wide)
        if frame.empty:
            return frame
        close = frame[_feature_column("close", self.total_steps - 1)].astype("float64")
        volume = frame[_feature_column("volume", self.total_steps - 1)].astype("float64")
        valid = np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
        frame = frame.loc[valid].copy()
        for feature in self.output_features:
            columns = [_feature_column(feature, step) for step in range(self.total_steps)]
            denominator = volume.loc[valid] if feature == "volume" else close.loc[valid]
            frame[columns] = frame[columns].astype("float64").div(denominator, axis=0)
        finite = np.isfinite(frame[self.feature_columns].to_numpy(dtype="float64")).all(axis=1)
        return frame.loc[finite].reset_index(drop=True)
