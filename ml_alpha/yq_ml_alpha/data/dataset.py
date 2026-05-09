from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from yq_ml_alpha.config import MlAlphaConfig
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.factor_frame import FactorFrameProvider
from yq_ml_alpha.features.raw_panel import RawPanelProvider


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str


class DatasetBuilder:
    def __init__(self, config: MlAlphaConfig) -> None:
        self.config = config
        self.feature_provider = make_feature_provider(config)
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)

    def load(self, dates: list[int], include_label: bool) -> DatasetBundle:
        frames = []
        for trade_date in dates:
            features = self.feature_provider.load(trade_date)
            if features.empty:
                continue
            frame = self.universe.filter(features, trade_date)
            frame = self.filters.apply(frame, trade_date)
            if include_label:
                labels = read_daily(self.config.label.root, trade_date, [self.config.label.id])
                frame = frame.merge(labels[["trade_date", "ts_code", self.config.label.id]], on=["trade_date", "ts_code"], how="left")
                frame = frame.loc[frame[self.config.label.id].notna()]
            frames.append(frame)
        if frames:
            output = pd.concat(frames, ignore_index=True)
        else:
            columns = ["trade_date", "ts_code", *self.feature_provider.feature_columns]
            if include_label:
                columns.append(self.config.label.id)
            output = pd.DataFrame(columns=columns)
        self._maybe_cache(output, dates, include_label)
        return DatasetBundle(output, self.feature_provider.feature_columns, self.config.label.id)

    def _maybe_cache(self, frame: pd.DataFrame, dates: list[int], include_label: bool) -> None:
        if not self.config.materialize.cache_samples or not dates:
            return
        cache_dir = Path(self.config.materialize.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        split = "labeled" if include_label else "predict"
        frame.to_parquet(cache_dir / f"{split}_{dates[0]}_{dates[-1]}.parquet", index=False)


def make_feature_provider(config: MlAlphaConfig) -> FeatureProvider:
    if config.features.type == "factor_frame":
        return FactorFrameProvider(config.features.root, config.features.columns)
    if config.features.type == "raw_panel":
        return RawPanelProvider(config.features.root, config.features.columns)
    raise ValueError(f"unsupported features.type: {config.features.type}")
