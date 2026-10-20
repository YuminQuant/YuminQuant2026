"""Daily normalized factor snapshots shared by graph and temporal encoders."""
from pathlib import Path

import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.stock_relations import StockRelations
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.transforms import apply_cross_section_transform


class FactorHistory(DailyTensorSequence):
    def __init__(self, config, provider):
        params = config.features.params
        super().__init__(config.features.root, provider.feature_columns, (len(provider.feature_columns),),
                         int(params.get("sequence_length", 40)), int(params.get("memory_budget_mb", 512)))
        self.provider = provider
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self.fill_value = config.preprocess.feature_fill_value
        self.available_dates = sorted(int(p.stem) for p in Path(self.root).glob("*/*.parquet"))
        if not self.available_dates:
            raise ValueError(f"no factor history under {self.root}")
        self.first_date = self.available_dates[0]

    def day(self, date):
        if date in self.cache:
            self.cache.move_to_end(date)
            return self.cache[date][0]
        if not daily_path(self.root, date).exists():
            raise FileNotFoundError(f"missing factor history for {date}")
        frame = self.provider.load(date)
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid factor snapshot keys on {date}")
        frame = self.filters.apply(self.universe.filter(frame, date), date)
        # Normalize before label joining. Missing future labels never change graph inputs.
        frame[self.columns] = frame[self.columns].apply(pd.to_numeric, errors="coerce").astype("float64")
        frame = apply_cross_section_transform(
            frame, "mad5_zscore_rank_label", self.columns, [], self.fill_value,
        ).set_index("ts_code")[self.columns].astype("float32")
        size = int(frame.memory_usage(index=True, deep=True).sum())
        while self.cache and self.cache_bytes + size > self.budget:
            _, (_, removed) = self.cache.popitem(last=False)
            self.cache_bytes -= removed
        if size <= self.budget:
            self.cache[date] = (frame, size)
            self.cache_bytes += size
        return frame

    def symbols(self, date):
        dates = self.window_dates(date)
        if not dates or dates[0] < self.first_date:
            return []
        return super().symbols(date)


class GraphFactorFrameProvider(FeatureProvider):
    def __init__(self, config, provider):
        self.feature_columns = list(provider.feature_columns)
        self.sequence = FactorHistory(config, provider)
        self.relations = StockRelations(_project_path(config.model.params.get("relation_root", "data/derived/stock/relations")))

    def dataset_identity(self):
        return {"factor_history": "mad5_zscore_v1", "relations": self.relations.identity()}

    def set_calendar_dates(self, dates):
        self.sequence.set_calendar_dates(dates)

    def load(self, trade_date):
        self.relations.identity()
        symbols = self.sequence.symbols(trade_date)
        if symbols and not daily_path(self.relations.root, trade_date).exists():
            raise FileNotFoundError(f"missing stock relations for {trade_date}; run derive-stock-relations")
        # Keep only keys: current features and histories are read lazily with a RAM cap.
        return pd.DataFrame({"trade_date": [trade_date] * len(symbols), "ts_code": symbols})
