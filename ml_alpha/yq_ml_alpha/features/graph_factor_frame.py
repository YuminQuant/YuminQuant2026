"""Daily normalized factor snapshots shared by graph and temporal encoders."""
from yq_ml_alpha.output import artifact_io as data_io
from contextlib import nullcontext
import numpy as np
import pandas as pd

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.data.factor_files import factor_dates
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.stock_relations import StockRelations
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.features.transforms import zscore_observed
from yq_ml_alpha.output.parquet_reader import transient_parquet_tables


class FactorHistory(DailyTensorSequence):
    time_major_batch = True

    def __init__(self, config, provider, sequence_length=None, *, daily_source=None):
        params = config.features.params
        super().__init__(config.features.root, provider.feature_columns, (len(provider.feature_columns),),
                         int(params.get("sequence_length", 40) if sequence_length is None else sequence_length))
        self.read_workers = data_io.io_options(self.root)["workers"]
        self.config = config
        self.provider = provider
        self.universe = daily_source.universe if daily_source is not None else Universe(config.universe.id, config.data_root)
        self.filters = daily_source.filters if daily_source is not None else TradeFilters(config.filters, config.data_root)
        self.fill_value = config.preprocess.feature_fill_value
        self.history_transform = str(params.get("history_transform", "mad5_zscore_rank_label"))
        if self.history_transform not in {"mad5_zscore_rank_label", "zscore_clip3"}:
            raise ValueError("unsupported factor history_transform")
        self.available_dates = daily_source.available_dates if daily_source is not None else factor_dates(self.root)
        if not self.available_dates:
            raise ValueError(f"no factor history under {self.root}")
        self.first_date = self.available_dates[0]
        if daily_source is not None:
            self.share_days(daily_source)

    def _load_day(self, date):
        with transient_parquet_tables():
            frame = self.provider.load(date, require_file=True)
        if frame.ts_code.isna().any() or frame.ts_code.duplicated().any() or not frame.trade_date.eq(date).all():
            raise ValueError(f"invalid factor snapshot keys on {date}")
        frame = self.filters.apply(self.universe.filter(frame, date), date)
        frame = self._filter_frame(frame, date)
        # Normalize before label joining. Missing future labels never change graph inputs.
        numeric = frame[self.columns]
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in numeric.dtypes):
            numeric = numeric.apply(pd.to_numeric, errors="coerce")
        raw = numeric.to_numpy(dtype="float64", na_value=np.nan)
        clip = self.history_transform == "zscore_clip3"
        values = zscore_observed(raw, mad5=not clip, fill_value=self.fill_value)
        if clip:
            np.clip(values, -3.0, 3.0, out=values)
            values[~np.isfinite(raw)] = self.fill_value
        return pd.DataFrame(values, columns=self.columns, index=pd.Index(frame.ts_code), copy=False)

    def _filter_frame(self, frame, date):
        """Optional universe refinement before any cross-sectional normalization."""
        return frame

    def symbols(self, date):
        dates = self.window_dates(date)
        if not dates or dates[0] < self.first_date:
            return []
        return super().symbols(date)

    def prefetch_targets(self, dates):
        if not data_io.backend(self.root) or list(dates) != sorted(dates):
            return nullcontext()
        source_dates = {
            day for date in dates if (window := self.window_dates(date)) and window[0] >= self.first_date
            for day in window
        }
        return self.prefetch_sources(sorted(source_dates))

    def forward_view(self, length):
        """Future slices share observed days and keep their own alignment."""
        view = FactorHistory(self.config, self.provider, sequence_length=length, daily_source=self)
        view.set_calendar_dates(self.dates)
        return view


class FactorSequenceProvider(DailySequenceProvider):
    """Complete eligible cross-sections with lazy in-memory factor histories."""

    def __init__(self, config, provider):
        super().__init__(FactorHistory(config, provider))

    def dataset_identity(self):
        transform = "mad5_zscore_v1" if self.sequence.history_transform == "mad5_zscore_rank_label" else "zscore_clip3_v1"
        return {"factor_history": transform, "membership": "complete_history_before_label_join"}


class GraphFactorFrameProvider(FactorSequenceProvider):
    def __init__(self, config, provider):
        super().__init__(config, provider)
        self.relations = StockRelations(_project_path(config.model.params.get("relation_root", "data/derived/stock/relations")))

    def dataset_identity(self):
        return {"factor_history": super().dataset_identity()["factor_history"], "relations": self.relations.identity()}

    def load(self, trade_date):
        self.relations.identity()
        frame = super().load(trade_date)
        if not frame.empty and not data_io.exists(daily_path(self.relations.root, trade_date)):
            raise FileNotFoundError(f"missing stock relations for {trade_date}; run derive-stock-relations")
        return frame
