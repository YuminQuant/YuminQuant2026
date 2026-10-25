"""Single-date factor inputs normalized within the visible analyst-covered universe."""
from __future__ import annotations

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.analyst_coverage import AnalystCoverage
from yq_ml_alpha.features.graph_factor_frame import FactorHistory, FactorSequenceProvider


class AnalystFactorHistory(FactorHistory):
    def __init__(self, config, provider, coverage):
        if config.features.params.get("sequence_length", 1) != 1:
            raise ValueError("analyst_factor_sequence requires sequence_length=1")
        super().__init__(config, provider, sequence_length=1)
        self.coverage = coverage

    def set_calendar_dates(self, dates):
        super().set_calendar_dates(dates)
        self.coverage.set_calendar_dates(dates)

    def _filter_frame(self, frame, date):
        return frame.loc[frame.ts_code.isin(self.coverage.covered_symbols(date))].copy()


class AnalystFactorFrameProvider(FactorSequenceProvider):
    requires_full_cross_section = True

    def __init__(self, config, provider):
        params = config.features.params
        self.feature_columns = list(provider.feature_columns)
        self.coverage = AnalystCoverage(
            _project_path(params.get("coverage_root", "data/derived/stock/analyst_coverage")),
            params.get("coverage_window_months", 6),
        )
        self.sequence = AnalystFactorHistory(config, provider, self.coverage)

    def dataset_identity(self):
        identity = super().dataset_identity()
        identity.update(
            coverage=self.coverage.identity(),
            membership="visible_analyst_coverage_before_normalization_and_label_join",
        )
        return identity

    def load(self, trade_date):
        self.coverage.identity()
        return super().load(trade_date)
