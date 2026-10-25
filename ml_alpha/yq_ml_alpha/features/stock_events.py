"""Strict 28-event inputs and optional quarterly Top-K neighbor event sums."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yq_ml_alpha.config import _project_path
from yq_ml_alpha.data.analyst_coverage import AnalystCoverage
from yq_ml_alpha.data.stores import daily_path, is_all_column_request
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.output import artifact_io as data_io


EVENT_TYPES = (
    "earnings_profit", "equity_incentive", "major_contract", "fiscal_subsidy",
    "shareholder_increase", "litigation_win", "negative_rumor_clarification",
    "dishonest_execution_removed", "issuer_rating_upgrade", "earnings_loss_warning",
    "enforcement_subject", "shareholder_reduction_plan", "equity_freeze",
    "regulatory_rectification", "litigation_defendant", "dishonest_execution",
    "issuer_rating_downgrade", "exchange_inquiry", "abnormal_auditor_change",
    "seasoned_offering_termination", "survey_securities_asset_management",
    "survey_securities_proprietary", "survey_fund", "survey_insurance", "survey_investment",
    "survey_foreign", "survey_securities", "survey_other",
)
EVENT_COLUMNS = tuple("event_" + name for name in EVENT_TYPES)
RELATED_EVENT_COLUMNS = tuple("related_" + name for name in EVENT_COLUMNS)
EVENT_DATASET_IDENTITY = dict(
    version="stock_events_v1", event_types=list(EVENT_TYPES), columns=list(EVENT_COLUMNS),
    window_months=3, half_life_trading_days=20,
    availability="explicit_first_available_trading_day",
    source_version="stock_events_source_v1", source_universe="all_a_shares",
    row_universe="daily_pv_union_active_event_stocks",
    missing_neighbor="zero_only_under_complete_event_coverage", source_complete=True,
)


class StockEventsProvider(FeatureProvider):
    """Read complete event-count vectors; source absence never means zero events."""

    def __init__(self, config):
        if config.preprocess.cross_section_transform.strip().lower() != "none":
            raise ValueError("stock_events requires preprocess.cross_section_transform=none")
        if config.materialize.cache_samples:
            raise ValueError("stock_events requires materialize.cache_samples=false")
        params = config.features.params
        self.root = _project_path(params.get("event_root", config.features.root))
        self.include_related = params.get("include_related", False)
        if not isinstance(self.include_related, bool):
            raise ValueError("stock_events include_related must be boolean")
        self.active_scope = params.get("active_scope", "own")
        if self.active_scope != "own":
            raise ValueError("stock_events active_scope must be own")
        value = params.get("top_neighbors", 3)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError("stock_events top_neighbors must be a positive integer")
        self.top_neighbors = int(value)
        self.feature_columns = list(EVENT_COLUMNS)
        if self.include_related:
            self.feature_columns.extend(RELATED_EVENT_COLUMNS)
        if (not is_all_column_request(config.features.columns)
                and list(config.features.columns) != self.feature_columns):
            raise ValueError("stock_events columns must follow the complete ordered own/related event taxonomy")
        self.coverage = AnalystCoverage(
            _project_path(params.get("coverage_root", "data/derived/stock/analyst_coverage")),
            params.get("coverage_window_months", 6),
        ) if self.include_related else None
        self._manifest = None
        self._cached_date = None
        self._cached_frame = None
        self._cached_values = None

    def set_calendar_dates(self, dates):
        if self.coverage is not None:
            self.coverage.set_calendar_dates(dates)

    def _load_manifest(self):
        if self._manifest is None:
            path = self.root / "manifest.json"
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing stock event manifest: {path}; run derive-stock-events")
            manifest = json.loads(data_io.read_text(path, encoding="utf-8"))
            if (manifest.get("source_complete") is not True
                    or any(manifest.get(key) != value for key, value in EVENT_DATASET_IDENTITY.items())):
                raise ValueError("stock event dataset taxonomy/coverage/decay policy mismatch; run derive-stock-events")
            revision = manifest.get("source_revision")
            if not isinstance(revision, str) or not revision.strip():
                raise ValueError("stock event manifest requires a nonempty source_revision")
            bounds = [manifest.get("source_coverage_start"), manifest.get("source_coverage_end")]
            if any(isinstance(value, bool) or not isinstance(value, int) for value in bounds):
                raise ValueError("stock event manifest requires integer source coverage bounds")
            for value in bounds:
                try:
                    parsed = pd.Timestamp(str(value))
                except (ValueError, TypeError) as exc:
                    raise ValueError("invalid stock event source coverage date") from exc
                if parsed.strftime("%Y%m%d") != str(value):
                    raise ValueError("stock event source coverage dates must be YYYYMMDD")
            if bounds[0] > bounds[1]:
                raise ValueError("stock event source coverage bounds are reversed")
            self._manifest = manifest
        return self._manifest

    def dataset_identity(self):
        manifest = self._load_manifest()
        # Coverage may extend without changing historical data. Corrections or
        # backfills must change source_revision, which is part of this identity.
        semantics = {key: manifest[key] for key in [*EVENT_DATASET_IDENTITY, "source_revision"]}
        identity = dict(stock_events=semantics, active_scope=self.active_scope,
                        include_related=self.include_related, candidate_rows="is_daily_pv_true")
        if self.coverage is not None:
            identity.update(
                coverage=self.coverage.identity(), top_neighbors=self.top_neighbors,
                neighbor_policy="full_snapshot_shared_analyst_count_desc_code_asc_excluding_self",
                aggregation="sum_neighbor_event_counts_no_weighting_no_rerouting",
            )
        return identity

    def _read_date(self, trade_date):
        manifest = self._load_manifest()
        lower = int((pd.Timestamp(str(trade_date)) - pd.DateOffset(months=3)).strftime("%Y%m%d"))
        if lower < manifest["source_coverage_start"] or trade_date > manifest["source_coverage_end"]:
            raise ValueError(
                f"incomplete stock event source coverage for {trade_date}: need ({lower}, {trade_date}]; "
                "extend canonical stock_events coverage and run derive-stock-events"
            )
        if self._cached_date != trade_date:
            path = daily_path(self.root, trade_date)
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing daily stock events: {path}; run derive-stock-events")
            columns = ["trade_date", "ts_code", "is_daily_pv", *EVENT_COLUMNS]
            frame = data_io.read_projected_frame(path, columns, required=columns)
            if (frame["trade_date"].isna().any() or not frame["trade_date"].eq(trade_date).all()
                    or not pd.api.types.is_integer_dtype(frame["trade_date"].dtype)
                    or frame["ts_code"].isna().any() or frame["ts_code"].duplicated().any()
                    or not frame["ts_code"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                    or not pd.api.types.is_bool_dtype(frame["is_daily_pv"].dtype)
                    or frame["is_daily_pv"].isna().any()):
                raise ValueError(f"invalid stock event date/stock/candidate keys: {path}")
            for column in EVENT_COLUMNS:
                if (not pd.api.types.is_numeric_dtype(frame[column].dtype)
                        or pd.api.types.is_bool_dtype(frame[column].dtype)):
                    raise ValueError(f"stock event column must be numeric: {column} in {path}")
            if frame[list(EVENT_COLUMNS)].isna().any().any():
                raise ValueError(f"missing stock event counts cannot be treated as zero: {path}")
            values = frame[list(EVENT_COLUMNS)].to_numpy(dtype="float64")
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"stock event counts must be finite and nonnegative: {path}")
            self._cached_date, self._cached_frame, self._cached_values = trade_date, frame, values
        return self._cached_frame, self._cached_values

    def load(self, trade_date):
        frame, values = self._read_date(trade_date)
        positions = np.flatnonzero(frame["is_daily_pv"].to_numpy(dtype=bool))
        positions = positions[(values[positions] > 0).any(axis=1)]
        output = frame.iloc[positions][["trade_date", "ts_code", *EVENT_COLUMNS]].copy().reset_index(drop=True)
        if self.coverage is not None:
            symbols = output["ts_code"].tolist()
            neighbors = self.coverage.top_neighbors(trade_date, symbols, self.top_neighbors)
            lookup = pd.Index(frame["ts_code"])
            related = np.zeros((len(output), len(EVENT_COLUMNS)), dtype="float64")
            for row, stock in enumerate(symbols):
                indices = lookup.get_indexer(neighbors[stock])
                # The validated row-union protocol guarantees every nonzero
                # event stock has a row. Missing fixed Top-K neighbors are zero;
                # they are never replaced with lower-ranked candidates.
                indices = indices[indices >= 0]
                related[row] = values[indices].sum(axis=0)
            if not np.isfinite(related).all():
                raise ValueError(f"nonfinite related stock event sums on {trade_date}")
            output[list(RELATED_EVENT_COLUMNS)] = related
        return output[["trade_date", "ts_code", *self.feature_columns]]
