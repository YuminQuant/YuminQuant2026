from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import read_daily
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.bar_panel import BarPanelProvider, MultiBarPanelProvider
from yq_ml_alpha.features.factor_frame import FactorFrameProvider
from yq_ml_alpha.features.raw_panel import RawPanelProvider
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.features.transforms import apply_cross_section_transform


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str


class DatasetBuilder:
    def __init__(self, config: MlAlphaConfig) -> None:
        self.config = config
        self.feature_provider = make_feature_provider(config)
        if hasattr(self.feature_provider, "set_calendar_dates"):
            self.feature_provider.set_calendar_dates(TradingCalendar.load(config.data_root).dates)
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self._source_st_symbol_cache: dict[int, set[str]] = {}

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
            frame = self._preprocess(frame, include_label)
            if include_label:
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

    def load_sequence(
        self,
        dates: list[int],
        include_label: bool,
        calendar: TradingCalendar,
        sequence_length: int,
        sequence_frequency: str,
    ) -> DatasetBundle:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        base_columns = list(self.feature_provider.feature_columns)
        sequence_columns = _sequence_feature_columns(base_columns, sequence_length)
        feature_cache: dict[int, pd.DataFrame] = {}
        frames = []

        for trade_date in dates:
            sequence_dates = _sequence_dates(calendar, trade_date, sequence_frequency, sequence_length)
            if len(sequence_dates) < sequence_length:
                continue

            target = self.feature_provider.load(trade_date)
            if target.empty:
                continue
            target = self.universe.filter(target[["trade_date", "ts_code"]].copy(), trade_date)
            target = self.filters.apply(target, trade_date)
            frame = target[["trade_date", "ts_code"]].copy()

            for step, sequence_date in enumerate(sequence_dates):
                features = self._sequence_feature_frame(sequence_date, feature_cache)
                renamed = features.rename(
                    columns={column: _sequence_feature_column(column, step) for column in base_columns}
                )
                frame = frame.merge(
                    renamed[["ts_code", *[_sequence_feature_column(column, step) for column in base_columns]]],
                    on="ts_code",
                    how="left",
                )

            frame[sequence_columns] = frame[sequence_columns].replace([np.inf, -np.inf], np.nan).fillna(
                self.config.preprocess.feature_fill_value
            )
            if include_label:
                labels = read_daily(self.config.label.root, trade_date, [self.config.label.id])
                labels = self._preprocess_labels(labels)
                frame = frame.merge(
                    labels[["trade_date", "ts_code", self.config.label.id]],
                    on=["trade_date", "ts_code"],
                    how="left",
                )
                frame = frame.loc[frame[self.config.label.id].notna()]
            frames.append(frame)

        if frames:
            output = pd.concat(frames, ignore_index=True)
        else:
            columns = ["trade_date", "ts_code", *sequence_columns]
            if include_label:
                columns.append(self.config.label.id)
            output = pd.DataFrame(columns=columns)
        self._maybe_cache(output, dates, include_label)
        return DatasetBundle(output, sequence_columns, self.config.label.id)

    def load_bar_panel(
        self,
        dates: list[int],
        include_label: bool,
        calendar: TradingCalendar,
    ) -> DatasetBundle:
        if not isinstance(self.feature_provider, (BarPanelProvider, MultiBarPanelProvider)):
            raise TypeError("load_bar_panel requires BarPanelProvider or MultiBarPanelProvider")
        frames = []
        total_dates = len(dates)
        split_name = "labeled" if include_label else "predict"
        for date_idx, trade_date in enumerate(dates, start=1):
            history_dates = calendar.between(calendar.dates[0], trade_date)
            source_dates = self.feature_provider.required_history_dates(history_dates)
            progress = _bar_panel_progress(split_name, date_idx, total_dates, trade_date)
            features = self.feature_provider.load_window(
                trade_date,
                history_dates,
                exclude_bj=self.config.filters.exclude_bj,
                st_symbols_by_date=self._source_st_symbols_by_date(source_dates),
                progress=progress,
            )
            progress(f"target_done feature_rows={len(features)}")
            if features.empty:
                continue
            frame = self.universe.filter(features, trade_date)
            frame = self.filters.apply(frame, trade_date)
            if include_label:
                labels = read_daily(self.config.label.root, trade_date, [self.config.label.id])
                frame = frame.merge(
                    labels[["trade_date", "ts_code", self.config.label.id]],
                    on=["trade_date", "ts_code"],
                    how="left",
                )
            frame = self._preprocess(frame, include_label)
            if include_label:
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

    def _source_st_symbols_by_date(self, dates: list[int]) -> dict[int, set[str]]:
        if not self.config.filters.exclude_st:
            return {}
        output: dict[int, set[str]] = {}
        for trade_date in dates:
            if trade_date < 20160101:
                continue
            path = (
                Path(self.config.data_root)
                / "stock_data"
                / "daily"
                / "trade_filter"
                / str(trade_date // 10000)
                / f"{trade_date}.parquet"
            )
            if not path.exists():
                raise FileNotFoundError(f"missing trade_filter for {trade_date}: {path}")
            if trade_date not in self._source_st_symbol_cache:
                mask = pd.read_parquet(path, columns=["ts_code", "is_st"])
                st_mask = mask["is_st"].fillna(False).astype(bool)
                self._source_st_symbol_cache[trade_date] = set(mask.loc[st_mask, "ts_code"].astype(str))
            symbols = self._source_st_symbol_cache[trade_date]
            if symbols:
                output[trade_date] = symbols
        return output

    def _preprocess(self, frame: pd.DataFrame, include_label: bool) -> pd.DataFrame:
        columns = list(self.feature_provider.feature_columns)
        label_columns = [self.config.label.id] if include_label and self.config.label.id in frame.columns else []
        return apply_cross_section_transform(
            frame,
            self.config.preprocess.cross_section_transform,
            columns,
            label_columns=label_columns,
            feature_fill_value=self.config.preprocess.feature_fill_value,
        )

    def _preprocess_labels(self, frame: pd.DataFrame) -> pd.DataFrame:
        return apply_cross_section_transform(
            frame,
            self.config.preprocess.cross_section_transform,
            [],
            label_columns=[self.config.label.id] if self.config.label.id in frame.columns else [],
            feature_fill_value=self.config.preprocess.feature_fill_value,
        )

    def _sequence_feature_frame(self, trade_date: int, cache: dict[int, pd.DataFrame]) -> pd.DataFrame:
        if trade_date in cache:
            return cache[trade_date]
        features = self.feature_provider.load(trade_date)
        if features.empty:
            columns = ["trade_date", "ts_code", *self.feature_provider.feature_columns]
            frame = pd.DataFrame(columns=columns)
        else:
            frame = apply_cross_section_transform(
                features,
                self.config.preprocess.cross_section_transform,
                list(self.feature_provider.feature_columns),
                label_columns=[],
                feature_fill_value=self.config.preprocess.feature_fill_value,
            )
            frame = frame[["trade_date", "ts_code", *self.feature_provider.feature_columns]]
        cache[trade_date] = frame
        return frame

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
    if config.features.type == "logsig_signature":
        return LogsigSignatureProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "bar_panel":
        return BarPanelProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "multi_bar_panel":
        return MultiBarPanelProvider(config.features.params)
    raise ValueError(f"unsupported features.type: {config.features.type}")


def _bar_panel_progress(split_name: str, current: int, total: int, trade_date: int):
    def emit(message: str) -> None:
        print(
            f"bar-panel {split_name} [{current}/{total}] target={trade_date} {message}",
            flush=True,
        )

    return emit


def _sequence_dates(
    calendar: TradingCalendar,
    target_date: int,
    frequency: str,
    sequence_length: int,
) -> list[int]:
    candidates = sample_dates(calendar, (calendar.dates[0], target_date), frequency)
    candidates = [date for date in candidates if date <= target_date]
    return candidates[-sequence_length:]


def _sequence_feature_columns(columns: list[str], sequence_length: int) -> list[str]:
    return [_sequence_feature_column(column, step) for step in range(sequence_length) for column in columns]


def _sequence_feature_column(column: str, step: int) -> str:
    return f"{column}__seq{step}"
