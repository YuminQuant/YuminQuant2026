from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

from dataclasses import dataclass
from pathlib import Path
import json
import warnings
from time import perf_counter

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
from yq_ml_alpha.features.graph_factor_frame import FactorSequenceProvider, GraphFactorFrameProvider
from yq_ml_alpha.features.hist_factor_frame import HistFactorFrameProvider, ConceptFactorSequenceProvider
from yq_ml_alpha.features.raw_panel import RawPanelProvider
from yq_ml_alpha.features.rawbar import RawBarProvider
from yq_ml_alpha.features.resnet_panel import ResNetPanelProvider
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider
from yq_ml_alpha.features.derived_logsig import DerivedLogsigProvider
from yq_ml_alpha.features.derived_chip_age import DerivedChipAgeProvider
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.data.tensor_storage import TensorSpool, MemoryTensorSpool
from yq_ml_alpha.data.indexed_panel import IndexedPanelStore, IndexedPanelTensor
from yq_ml_alpha.features.transforms import apply_cross_section_transform


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    tensors: dict[str, np.ndarray | IndexedPanelTensor] | None = None
    tensor_columns: dict[str, list[str]] | None = None
    storage: TensorSpool | MemoryTensorSpool | IndexedPanelStore | None = None
    tensor_source: object | None = None
    full_cross_section: bool = False
    information_cutoff: int | None = None
    supervision_lookahead_days: int = 0

    def close(self) -> None:
        if self.storage is not None:
            self.storage.close()
            self.storage = None
            self.tensors = None


class DatasetBuilder:
    def __init__(self, config: MlAlphaConfig) -> None:
        self.config = config
        self.feature_provider = make_feature_provider(config)
        if hasattr(self.feature_provider, "set_calendar_dates"):
            self.feature_provider.set_calendar_dates(TradingCalendar.load(config.data_root).dates)
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self._source_st_symbol_cache: dict[int, set[str]] = {}

    def close(self, *, clear_read_cache=True):
        def clear(provider):
            sequence = getattr(provider, "sequence", None)
            if sequence is not None:
                print(f"sequence-cache hits={sequence.cache_hits} misses={sequence.cache_misses} "
                      f"uncached_reads={sequence.uncached_reads} "
                      f"resident_mib={sequence.cache_bytes / 2**20:.1f}", flush=True)
                sequence.close()
            for name in ("_session_cache", "_sessions", "_bar_cache", "_tensor_blocks"):
                cache = getattr(provider, name, None)
                if cache is not None:
                    cache.clear()
            for _, child in getattr(provider, "panels", []):
                clear(child)
        clear(self.feature_provider)
        self._source_st_symbol_cache.clear()
        if clear_read_cache:
            data_io.invalidate_parquet()

    def load(self, dates: list[int], include_label: bool, split_name: str | None = None) -> DatasetBundle:
        graph_tensor = isinstance(self.feature_provider, FactorSequenceProvider)
        lazy_tensor = isinstance(self.feature_provider, (DerivedChipAgeProvider, FactorSequenceProvider))
        if lazy_tensor and (self.config.preprocess.cross_section_transform != "none" or self.config.materialize.cache_samples):
            raise ValueError("lazy tensors require transform=none and cache_samples=false")
        frames = []
        total_dates = len(dates)
        split_name = split_name or ("labeled" if include_label else "predict")
        started = perf_counter()
        total_rows = 0

        def log_date(status, current, date, rows, date_started):
            if include_label:
                now = perf_counter()
                print(
                    f"ml-alpha load split={split_name} [{current}/{total_dates}] "
                    f"date={date} step={status} rows={rows} total_rows={total_rows} "
                    f"date_elapsed_s={now - date_started:.2f} elapsed_s={now - started:.2f}",
                    flush=True,
                )
        if isinstance(self.feature_provider, LogsigSignatureProvider):
            self.feature_provider.set_cache_days_for_target_dates(dates)
        for date_idx, trade_date in enumerate(dates, start=1):
            date_started = perf_counter()
            log_date("start", date_idx, trade_date, 0, date_started)
            if isinstance(self.feature_provider, LogsigSignatureProvider):
                progress = _logsig_signature_progress(split_name, date_idx, total_dates, trade_date)
                features = self.feature_provider.load(trade_date, progress=progress)
                progress(f"target_done feature_rows={len(features)}")
            else:
                features = self.feature_provider.load(trade_date)
            if features.empty:
                log_date("done", date_idx, trade_date, 0, date_started)
                continue
            frame = features if graph_tensor else self.filters.apply(self.universe.filter(features, trade_date), trade_date)
            if include_label:
                labels = self._read_labels(trade_date)
                frame = frame.merge(labels, on=["trade_date", "ts_code"], how="left")
            if not lazy_tensor:
                frame = self._preprocess(frame, include_label)
            if include_label and not graph_tensor:
                keep = self._valid_label_rows(frame, finite=lazy_tensor)
                frame = frame.loc[keep]
            frames.append(frame)
            total_rows += len(frame)
            log_date("done", date_idx, trade_date, len(frame), date_started)
        if frames:
            output = pd.concat(frames, ignore_index=True)
        else:
            columns = ["trade_date", "ts_code", *self.feature_provider.feature_columns]
            if include_label:
                columns.extend(self.config.label.columns)
            output = pd.DataFrame(columns=columns)
        self._maybe_cache(output, dates, include_label)
        if include_label:
            print(
                f"ml-alpha load split={split_name} step=complete dates={total_dates} "
                f"total_rows={len(output)} elapsed_s={perf_counter() - started:.2f}",
                flush=True,
            )
        return DatasetBundle(output, self.feature_provider.feature_columns, self.config.label.id,
                             tensor_source=self.feature_provider.sequence if lazy_tensor else None)

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
        total_dates = len(dates)
        split_name = "labeled" if include_label else "predict"

        for date_idx, trade_date in enumerate(dates, start=1):
            progress = _factor_sequence_progress(split_name, date_idx, total_dates, trade_date)
            sequence_dates = _sequence_dates(calendar, trade_date, sequence_frequency, sequence_length)
            for old_date in list(feature_cache):
                if sequence_dates and old_date < sequence_dates[0]:
                    del feature_cache[old_date]
            if len(sequence_dates) < sequence_length:
                progress(
                    f"step=skip insufficient_sequence got={len(sequence_dates)} required={sequence_length}"
                )
                continue
            progress(
                "step=sequence_dates "
                f"source_dates={sequence_dates[0]}..{sequence_dates[-1]} "
                f"sequence_length={sequence_length}"
            )

            progress("target step=read")
            target = self.feature_provider.load(trade_date)
            progress(f"target step=done rows={len(target)}")
            if target.empty:
                progress("target_done feature_rows=0")
                continue
            target = self.universe.filter(target[["trade_date", "ts_code"]].copy(), trade_date)
            target = self.filters.apply(target, trade_date)
            progress(f"target step=filter rows={len(target)}")
            frame = target[["trade_date", "ts_code"]].copy()

            for step, sequence_date in enumerate(sequence_dates):
                source_progress = (
                    lambda message, source_idx=step, source_date=sequence_date: progress(
                        f"source {source_idx + 1}/{sequence_length} date={source_date} {message}"
                    )
                )
                features = self._sequence_feature_frame(sequence_date, feature_cache, source_progress)
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
                labels = self._read_labels(trade_date)
                labels = self._preprocess_labels(labels)
                frame = frame.merge(
                    labels,
                    on=["trade_date", "ts_code"],
                    how="left",
                )
                frame = frame.loc[self._valid_label_rows(frame)]
                progress(f"label step=done rows={len(frame)}")
            progress(f"target_done feature_rows={len(frame)}")
            frames.append(frame)

        if frames:
            output = pd.concat(frames, ignore_index=True)
        else:
            columns = ["trade_date", "ts_code", *sequence_columns]
            if include_label:
                columns.extend(self.config.label.columns)
            output = pd.DataFrame(columns=columns)
        self._maybe_cache(output, dates, include_label)
        return DatasetBundle(output, sequence_columns, self.config.label.id)

    def load_bar_panel(
        self, dates, include_label, calendar,
    ) -> DatasetBundle:
        self._building_storage = None
        try:
            return self._load_bar_panel(dates, include_label, calendar)
        except BaseException:
            if self._building_storage is not None:
                self._building_storage.close()
            raise
        finally:
            self._building_storage = None

    def _load_bar_panel(
        self,
        dates: list[int],
        include_label: bool,
        calendar: TradingCalendar,
    ) -> DatasetBundle:
        if not isinstance(self.feature_provider, (BarPanelProvider, MultiBarPanelProvider, ResNetPanelProvider)):
            raise TypeError("load_bar_panel requires a bar tensor feature provider")
        frames = []
        tensor_chunks: list[dict[str, np.ndarray]] = []
        storage_mode = self.config.materialize.tensor_storage
        remote_memory = data_io.remote_enabled()
        if remote_memory:
            storage_mode = 'memory'
        if storage_mode not in {"indexed", "mmap", "memory"}:
            raise ValueError("materialize.tensor_storage must be indexed, mmap or memory")
        spool = TensorSpool(Path(self.config.materialize.cache_dir) / "scratch") if storage_mode == "mmap" and include_label else None
        if remote_memory:
            spool = MemoryTensorSpool()
        self._building_storage = spool
        tensor_columns: dict[str, list[str]] | None = None
        total_dates = len(dates)
        split_name = "labeled" if include_label else "predict"
        if hasattr(self.feature_provider, "set_cache_sessions_for_target_dates"):
            self.feature_provider.set_cache_sessions_for_target_dates(dates, calendar.dates)
        if hasattr(self.feature_provider, "cache_policy_summary"):
            print(
                f"bar-panel {split_name} cache_auto {self.feature_provider.cache_policy_summary()}",
                flush=True,
            )
        indexed = IndexedPanelStore(
            Path(self.config.materialize.cache_dir) / "scratch", self.feature_provider,
            lambda frame, tensors: self._preprocess_tensors(frame, tensors, False)[1],
        ) if storage_mode == "indexed" and include_label else None
        self._building_storage = indexed if indexed is not None else spool
        provider = indexed.provider if indexed is not None else self.feature_provider
        shapes = {}
        for date_idx, trade_date in enumerate(dates, start=1):
            history_dates = calendar.between(calendar.dates[0], trade_date)
            source_dates = self.feature_provider.required_history_dates(history_dates)
            progress = _bar_panel_progress(split_name, date_idx, total_dates, trade_date)
            source_st = self._source_st_symbols_by_date(source_dates)
            window = provider.load_window_tensor(
                trade_date,
                history_dates,
                exclude_bj=self.config.filters.exclude_bj,
                st_symbols_by_date=source_st,
                progress=progress,
            )
            progress(f"target_done feature_rows={len(window.frame)}")
            if window.frame.empty:
                continue
            frame = self.universe.filter(window.frame, trade_date)
            frame = self.filters.apply(frame, trade_date)
            frame = frame[["trade_date", "ts_code"]].reset_index(drop=True)
            eligible_symbols = frame.ts_code.tolist()
            tensors = _select_tensors_by_symbols(
                window.tensors,
                window.frame["ts_code"].astype(str).tolist(),
                frame["ts_code"].astype(str).tolist(),
            )
            if include_label:
                labels = self._read_labels(trade_date)
                frame = frame.merge(
                    labels,
                    on=["trade_date", "ts_code"],
                    how="left",
                )
            frame, tensors = self._preprocess_tensors(frame, tensors, include_label)
            if include_label and not self.feature_provider.requires_full_cross_section:
                keep = self._valid_label_rows(frame).to_numpy()
                frame = frame.loc[keep].reset_index(drop=True)
                tensors = _take_tensor_rows(tensors, keep)
            else:
                frame = frame.reset_index(drop=True)
            frames.append(frame)
            if indexed is not None:
                indexed.add(trade_date, history_dates,
                            dict(exclude_bj=self.config.filters.exclude_bj, st_symbols_by_date=source_st),
                            eligible_symbols, frame.ts_code.tolist())
                shapes = {key: value.shape[1:] for key, value in tensors.items()}
            elif spool is not None:
                spool.append(tensors)
            else:
                tensor_chunks.append(tensors)
            tensor_columns = window.tensor_columns
        if frames:
            output = pd.concat(frames, ignore_index=True)
            output_tensors = (indexed.finish(output, shapes) if indexed is not None else
                              spool.finish() if spool is not None else _concat_tensor_chunks(tensor_chunks))
        else:
            columns = ["trade_date", "ts_code"]
            if include_label:
                columns.extend(self.config.label.columns)
            output = pd.DataFrame(columns=columns)
            output_tensors, tensor_columns = _empty_provider_tensors(self.feature_provider)
        self._maybe_cache(output, dates, include_label)
        return DatasetBundle(
            output,
            self.feature_provider.feature_columns,
            self.config.label.id,
            tensors=output_tensors,
            tensor_columns=tensor_columns,
            storage=indexed if indexed is not None else spool,
            full_cross_section=self.feature_provider.requires_full_cross_section,
        )

    def _source_st_symbols_by_date(self, dates: list[int]) -> dict[int, set[str]]:
        if not self.config.filters.exclude_st:
            return {}
        output: dict[int, set[str]] = {}
        for trade_date in dates:
            if trade_date < 20160101:
                continue
            path = (
                Path(self.config.data_root)
                / "chn_stock_data"
                / "daily"
                / "trade_filter"
                / str(trade_date // 10000)
                / f"{trade_date}.parquet"
            )
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing trade_filter for {trade_date}: {path}")
            if trade_date not in self._source_st_symbol_cache:
                mask = data_io.read_parquet(path, columns=["ts_code", "is_st"])
                st_mask = mask["is_st"].fillna(False).astype(bool)
                self._source_st_symbol_cache[trade_date] = set(mask.loc[st_mask, "ts_code"].astype(str))
            symbols = self._source_st_symbol_cache[trade_date]
            if symbols:
                output[trade_date] = symbols
        return output

    def _read_labels(self, trade_date: int) -> pd.DataFrame:
        return read_daily(self.config.label.root, trade_date, self.config.label.columns)

    def _valid_label_rows(self, frame: pd.DataFrame, *, finite: bool = False) -> pd.Series:
        if self.config.label.auxiliary:
            # Multi-task training uses only stocks whose complete target vector
            # exists. This runs after feature transforms, preserving the same
            # feature cross-section regardless of future-label availability.
            labels = frame[self.config.label.columns].apply(pd.to_numeric, errors="coerce")
            return labels.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        target = frame[self.config.label.id]
        return np.isfinite(target) if finite else target.notna()

    def _preprocess(self, frame: pd.DataFrame, include_label: bool) -> pd.DataFrame:
        columns = list(self.feature_provider.feature_columns)
        label_columns = [column for column in self.config.label.columns if include_label and column in frame.columns]
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
            label_columns=[column for column in self.config.label.columns if column in frame.columns],
            feature_fill_value=self.config.preprocess.feature_fill_value,
        )

    def _preprocess_tensors(
        self,
        frame: pd.DataFrame,
        tensors: dict[str, np.ndarray],
        include_label: bool,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        transform = self.config.preprocess.cross_section_transform.strip().lower()
        label_frame = self._preprocess_labels(frame) if include_label else frame.copy()
        if transform in {"", "none"}:
            return label_frame, {
                key: _fill_tensor_features(value, self.config.preprocess.feature_fill_value)
                for key, value in tensors.items()
            }
        if transform in {"zscore", "cs_zscore"}:
            return label_frame, {
                key: _zscore_tensor_features(value, self.config.preprocess.feature_fill_value)
                for key, value in tensors.items()
            }
        return self._preprocess_tensors_via_frame(label_frame, tensors)

    def _preprocess_tensors_via_frame(
        self,
        frame: pd.DataFrame,
        tensors: dict[str, np.ndarray],
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        flat, columns_by_key = _flatten_tensors(frame, tensors)
        processed = apply_cross_section_transform(
            flat,
            self.config.preprocess.cross_section_transform,
            [column for columns in columns_by_key.values() for column in columns],
            label_columns=[],
            feature_fill_value=self.config.preprocess.feature_fill_value,
        )
        return (
            processed[[column for column in frame.columns]].copy(),
            _unflatten_tensors(processed, columns_by_key, tensors),
        )

    def _sequence_feature_frame(
        self,
        trade_date: int,
        cache: dict[int, pd.DataFrame],
        progress=None,
    ) -> pd.DataFrame:
        if trade_date in cache:
            if progress is not None:
                progress(f"cache=hit rows={len(cache[trade_date])}")
            return cache[trade_date]
        if progress is not None:
            progress("cache=miss step=read")
        features = self.feature_provider.load(trade_date)
        if progress is not None:
            progress(f"step=read_done rows={len(features)} columns={len(self.feature_provider.feature_columns)}")
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
        if progress is not None:
            progress(f"step=done rows={len(frame)}")
        cache[trade_date] = frame
        return frame

    def _maybe_cache(self, frame: pd.DataFrame, dates: list[int], include_label: bool) -> None:
        if not self.config.materialize.cache_samples or not dates:
            return
        cache_dir = Path(self.config.materialize.cache_dir)
        data_io.mkdir(cache_dir, parents=True, exist_ok=True)
        split = "labeled" if include_label else "predict"
        data_io.write_parquet(frame, cache_dir / f"{split}_{dates[0]}_{dates[-1]}.parquet", index=False)


def make_feature_provider(config: MlAlphaConfig) -> FeatureProvider:
    if config.features.type == "resnet_panel":
        return ResNetPanelProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "rawbar":
        return RawBarProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "derived_chip_age":
        return DerivedChipAgeProvider(config.features.root, config.features.params)
    if config.features.type == "derived_logsig":
        return DerivedLogsigProvider(config.features.root)
    if config.features.type in {"factor_frame", "factor_sequence", "graph_factor_frame", "hist_factor_frame", "concept_factor_sequence"}:
        excluded = {config.output.id}
        metadata = Path(config.data_root) / "factors" / "factor_metadata.parquet"
        if data_io.exists(metadata):
            for row in data_io.read_parquet(metadata, columns=["output_column", "tags_json"]).itertuples(index=False):
                tags = json.loads(row.tags_json or "[]")
                if "model_generated" in tags or "deprecated" in tags:
                    excluded.add(row.output_column)
        provider = FactorFrameProvider(config.features.root, config.features.columns, config.dates.train[1], excluded)
        if config.features.type == "factor_sequence":
            return FactorSequenceProvider(config, provider)
        if config.features.type == "graph_factor_frame":
            return GraphFactorFrameProvider(config, provider)
        if config.features.type == "hist_factor_frame":
            return HistFactorFrameProvider(config, provider)
        if config.features.type == "concept_factor_sequence":
            return ConceptFactorSequenceProvider(config, provider)
        return provider
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


def _logsig_signature_progress(split_name: str, current: int, total: int, trade_date: int):
    def emit(message: str) -> None:
        print(
            f"logsig-signature {split_name} [{current}/{total}] target={trade_date} {message}",
            flush=True,
        )

    return emit


def _factor_sequence_progress(split_name: str, current: int, total: int, trade_date: int):
    def emit(message: str) -> None:
        print(
            f"factor-sequence {split_name} [{current}/{total}] target={trade_date} {message}",
            flush=True,
        )

    return emit


def _select_tensors_by_symbols(
    tensors: dict[str, np.ndarray],
    source_symbols: list[str],
    target_symbols: list[str],
) -> dict[str, np.ndarray]:
    source_pos = {symbol: idx for idx, symbol in enumerate(source_symbols)}
    source_indices = [source_pos[symbol] for symbol in target_symbols]
    return {key: value[np.asarray(source_indices, dtype=np.int64)] for key, value in tensors.items()}


def _take_tensor_rows(tensors: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[keep] for key, value in tensors.items()}


def _concat_tensor_chunks(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        return {}
    keys = list(chunks[0])
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _empty_provider_tensors(provider) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    if hasattr(provider, "empty_tensor_window"):
        window = provider.empty_tensor_window()
        return window.tensors, window.tensor_columns
    return {}, {}


def _fill_tensor_features(values: np.ndarray, fill_value: float) -> np.ndarray:
    data = np.asarray(values, dtype="float64")
    data = np.where(np.isfinite(data), data, float(fill_value))
    return data.astype("float32", copy=False)


def _zscore_tensor_features(values: np.ndarray, fill_value: float) -> np.ndarray:
    data = np.asarray(values, dtype="float64")
    finite = np.isfinite(data)
    masked = np.where(finite, data, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(masked, axis=0)
            std = np.nanstd(masked, axis=0)
        transformed = (data - mean[None, :, :]) / std[None, :, :]
    bad_std = (~np.isfinite(std)) | (std <= 0.0)
    if np.any(bad_std):
        bad = bad_std[None, :, :]
        transformed = np.where(bad & finite, 0.0, transformed)
    transformed = np.where(finite, transformed, np.nan)
    transformed = np.where(np.isfinite(transformed), transformed, float(fill_value))
    return transformed.astype("float32", copy=False)


def _flatten_tensors(frame: pd.DataFrame, tensors: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    output = frame.copy()
    columns_by_key: dict[str, list[str]] = {}
    for key, tensor in tensors.items():
        columns = [
            f"{key}__t{step:03d}__f{feature:03d}"
            for step in range(tensor.shape[1])
            for feature in range(tensor.shape[2])
        ]
        output[columns] = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])
        columns_by_key[key] = columns
    return output, columns_by_key


def _unflatten_tensors(
    frame: pd.DataFrame,
    columns_by_key: dict[str, list[str]],
    original: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    for key, columns in columns_by_key.items():
        shape = original[key].shape
        values = frame[columns].to_numpy(dtype="float32", copy=False)
        tensors[key] = values.reshape(shape[0], shape[1], shape[2]).astype("float32", copy=False)
    return tensors


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
