from __future__ import annotations
from yq_ml_alpha.output import artifact_io as data_io

from bisect import bisect_right
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
import warnings
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import MlAlphaConfig
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.column_catalog import FACTOR_FRAME_TYPES, factor_columns
from yq_ml_alpha.data.future_pool import FutureLiquidityPool, future_products
from yq_ml_alpha.data.stores import daily_path, read_daily, read_minute, require_hdfs_input
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.base import FeatureProvider
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.features.market_style_sequence import MarketStyleSequenceProvider
from yq_ml_alpha.features.kd_components import KDComponentsProvider
from yq_ml_alpha.features.rawbar_sequence import RawBarSequenceProvider
from yq_ml_alpha.features.derived_logsigrnn import DerivedLogsigRnnProvider
from yq_ml_alpha.features.bar_panel import BarPanelProvider, MultiBarPanelProvider
from yq_ml_alpha.features.factor_frame import FactorFrameProvider, MinuteFactorFrameProvider
from yq_ml_alpha.features.future_sector_ic import apply_future_sector_ic, fit_future_sector_ic
from yq_ml_alpha.features.graph_factor_frame import FactorSequenceProvider, GraphFactorFrameProvider
from yq_ml_alpha.features.analyst_factor_frame import AnalystFactorFrameProvider
from yq_ml_alpha.features.derived_analyst_gat import DerivedAnalystGATProvider
from yq_ml_alpha.features.stock_events import StockEventsProvider
from yq_ml_alpha.features.hist_factor_frame import HistFactorFrameProvider, ConceptFactorSequenceProvider
from yq_ml_alpha.features.raw_panel import RawPanelProvider
from yq_ml_alpha.features.rawbar import RawBarProvider
from yq_ml_alpha.features.resnet_panel import ResNetPanelProvider
from yq_ml_alpha.features.derived_logsig import DerivedLogsigProvider
from yq_ml_alpha.features.derived_chip_age import DerivedChipAgeProvider
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.data.tensor_storage import ChunkedTensor, TensorSpool, MemoryTensorSpool
from yq_ml_alpha.data.indexed_panel import IndexedPanelStore, IndexedPanelTensor
from yq_ml_alpha.features.transforms import apply_cross_section_transform
from yq_ml_alpha.models.base import load_model_class
from yq_ml_alpha.output.parquet_reader import parquet_input_snapshot


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    tensors: dict[str, np.ndarray | IndexedPanelTensor | ChunkedTensor] | None = None
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
    def __init__(self, config: MlAlphaConfig, *, calendar: TradingCalendar | None = None) -> None:
        self.config = config
        self.feature_provider = make_feature_provider(config)
        self.minute_frame = isinstance(self.feature_provider, MinuteFactorFrameProvider)
        self.future_pool = None
        self.future_sector_ic = None
        if self.minute_frame:
            if config.filters.exclude_limit or config.filters.exclude_st or config.filters.exclude_bj:
                raise ValueError("minute_factor_frame requires disabling the stock filters")
            if config.preprocess.cross_section_transform.strip().lower() not in {"", "none"}:
                raise ValueError("minute_factor_frame requires cross_section_transform='none'")
            if config.train_scheme.validation_unit != "dates":
                raise ValueError("minute_factor_frame requires date-based validation")
            if config.filters.future_liquidity_enabled:
                self.future_pool = FutureLiquidityPool(
                    config.data_root, config.filters.future_pool_id,
                    require_hdfs=self.feature_provider.require_hdfs,
                )
        if hasattr(self.feature_provider, "set_calendar_dates"):
            self.feature_provider.set_calendar_dates((calendar or TradingCalendar.for_config(config)).dates)
        self.universe = Universe(config.universe.id, config.data_root)
        self.filters = TradeFilters(config.filters, config.data_root)
        self._source_st_symbol_cache: dict[int, set[str]] = {}

    def fit_window_preprocess(self, train_dates: list[int]) -> None:
        if self.config.preprocess.future_sector_ic_align:
            self.future_sector_ic = fit_future_sector_ic(
                self.config, self.feature_provider.feature_columns, train_dates,
            )

    def close(self, *, clear_read_cache=True):
        def clear(provider):
            sequence = getattr(provider, "sequence", None)
            if sequence is not None:
                print(f"sequence-cache hits={sequence.cache_hits} misses={sequence.cache_misses} "
                      f"uncached_reads={sequence.uncached_reads} "
                      f"resident_mib={sequence.cache_bytes / 2**20:.1f}", flush=True)
                sequence.close()
            for name in ("_session_cache", "_sessions", "_bar_cache", "_tensor_blocks", "_tensor_symbols"):
                cache = getattr(provider, name, None)
                if cache is not None:
                    cache.clear()
            for _, child in getattr(provider, "panels", []):
                clear(child)
        clear(self.feature_provider)
        close = getattr(self.feature_provider, "close", None)
        if callable(close):
            close()
        self._source_st_symbol_cache.clear()
        if clear_read_cache:
            data_io.invalidate_parquet()

    @parquet_input_snapshot()
    def load(self, dates: list[int], include_label: bool, split_name: str | None = None) -> DatasetBundle:
        if self.future_pool is not None:
            self.future_pool.prepare(dates)
        if self.config.preprocess.future_sector_ic_align and self.future_sector_ic is None:
            raise ValueError("futures sector IC preprocessing requires the window's fitted directions")
        if dates and include_label and self.minute_frame and self.feature_provider.require_hdfs:
            require_hdfs_input(self.config.label.root)
        graph_tensor = isinstance(self.feature_provider, DailySequenceProvider)
        lazy_tensor = isinstance(self.feature_provider, (DerivedChipAgeProvider, DailySequenceProvider))
        if lazy_tensor and self.config.preprocess.cross_section_transform != "none":
            raise ValueError("lazy tensors require transform=none")
        frames = []
        total_dates = len(dates)
        split_name = split_name or ("labeled" if include_label else "predict")
        started = perf_counter()
        preloaded = None
        batch_load = getattr(self.feature_provider, "load_many", None)
        if callable(batch_load):
            def load_batches():
                with closing(batch_load(dates)) as frames:
                    for frame in frames:
                        yield frame, None
            preloaded = load_batches()
        elif type(self.feature_provider) in (FactorFrameProvider, MinuteFactorFrameProvider, RawPanelProvider):
            root = self.feature_provider.root
            if data_io.backend(root) is not None:
                def load_input(date):
                    features = self.feature_provider.load(date)
                    labels = (self._read_labels(date)
                              if include_label and (self.minute_frame or not features.empty) else None)
                    return features, labels
                preloaded = data_io.ordered_io_map(load_input, dates, root=root)
        try:
            prefetch_targets = getattr(getattr(self.feature_provider, "sequence", None), "prefetch_targets", None)
            sequence_labels = (
                self._prefetch_sequence_labels(dates)
                if (prefetch_targets is not None or callable(batch_load)) and include_label
                and data_io.backend(self.config.label.root) is not None
                else {}
            )
            with prefetch_targets(dates) if prefetch_targets is not None else nullcontext():
                return self._load_dates(dates, include_label, split_name, graph_tensor, lazy_tensor,
                                        frames, total_dates, started, preloaded, sequence_labels)
        finally:
            if preloaded is not None: preloaded.close()

    def _prefetch_sequence_labels(self, dates):
        def load(date):
            try:
                return date, self._read_labels(date)
            except Exception as error:
                # A date with no eligible stocks did not require a label file.
                return date, error

        started = perf_counter()
        with closing(data_io.ordered_io_map(load, dict.fromkeys(dates), root=self.config.label.root)) as reads:
            labels = dict(reads)
        print(f"sequence-labels dates={len(labels)} load_s={perf_counter() - started:.2f}", flush=True)
        return labels

    def _load_dates(self, dates, include_label, split_name, graph_tensor, lazy_tensor,
                    frames, total_dates, started, preloaded, sequence_labels):
        total_rows = 0
        pool_rows_before = pool_rows_after = 0
        pool_contracts_before: set[str] = set()
        pool_contracts_after: set[str] = set()
        def log_date(status, current, date, rows, date_started):
            if include_label:
                now = perf_counter()
                print(
                    f"ml-alpha load split={split_name} [{current}/{total_dates}] "
                    f"date={date} step={status} rows={rows} total_rows={total_rows} "
                    f"date_elapsed_s={now - date_started:.2f} elapsed_s={now - started:.2f}",
                    flush=True,
                )
        for date_idx, trade_date in enumerate(dates, start=1):
            date_started = perf_counter()
            labels = None
            log_date("start", date_idx, trade_date, 0, date_started)
            if preloaded is not None:
                features, labels = next(preloaded)
            else:
                features = self.feature_provider.load(trade_date)
            if self.future_pool is not None:
                pool_rows_before += len(features)
                pool_contracts_before.update(features["ts_code"].unique())
                features = self.future_pool.filter(features, trade_date)
                pool_rows_after += len(features)
                pool_contracts_after.update(features["ts_code"].unique())
            if features.empty and not (self.minute_frame and include_label):
                log_date("done", date_idx, trade_date, 0, date_started)
                continue
            frame = features if graph_tensor else self.filters.apply(self.universe.filter(features, trade_date), trade_date)
            if include_label:
                if labels is None:
                    if trade_date in sequence_labels:
                        labels = sequence_labels[trade_date]
                        if isinstance(labels, Exception):
                            raise labels
                    else:
                        labels = self._read_labels(trade_date)
                if self.minute_frame:
                    feature_rows = len(frame)
                    frame = frame.merge(labels, on=self.config.key_columns, how="inner", validate="one_to_one")
                    if feature_rows and frame.empty:
                        raise ValueError(f"no matching minute factor and label keys on {trade_date}")
                    if len(frame) < feature_rows:
                        log_date(f"matched_label_keys_of_{feature_rows}", date_idx, trade_date, len(frame), date_started)
                else:
                    frame = frame.merge(labels, on=self.config.key_columns, how="left")
            if not lazy_tensor:
                frame = self._preprocess(frame, include_label)
            if include_label and not graph_tensor:
                keep = self._valid_label_rows(frame, finite=lazy_tensor or self.minute_frame)
                frame = frame.loc[keep]
            frames.append(frame)
            total_rows += len(frame)
            log_date("done", date_idx, trade_date, len(frame), date_started)
        if frames:
            output = pd.concat(frames, ignore_index=True)
        else:
            columns = [*self.config.key_columns, *self.feature_provider.feature_columns]
            if include_label:
                columns.extend(self.config.label.columns)
            output = pd.DataFrame(columns=columns)
        if self.future_pool is not None and dates:
            products_before = future_products(pd.Series(list(pool_contracts_before), dtype="string")).nunique()
            products_after = future_products(pd.Series(list(pool_contracts_after), dtype="string")).nunique()
            print(f"ml-alpha future pool={self.config.filters.future_pool_id} split={split_name} "
                  f"dates={dates[0]}..{dates[-1]} rows={pool_rows_before}->{pool_rows_after} "
                  f"contracts={len(pool_contracts_before)}->{len(pool_contracts_after)} "
                  f"products={products_before}->{products_after}", flush=True)
        if self.minute_frame and dates and include_label and output.empty:
            raise ValueError(f"no usable minute labels for {split_name} {dates[0]}..{dates[-1]}")
        if include_label:
            print(
                f"ml-alpha load split={split_name} step=complete dates={total_dates} "
                f"total_rows={len(output)} elapsed_s={perf_counter() - started:.2f}",
                flush=True,
            )
        return DatasetBundle(output, self.feature_provider.feature_columns, self.config.label.id,
                             tensor_source=self.feature_provider.sequence if lazy_tensor else None)

    @parquet_input_snapshot()
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
            target = self._sequence_feature_frame(trade_date, feature_cache).reset_index()
            progress(f"target step=done rows={len(target)}")
            if target.empty:
                progress("target_done feature_rows=0")
                continue
            target = self.universe.filter(target[["trade_date", "ts_code"]].copy(), trade_date)
            target = self.filters.apply(target, trade_date)
            progress(f"target step=filter rows={len(target)}")
            frame = target[["trade_date", "ts_code"]].reset_index(drop=True)
            input_blocks = []

            for step, sequence_date in enumerate(sequence_dates):
                source_progress = (
                    lambda message, source_idx=step, source_date=sequence_date: progress(
                        f"source {source_idx + 1}/{sequence_length} date={source_date} {message}"
                    )
                )
                features = self._sequence_feature_frame(sequence_date, feature_cache, source_progress)
                values = features[base_columns].reindex(frame.ts_code)
                input_blocks.append(values.set_axis(
                    [_sequence_feature_column(column, step) for column in base_columns], axis=1
                ).reset_index(drop=True))

            values = pd.concat(input_blocks, axis=1).replace([np.inf, -np.inf], np.nan).fillna(
                self.config.preprocess.feature_fill_value
            )
            frame = pd.concat([frame, values], axis=1)
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
        return DatasetBundle(output, sequence_columns, self.config.label.id)

    @parquet_input_snapshot()
    def load_bar_panel(
        self, dates, include_label, calendar,
    ) -> DatasetBundle:
        self._building_storage = None
        try:
            labels = (self._prefetch_sequence_labels(dates) if include_label
                      and data_io.backend(self.config.label.root) is not None else {})
            with self._prefetch_bar_panel(dates, include_label, calendar):
                return self._load_bar_panel(dates, include_label, calendar, labels)
        except BaseException:
            if self._building_storage is not None:
                self._building_storage.close()
            raise
        finally:
            self._building_storage = None

    @contextmanager
    def _prefetch_bar_panel(self, dates, include_label, calendar):
        provider = self.feature_provider
        if (not dates or not isinstance(provider, ResNetPanelProvider)
                or not data_io.remote_enabled() or dates != sorted(dates)):
            yield
            return
        histories = {date: _bar_history(calendar, date, provider.lookback_sessions) for date in dates}
        started = perf_counter()
        if self.config.filters.exclude_st:
            st_dates = sorted({source for history in histories.values() for source in history if source >= 20160101})
            missing = self._load_source_st_symbols(st_dates)
            split = "labeled" if include_label else "predict"
            print(f"bar-panel {split} st_preload sessions={len(st_dates)} reads={missing} "
                  f"elapsed_s={perf_counter() - started:.3f}", flush=True)
        source_dates = sorted({
            source for date, history in histories.items()
            if len(history) == provider.lookback_sessions and history[-1] == date
            for source in history
        })
        with provider.prefetch_sessions(source_dates):
            yield

    def _load_bar_panel(
        self,
        dates: list[int],
        include_label: bool,
        calendar: TradingCalendar,
        preloaded_labels,
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
        if storage_mode == "memory":
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
        resnet_panel = isinstance(provider, ResNetPanelProvider)
        lookback = (max(panel.lookback_sessions for _, panel in provider.panels)
                    if isinstance(provider, MultiBarPanelProvider) else provider.lookback_sessions)
        shapes = {}
        for date_idx, trade_date in enumerate(dates, start=1):
            history_dates = _bar_history(calendar, trade_date, lookback)
            if resnet_panel:
                source_dates = history_dates
            else:
                source_dates = self.feature_provider.required_history_dates(history_dates)
            progress = _bar_panel_progress(split_name, date_idx, total_dates, trade_date)
            started = perf_counter()
            source_st = self._source_st_symbols_by_date(source_dates)
            if resnet_panel:
                progress(f"st_lookup_s={perf_counter() - started:.3f}")
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
            prepared = perf_counter()
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
                labels = preloaded_labels.get(trade_date)
                if isinstance(labels, Exception):
                    raise labels
                if labels is None:
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
            if resnet_panel:
                progress(f"target_prepare_s={perf_counter() - prepared:.3f} dataset_rows={len(frame)}")
        joined = perf_counter()
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
        if resnet_panel:
            print(f"bar-panel {split_name} batch rows={len(output)} "
                  f"finalize_s={perf_counter() - joined:.3f}", flush=True)
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
        self._load_source_st_symbols(dates)
        return {date: self._source_st_symbol_cache[date] for date in dates
                if self._source_st_symbol_cache.get(date)}

    def _load_source_st_symbols(self, dates: list[int]) -> int:
        cache = self._source_st_symbol_cache
        missing = sorted({date for date in dates if date >= 20160101 and date not in cache})
        if not missing:
            return 0
        root = Path(self.config.data_root) / "chn_stock_data" / "daily" / "trade_filter"

        def read_mask(date):
            path = daily_path(root, date)
            if not data_io.exists(path):
                raise FileNotFoundError(f"missing trade_filter for {date}: {path}")
            mask = data_io.read_parquet(path, columns=["ts_code", "is_st"])
            st_mask = mask["is_st"].fillna(False).astype(bool)
            return date, set(mask.loc[st_mask, "ts_code"].astype(str))

        reads = data_io.ordered_io_map(read_mask, missing, root=root)
        try:
            cache.update(reads)
        finally:
            reads.close()
        return len(missing)

    def _read_labels(self, trade_date: int) -> pd.DataFrame:
        if self.minute_frame:
            return read_minute(
                self.config.label.root, trade_date, self.config.label.columns,
                require_hdfs=self.feature_provider.require_hdfs,
            )
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
        if self.future_sector_ic is not None:
            frame = apply_future_sector_ic(frame, self.future_sector_ic)
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
        cache[trade_date] = frame.set_index("ts_code")
        return cache[trade_date]

def make_feature_provider(config: MlAlphaConfig) -> FeatureProvider:
    factory = getattr(load_model_class(config.model.class_path), "make_feature_provider", None)
    if callable(factory):
        provider = factory(config)
        if provider is not None:
            return provider
    if config.features.type == "kd_components":
        return KDComponentsProvider(config)
    if config.features.type == "minute_factor_frame":
        if set(config.label.columns).intersection([*config.key_columns, *config.features.columns]):
            raise ValueError("minute label columns must be separate from keys and feature columns")
        return MinuteFactorFrameProvider(
            config.features.root, config.features.columns,
            require_hdfs=config.features.params.get("require_hdfs", False),
        )
    if config.features.type == "derived_logsigrnn":
        return DerivedLogsigRnnProvider(config)
    if config.features.type == "rawbar_sequence":
        return RawBarSequenceProvider(config)
    if config.features.type == "market_style_sequence":
        return MarketStyleSequenceProvider(config)
    if config.features.type == "stock_events":
        return StockEventsProvider(config)
    if config.features.type == "analyst_raw40_sequence":
        provider = DerivedAnalystGATProvider(config.features.root, config.features.columns)
        return AnalystFactorFrameProvider(config, provider)
    if config.features.type == "resnet_panel":
        return ResNetPanelProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "rawbar":
        return RawBarProvider(config.features.root, config.features.columns, config.features.params)
    if config.features.type == "derived_chip_age":
        return DerivedChipAgeProvider(config.features.root, config.features.params)
    if config.features.type == "derived_logsig":
        return DerivedLogsigProvider(config.features.root)
    if config.features.type in FACTOR_FRAME_TYPES:
        provider = FactorFrameProvider(config.features.root, factor_columns(config))
        if config.features.type == "factor_sequence":
            return FactorSequenceProvider(config, provider)
        if config.features.type == "graph_factor_frame":
            return GraphFactorFrameProvider(config, provider)
        if config.features.type == "analyst_factor_sequence":
            return AnalystFactorFrameProvider(config, provider)
        if config.features.type == "hist_factor_frame":
            return HistFactorFrameProvider(config, provider)
        if config.features.type == "concept_factor_sequence":
            return ConceptFactorSequenceProvider(config, provider)
        return provider
    if config.features.type == "raw_panel":
        return RawPanelProvider(config.features.root, config.features.columns)
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


def _bar_history(calendar: TradingCalendar, date: int, lookback: int) -> list[int]:
    end = bisect_right(calendar.dates, date)
    return calendar.dates[max(0, end - lookback):end]


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
    if target_symbols == source_symbols:
        return tensors
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
    if values.dtype == np.float32 and np.isfinite(values).all():
        return values
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
