"""Assemble globally normalized 21-day paths from daily homogeneous features."""
from __future__ import annotations

from itertools import product

import numpy as np

from yq_ml_alpha.data.daily_tensor_sequence import DailyTensorSequence
from yq_ml_alpha.data.filters import TradeFilters
from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.data.universe import Universe
from yq_ml_alpha.features.daily_sequence import DailySequenceProvider
from yq_ml_alpha.output import artifact_io


WIDTH = 728
INTRA = [f"intra_{i:04}" for i in range(1, WIDTH + 1)]
BRIDGE = [f"bridge_{i:04}" for i in range(1, WIDTH + 1)]
STATS = ["upper_mean", "upper_m2", "lower_mean", "lower_m2", "volume_sum"]
FEATURES = [f"logsig_{i:04}" for i in range(1, WIDTH + 1)]


def lyndon_words():
    # Same degree/lexicographic ordering and standard brackets as Rust.
    return [word for degree in range(1, 5) for word in product(range(7), repeat=degree)
            if all(word < word[shift:] + word[:shift] for shift in range(1, degree))]


class LogsigRnnHistory(DailyTensorSequence):
    def __init__(self, config):
        params = config.features.params
        if config.features.columns != FEATURES or params["sequence_length"] != 21:
            raise ValueError("derived_logsigrnn requires 21 sessions and 728 ordered Lyndon coordinates")
        super().__init__(config.features.root, FEATURES, (WIDTH,), length=21,
                         memory_budget_mb=params.get("memory_budget_mb"))
        self.filters = TradeFilters(config.filters, config.data_root)
        self.universe = Universe(config.universe.id, config.data_root)
        workers, prefetch = int(params["history_read_workers"]), int(params["history_prefetch_mb"])
        if workers < 1 or prefetch < 1:
            raise ValueError("Logsig-RNN history workers/prefetch must be positive")
        self.read_workers = workers if artifact_io.remote_enabled() else 1
        self.prefetch_bytes = prefetch * 1024**2
        self._counts = np.array([[word.count(axis) for axis in (3, 4, 5)] for word in lyndon_words()])
        if self._counts.shape != (WIDTH, 3):
            raise ValueError("Logsig-RNN basis width mismatch")
        self._window_std = self._volume = None

    def _load_day(self, date):
        path = daily_path(self.root, date)
        frame = artifact_io.read_parquet(path, columns=["trade_date", "ts_code", "definition_version", *INTRA, *BRIDGE, *STATS])
        if (frame.ts_code.isna().any() or frame.ts_code.duplicated().any()
                or not frame.trade_date.eq(date).all() or not frame.definition_version.eq(1).all()):
            raise ValueError(f"invalid Logsig-RNN keys/definition: {path}; regenerate derive-logsigrnn")
        frame = self.filters.apply(self.universe.filter(frame, date), date).set_index("ts_code")
        frame = frame[[*INTRA, *BRIDGE, *STATS]].copy()
        frame[INTRA + BRIDGE] = frame[INTRA + BRIDGE].astype("float32")
        frame[STATS] = frame[STATS].astype("float64")
        valid = np.isfinite(frame[INTRA + STATS]).all(axis=1)
        valid &= frame[["upper_m2", "lower_m2", "volume_sum"]].ge(0).all(axis=1)
        return frame.loc[valid]

    def _activate(self, date):
        if self._active_date != date:
            self._window_std = self._volume = None
        super()._activate(date)

    def symbols(self, date):
        symbols = super().symbols(date)
        if not symbols or self._window_std is not None:
            return symbols
        mean = np.zeros((len(symbols), 2), dtype="float64")
        m2 = np.zeros_like(mean)
        volume = np.zeros(len(symbols), dtype="float64")
        keep = np.ones(len(symbols), dtype=bool)
        # Merge per-day Welford moments: exact population variance of all 1008
        # observed bars, without storing or revisiting the original minutes.
        for i, (day, frame) in enumerate(self._days(self.window_dates(date))):
            positions = self._positions_by_day[day]
            stats = frame[STATS].to_numpy(copy=False)[positions]
            current = stats[:, [0, 2]]
            delta = current - mean
            mean += delta / (i + 1)
            m2 += stats[:, [1, 3]] + delta**2 * (48 * i / (i + 1))
            volume += stats[:, 4]
            if i:
                keep &= np.isfinite(frame[BRIDGE].to_numpy(copy=False)[positions]).all(axis=1)
        std = np.sqrt(np.maximum(m2 / (48 * self.length), 0.0))
        keep &= np.isfinite(std).all(axis=1) & np.isfinite(volume) & (volume > 0)
        self._symbols = self._symbols[keep]
        self._positions_by_day = {day: positions[keep] for day, positions in self._positions_by_day.items()}
        self._window_std, self._volume = std[keep], volume[keep]
        return self._symbols.tolist()

    def get(self, date, symbols):
        self.symbols(date)
        if len(self.window_dates(date)) != self.length:
            raise ValueError(f"insufficient 21-session Logsig-RNN history on {date}")
        selected = self._symbols.get_indexer(symbols)
        if (selected < 0).any():
            raise ValueError(f"requested stocks lack complete Logsig-RNN history on {date}")
        output = np.empty((len(symbols), self.length, WIDTH), dtype="float32")
        for i, (day, frame) in enumerate(self._days(self.window_dates(date))):
            positions = self._positions_by_day[day][selected]
            stats = frame[STATS].to_numpy(copy=False)[positions]
            daily_std = np.sqrt(stats[:, [1, 3]] / 48)
            daily_scale = np.where(daily_std > 0, daily_std, 1.0)
            window_std = self._window_std[selected]
            scales = np.zeros((len(symbols), 3), dtype="float64")
            scales[:, :2] = np.divide(daily_scale, window_std, out=np.zeros_like(daily_scale), where=window_std > 0)
            scales[:, 2] = np.where(stats[:, 4] > 0, stats[:, 4], 1.0) / self._volume[selected]
            # Every standard Lyndon bracket is multihomogeneous. A coordinate
            # rescales by the product of its letters' channel scale factors.
            coefficients = frame[INTRA if i == 0 else BRIDGE].to_numpy(copy=False)[positions]
            factors = np.prod(scales[:, None, :] ** self._counts[None, :, :], axis=2)
            output[:, i] = coefficients * factors
        if not np.isfinite(output).all():
            raise ValueError(f"nonfinite normalized Logsig-RNN inputs on {date}")
        return output

    def close(self):
        super().close()
        self._window_std = self._volume = None


class DerivedLogsigRnnProvider(DailySequenceProvider):
    def __init__(self, config):
        super().__init__(LogsigRnnHistory(config))

    def dataset_identity(self):
        return {"derived_logsigrnn": "seven_channel_order4_lyndon_v1", "sequence_length": 21,
                "normalization": "whole_window_shadow_std_and_volume_sum", "boundary": "shared_previous_session_last_point"}
