from __future__ import annotations

import importlib
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from yq_ml_alpha.data.stores import daily_path
from yq_ml_alpha.features.base import FeatureProvider

ProgressCallback = Callable[[str], None]


def signature_width(order: int) -> int:
    return logsignature_width(order)


def tensor_signature_width(order: int) -> int:
    if order <= 0:
        raise ValueError("logsig_signature order must be positive")
    return sum(2**level for level in range(1, order + 1))


def logsignature_width(order: int) -> int:
    return len(_lyndon_words(order))


class LogsigSignatureProvider(FeatureProvider):
    """Compute Logsig-Alpha-v signature features from cached 5m bar files."""

    def __init__(
        self,
        root: str | Path,
        columns: list[str] | str,
        params: dict[str, Any] | None = None,
    ) -> None:
        params = dict(params or {})
        self.root = Path(root)
        self.lookback_days = int(params.get("lookback_days", 20))
        self.bar_size = int(params.get("bar_size", 5))
        self.order = int(params.get("order", 10))
        self.volume_column = str(params.get("volume_column", "volume"))
        cache_days = params.get("cache_days", "auto")
        self.cache_days_auto = str(cache_days).strip().lower() in {"", "auto"}
        self.cache_days = (
            max(1, self.lookback_days - 1)
            if self.cache_days_auto
            else max(1, int(cache_days))
        )
        self.progress_interval = max(1, int(params.get("progress_interval", 500)))
        if self.lookback_days <= 0:
            raise ValueError("logsig_signature lookback_days must be positive")
        if self.bar_size <= 0 or 240 % self.bar_size != 0:
            raise ValueError("logsig_signature bar_size must divide 240")
        if self.order <= 0:
            raise ValueError("logsig_signature order must be positive")
        self.steps_per_day = 240 // self.bar_size
        self.window_steps = self.lookback_days * self.steps_per_day
        width = signature_width(self.order)
        self.feature_columns = [f"logsig_{idx:04}" for idx in range(1, width + 1)]
        self._bar_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._calendar_dates: list[int] = []

    def set_calendar_dates(self, dates: list[int]) -> None:
        self._calendar_dates = list(dates)

    def set_cache_days_for_target_dates(self, dates: list[int]) -> None:
        if not self.cache_days_auto:
            return
        self.cache_days = self._recommended_cache_days(dates)
        self._trim_cache()

    def load(self, trade_date: int, progress: ProgressCallback | None = None) -> pd.DataFrame:
        source_dates = self._source_dates(trade_date)
        if len(source_dates) < self.lookback_days:
            if progress is not None:
                progress(
                    f"source_days={len(source_dates)}/{self.lookback_days} cache_days={self.cache_days} "
                    "step=insufficient_history"
                )
            return self._empty_frame()
        if progress is not None:
            progress(
                f"source_days={len(source_dates)} window_steps={self.window_steps} "
                f"order={self.order} width={len(self.feature_columns)} cache_days={self.cache_days} step=read"
            )
        daily_frames = []
        for day_idx, source_date in enumerate(source_dates, start=1):
            frame = self._load_bar_day(
                source_date,
                progress=(
                    (lambda message, idx=day_idx: progress(f"source {idx}/{len(source_dates)} {message}"))
                    if progress is not None
                    else None
                ),
            )
            if frame.empty:
                if progress is not None:
                    progress(f"source {day_idx}/{len(source_dates)} date={source_date} step=empty")
                return self._empty_frame()
            daily_frames.append(frame)
        if progress is not None:
            progress(f"step=matrix_start source_days={len(daily_frames)}")
        (
            ts_codes,
            volume_matrix,
            skipped_incomplete,
            skipped_nonfinite,
            candidate_count,
        ) = self._build_volume_matrix(daily_frames)
        if progress is not None:
            progress(
                f"step=matrix_done candidates={candidate_count} stocks={len(ts_codes)} "
                f"shape={volume_matrix.shape[0]}x{volume_matrix.shape[1] if volume_matrix.ndim == 2 else 0} "
                f"skipped_incomplete={skipped_incomplete} skipped_nonfinite={skipped_nonfinite}"
            )
        if not ts_codes:
            if progress is not None:
                progress(
                    f"step=signature_done stocks=0 skipped_incomplete={skipped_incomplete} "
                    f"skipped_nonfinite={skipped_nonfinite}"
                )
            return self._empty_frame()
        data, backend = _signature_batch_from_volume(volume_matrix, self.order, progress)
        output = pd.DataFrame(data, columns=self.feature_columns)
        output.insert(0, "ts_code", ts_codes)
        output.insert(0, "trade_date", np.int32(trade_date))
        if progress is not None:
            progress(
                f"step=signature_done backend={backend} stocks={len(output)} skipped_incomplete={skipped_incomplete} "
                f"skipped_nonfinite={skipped_nonfinite}"
            )
        return output

    def _build_volume_matrix(
        self,
        daily_frames: list[pd.DataFrame],
    ) -> tuple[list[str], np.ndarray, int, int, int]:
        processed_frames: list[pd.DataFrame] = []
        candidate_symbols: set[str] = set()
        columns = ["ts_code", "bar_index", self.volume_column]
        for frame in daily_frames:
            daily = frame[columns]
            daily = daily.loc[
                (daily["bar_index"] >= 0) & (daily["bar_index"] < self.steps_per_day),
                columns,
            ]
            daily = daily.drop_duplicates(["ts_code", "bar_index"], keep="last")
            processed_frames.append(daily)
            if not daily.empty:
                candidate_symbols.update(daily["ts_code"].astype(str).unique())

        ts_codes = sorted(candidate_symbols)
        candidate_count = len(ts_codes)
        if not ts_codes:
            return [], np.empty((0, self.window_steps), dtype=np.float64), 0, 0, 0

        row_by_symbol = {symbol: idx for idx, symbol in enumerate(ts_codes)}
        matrix = np.full((candidate_count, self.window_steps), np.nan, dtype=np.float64)
        valid_counts = np.zeros(candidate_count, dtype=np.int32)
        for day_idx, daily in enumerate(processed_frames):
            if daily.empty:
                continue
            rows = daily["ts_code"].map(row_by_symbol).to_numpy(dtype=np.int64)
            bar_index = daily["bar_index"].to_numpy(dtype=np.int64, copy=False)
            cols = day_idx * self.steps_per_day + bar_index
            values = daily[self.volume_column].to_numpy(dtype=np.float64, copy=False)
            matrix[rows, cols] = values
            np.add.at(valid_counts, rows, 1)

        complete = valid_counts == self.window_steps
        finite = np.zeros(candidate_count, dtype=bool)
        if complete.any():
            finite[complete] = np.isfinite(matrix[complete]).all(axis=1)
        keep = complete & finite
        kept_indices = np.flatnonzero(keep)
        kept_symbols = [ts_codes[idx] for idx in kept_indices]
        kept_matrix = np.ascontiguousarray(matrix[kept_indices], dtype=np.float64)
        skipped_incomplete = int((~complete).sum())
        skipped_nonfinite = int((complete & ~finite).sum())
        return kept_symbols, kept_matrix, skipped_incomplete, skipped_nonfinite, candidate_count

    def _source_dates(self, trade_date: int) -> list[int]:
        if self._calendar_dates:
            history = [date for date in self._calendar_dates if date <= trade_date]
            return history[-self.lookback_days :]
        candidates = sorted(int(path.stem) for path in self.root.glob("*/*.parquet"))
        history = [date for date in candidates if date <= trade_date]
        return history[-self.lookback_days :]

    def _recommended_cache_days(self, dates: list[int]) -> int:
        if len(dates) < 2 or not self._calendar_dates:
            return max(1, self.lookback_days - 1)
        index_by_date = {date: idx for idx, date in enumerate(self._calendar_dates)}
        indices = [index_by_date[date] for date in dates if date in index_by_date]
        strides = [right - left for left, right in zip(indices, indices[1:]) if right > left]
        if not strides:
            return max(1, self.lookback_days - 1)
        min_stride = min(strides)
        return max(1, self.lookback_days - min(min_stride, self.lookback_days))

    def _load_bar_day(self, trade_date: int, progress: ProgressCallback | None = None) -> pd.DataFrame:
        if trade_date in self._bar_cache:
            frame = self._bar_cache.pop(trade_date)
            self._bar_cache[trade_date] = frame
            if progress is not None:
                progress(f"date={trade_date} cache=hit rows={len(frame)} stocks={frame['ts_code'].nunique()}")
            return frame
        path = daily_path(self.root, trade_date)
        columns = ["trade_date", "ts_code", "bar_index", self.volume_column]
        if not path.exists():
            if progress is not None:
                progress(f"date={trade_date} cache=miss step=missing path={path}")
            return pd.DataFrame(columns=columns)
        if progress is not None:
            progress(f"date={trade_date} cache=miss step=read columns={','.join(columns)}")
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.dropna(subset=["trade_date", "ts_code", "bar_index", self.volume_column])
        frame["trade_date"] = frame["trade_date"].astype("int32")
        frame["bar_index"] = frame["bar_index"].astype("int32")
        frame["ts_code"] = frame["ts_code"].astype(str)
        self._bar_cache[trade_date] = frame[columns]
        self._trim_cache()
        if progress is not None:
            progress(
                f"date={trade_date} step=done rows={len(self._bar_cache[trade_date])} "
                f"stocks={self._bar_cache[trade_date]['ts_code'].nunique()}"
            )
        return self._bar_cache[trade_date]

    def _trim_cache(self) -> None:
        while len(self._bar_cache) > self.cache_days:
            self._bar_cache.popitem(last=False)

    def _empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["trade_date", "ts_code", *self.feature_columns])


def _signature_from_volume(volume: np.ndarray, order: int) -> np.ndarray:
    log_values = np.log(np.maximum(volume, 1.0)).astype(np.float64)
    width = tensor_signature_width(order)
    level_offsets = np.asarray(_level_offsets(order), dtype=np.int64)
    levels = np.zeros(width, dtype=np.float64)
    previous = np.empty(width, dtype=np.float64)
    scaled = np.empty(order + 1, dtype=np.float64)
    try:
        _signature_from_log_volume_numba(log_values, int(order), levels, previous, level_offsets, scaled)
        return levels
    except NameError as exc:  # pragma: no cover
        raise ImportError("LogsigSignatureProvider requires installing numba") from exc


def _signature_batch_from_volume(
    volume: np.ndarray,
    order: int,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, str]:
    try:
        rust = importlib.import_module("yq_factor_engine_py")
        rust_fn = getattr(rust, "logsig_signature_batch")
        if progress is not None:
            progress(
                f"step=signature_compute backend=rust_logsig rows={volume.shape[0]} "
                f"width={signature_width(order)} threads={_rust_logsig_threads(rust)}"
            )
        result = rust_fn(volume, int(order))
        output = np.asarray(result, dtype="float32")
        expected_shape = (volume.shape[0], signature_width(order))
        if output.shape != expected_shape:
            raise ValueError(f"Rust logsig signature returned shape {output.shape}, expected {expected_shape}")
        return np.ascontiguousarray(output, dtype="float32"), "rust_logsig"
    except Exception as exc:
        if progress is not None:
            reason = " ".join(str(exc).split())
            progress(
                f"step=signature_compute backend=numba_signature_fallback rows={volume.shape[0]} "
                f"width={signature_width(order)} fallback_reason={type(exc).__name__}: {reason}"
            )
        output = np.vstack([_logsignature_from_volume_fallback(row, order) for row in volume]).astype(
            "float32",
            copy=False,
        )
        return output, "numba_signature_fallback"


def _rust_logsig_threads(rust: Any) -> str:
    try:
        return str(getattr(rust, "logsig_signature_threads")())
    except Exception:
        return "unknown"


def _logsignature_from_volume_fallback(volume: np.ndarray, order: int) -> np.ndarray:
    signature = _signature_from_volume(volume, order)
    tensor_log = _tensor_log_from_signature(signature, order)
    return _project_tensor_log_to_lyndon(tensor_log, order)


@lru_cache(maxsize=None)
def _level_offsets(order: int) -> tuple[int, ...]:
    offsets = [0] * (order + 1)
    running = 0
    for level in range(1, order + 1):
        offsets[level] = running
        running += 2**level
    return tuple(offsets)


def _tensor_log_from_signature(signature: np.ndarray, order: int) -> np.ndarray:
    offsets = _level_offsets(order)
    powers = [np.zeros_like(signature) for _ in range(order + 1)]
    powers[1][:] = signature
    for power in range(2, order + 1):
        for level in range(power, order + 1):
            width = 2**level
            offset = offsets[level]
            for word in range(width):
                value = 0.0
                for prefix_len in range(1, level - power + 2):
                    suffix_len = level - prefix_len
                    prefix_word = word >> suffix_len
                    suffix_word = word & ((1 << suffix_len) - 1)
                    value += (
                        signature[offsets[prefix_len] + prefix_word]
                        * powers[power - 1][offsets[suffix_len] + suffix_word]
                    )
                powers[power][offset + word] = value

    output = np.zeros_like(signature)
    for power in range(1, order + 1):
        coefficient = (1.0 if power % 2 == 1 else -1.0) / power
        output += coefficient * powers[power]
    return output


def _project_tensor_log_to_lyndon(tensor_log: np.ndarray, order: int) -> np.ndarray:
    words, expansions = _lyndon_basis(order)
    offsets = _level_offsets(order)
    output: list[float] = []
    start = 0
    while start < len(words):
        degree = words[start][0]
        end = start
        while end < len(words) and words[end][0] == degree:
            end += 1
        residual = tensor_log[offsets[degree] : offsets[degree] + 2**degree].copy()
        for idx in range(start, end):
            word = words[idx][1]
            coefficient = float(residual[word])
            output.append(coefficient)
            for expanded_word, expanded_coeff in expansions[idx].items():
                residual[expanded_word] -= coefficient * expanded_coeff
        start = end
    return np.asarray(output, dtype=np.float64)


@lru_cache(maxsize=None)
def _lyndon_words(order: int) -> tuple[tuple[int, int], ...]:
    if order <= 0:
        raise ValueError("logsignature order must be positive")
    words = []
    for length in range(1, order + 1):
        for word in range(2**length):
            if _is_lyndon(word, length):
                words.append((length, word))
    return tuple(words)


def _is_lyndon(word: int, length: int) -> bool:
    if length == 1:
        return True
    return all(word < _rotate_left_word(word, length, shift) for shift in range(1, length))


def _rotate_left_word(word: int, length: int, shift: int) -> int:
    output = 0
    for pos in range(length):
        source_pos = (pos + shift) % length
        output = (output << 1) | ((word >> (length - 1 - source_pos)) & 1)
    return output


@lru_cache(maxsize=None)
def _lyndon_basis(order: int) -> tuple[tuple[tuple[int, int], ...], tuple[dict[int, float], ...]]:
    words = _lyndon_words(order)
    word_set = set(words)
    expansions_by_word: dict[tuple[int, int], dict[int, float]] = {}
    expansions: list[dict[int, float]] = []
    for length, word in words:
        if length == 1:
            expansion = {word: 1.0}
        else:
            prefix_len, prefix_word, suffix_len, suffix_word = _standard_factorization(
                length,
                word,
                word_set,
            )
            expansion = _bracket_expansion(
                expansions_by_word[(prefix_len, prefix_word)],
                prefix_len,
                expansions_by_word[(suffix_len, suffix_word)],
                suffix_len,
            )
        leading = expansion.get(word, 0.0)
        if abs(leading - 1.0) > 1e-10:
            raise ValueError(f"invalid Lyndon basis expansion for word {word}: leading coefficient {leading}")
        expansions_by_word[(length, word)] = expansion
        expansions.append(expansion)
    return words, tuple(expansions)


def _standard_factorization(
    length: int,
    word: int,
    word_set: set[tuple[int, int]],
) -> tuple[int, int, int, int]:
    for suffix_len in range(length - 1, 0, -1):
        suffix_word = word & ((1 << suffix_len) - 1)
        if (suffix_len, suffix_word) not in word_set:
            continue
        prefix_len = length - suffix_len
        prefix_word = word >> suffix_len
        if (prefix_len, prefix_word) in word_set:
            return prefix_len, prefix_word, suffix_len, suffix_word
    raise ValueError(f"could not factor Lyndon word length={length} word={word}")


def _bracket_expansion(
    left: dict[int, float],
    left_len: int,
    right: dict[int, float],
    right_len: int,
) -> dict[int, float]:
    output: dict[int, float] = {}
    for left_word, left_coeff in left.items():
        for right_word, right_coeff in right.items():
            coeff = left_coeff * right_coeff
            lr_word = (left_word << right_len) | right_word
            rl_word = (right_word << left_len) | left_word
            output[lr_word] = output.get(lr_word, 0.0) + coeff
            output[rl_word] = output.get(rl_word, 0.0) - coeff
    return {word: value for word, value in output.items() if abs(value) > 1e-14}


try:
    from numba import njit
except ImportError:  # pragma: no cover
    pass
else:

    @njit(cache=True)
    def _signature_from_log_volume_numba(values, order, levels, previous, level_offsets, scaled):
        for idx in range(1, len(values)):
            delta = values[idx] - values[idx - 1]
            if abs(delta) <= 1e-15:
                continue
            _append_axis_segment(levels, previous, level_offsets, scaled, 0, delta, order)
            _append_axis_segment(levels, previous, level_offsets, scaled, 1, delta, order)

    @njit(cache=True)
    def _append_axis_segment(levels, previous, level_offsets, scaled, axis, delta, order):
        for idx in range(levels.shape[0]):
            previous[idx] = levels[idx]
        scaled[0] = 1.0
        for level in range(1, order + 1):
            scaled[level] = scaled[level - 1] * delta / level
        for level in range(1, order + 1):
            width = 2**level
            offset = level_offsets[level]
            for word in range(width):
                suffix = _repeated_axis_suffix_len(word, level, axis)
                value = previous[offset + word]
                for repeat in range(1, suffix + 1):
                    if level - repeat == 0:
                        prefix_value = 1.0
                    else:
                        prefix_value = previous[level_offsets[level - repeat] + (word >> repeat)]
                    value += prefix_value * scaled[repeat]
                levels[offset + word] = value

    @njit(cache=True)
    def _repeated_axis_suffix_len(word, level, axis):
        count = 0
        for bit in range(level):
            if ((word >> bit) & 1) == axis:
                count += 1
            else:
                break
        return count
