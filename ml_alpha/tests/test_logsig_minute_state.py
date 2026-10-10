from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from yq_ml_alpha.features.logsig_minute_state import MinuteVolumeWindow, read_minute_volume_day
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider


def write_minutes(root, date, reverse=False):
    path = root / str(date // 10000) / f"{date}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(code, time, value) for code in ["000002.SZ", "000001.SZ"]
            for time, value in [("09:30:00", 10000.), ("09:31:00", 2.), ("09:32:00", 3.),
                                ("11:30:00", 5.), ("13:01:00", 7.), ("15:00:00", 11.)]]
    frame = pd.DataFrame(rows, columns=["ts_code", "trade_time", "vol"])
    if reverse:
        frame = frame.iloc[::-1]
    frame.to_parquet(path, index=False)
    return path


def test_minute_projection_session_boundaries_and_missing_bars(tmp_path):
    path = write_minutes(tmp_path, 20260105, reverse=True)
    import pyarrow.parquet as pq
    original = pq.read_table
    with mock.patch("yq_ml_alpha.features.logsig_minute_state.pq.read_table", wraps=original) as read:
        symbols, values = read_minute_volume_day(path, 120)
    assert read.call_args.kwargs["columns"] == ["ts_code", "trade_time", "vol"]
    assert symbols.tolist() == ["000001.SZ", "000002.SZ"]
    np.testing.assert_array_equal(values, [[10., 18.], [10., 18.]])
    # A missing 5min interval is not replaced by older data or zero.
    assert read_minute_volume_day(path, 5)[0].size == 0


def test_raw_minute_requires_calendar(tmp_path):
    provider = LogsigSignatureProvider(tmp_path, "__all__", dict(source="minute"))
    with pytest.raises(ValueError, match="trading calendar"):
        provider.load(20260105)


def test_window_reuses_days_aligns_symbols_and_reloads_revisions(tmp_path):
    dates = [20260105, 20260106, 20260107]
    paths = [write_minutes(tmp_path, d, reverse=i % 2 == 0) for i, d in enumerate(dates)]
    provider = LogsigSignatureProvider(tmp_path, "__all__", dict(source="minute", lookback_days=2, bar_size=120, order=2))
    provider.set_calendar_dates(dates)
    with mock.patch("yq_ml_alpha.features.logsig_minute_state.read_minute_volume_day", wraps=read_minute_volume_day) as read:
        first = provider.load(dates[1])
        assert read.call_count == 2
        second = provider.load(dates[2])
        assert read.call_count == 3
        assert set(provider._minute_state.days) == set(dates[1:])
        np.testing.assert_allclose(first.iloc[:, 2:], second.iloc[:, 2:])
        frame = pd.read_parquet(paths[1])
        frame.loc[frame.trade_time == "13:01:00", "vol"] *= 3
        frame.to_parquet(paths[1], index=False)
        changed = provider.load(dates[2])
        assert read.call_count == 4
        assert not np.allclose(second.iloc[:, 2:], changed.iloc[:, 2:])


def test_missing_date_and_backward_replay(tmp_path):
    dates = [20260105, 20260106, 20260107, 20260108]
    for date in dates:
        if date != 20260106:
            write_minutes(tmp_path, date)
    provider = LogsigSignatureProvider(tmp_path, "__all__", dict(source="minute", lookback_days=2, bar_size=120, order=2))
    provider.set_calendar_dates(dates)
    assert provider.load(dates[1]).empty
    assert provider.load(dates[2]).empty
    assert len(provider.load(dates[3])) == 2
    assert provider.load(dates[1]).empty
    assert len(provider._minute_state.days) == 2


def test_duplicates_nulls_and_partial_minute_groups_match_bar_rules(tmp_path):
    path = tmp_path / "minutes.parquet"
    pd.DataFrame({
        "ts_code": ["A", "A", "A", "A", None, "B"],
        "trade_time": ["09:31:00", "09:31:00", "13:01:00", "13:02:00", "09:31:00", "09:31:00"],
        "vol": [2., 3., 4., np.nan, 999., 1.],
    }).to_parquet(path, index=False)
    symbols, values = read_minute_volume_day(path, 120)
    assert symbols.tolist() == ["A"]
    np.testing.assert_array_equal(values, [[5., 4.]])


def test_minute_features_read_only_never_reads_raw(tmp_path):
    root = tmp_path / "minutes"
    write_minutes(root, 20260105)
    params = dict(source="minute", lookback_days=1, bar_size=120, order=2, feature_cache_dir=tmp_path / "features")
    producer = LogsigSignatureProvider(root, "__all__", params)
    reader = LogsigSignatureProvider(root, "__all__", dict(params, read_only=True))
    for provider in (producer, reader):
        provider.set_calendar_dates([20260105])
    expected = producer.load(20260105)
    with mock.patch.object(MinuteVolumeWindow, "matrix", side_effect=AssertionError("read-only")):
        pd.testing.assert_frame_equal(reader.load(20260105), expected)
