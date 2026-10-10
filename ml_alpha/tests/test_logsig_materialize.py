from dataclasses import replace
from pathlib import Path
from unittest import mock
import weakref

import numpy as np
import pandas as pd
import pytest

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config, DatesConfig, SampleConfig
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider
from yq_ml_alpha.pipelines import materialize


def test_read_only_requires_fresh_features_and_never_computes(tmp_path):
    root = tmp_path / "bars"
    path = root / "2026" / "20260105.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"trade_date": [20260105] * 2, "ts_code": ["000001.SZ"] * 2, "bar_index": [0, 1], "volume": [10., 20.]})
    frame.to_parquet(path)
    params = dict(lookback_days=1, bar_size=120, order=2, feature_cache_dir=tmp_path / "features")
    writer = LogsigSignatureProvider(root, "__all__", params)
    reader = LogsigSignatureProvider(root, "__all__", dict(params, read_only=True))
    for provider in (writer, reader):
        provider.set_calendar_dates([20260105])
    with mock.patch("yq_ml_alpha.features.logsig_signature._signature_batch_from_volume", return_value=(np.ones((1, 3)), "test")) as compute:
        with pytest.raises(FileNotFoundError, match="factor-materialize"):
            reader.load(20260105)
        compute.assert_not_called()
        expected = writer.load(20260105)
        pd.testing.assert_frame_equal(reader.load(20260105), expected)
        assert compute.call_count == 1
        frame["volume"] = [100., 200.]
        frame.to_parquet(path)
        with pytest.raises(ValueError, match="Stale"):
            reader.load(20260105)
        assert compute.call_count == 1


def test_empty_date_is_persisted_and_readable(tmp_path):
    params = dict(lookback_days=2, order=2, feature_cache_dir=tmp_path / "features")
    writer = LogsigSignatureProvider(tmp_path / "bars", "__all__", params)
    reader = LogsigSignatureProvider(tmp_path / "bars", "__all__", dict(params, read_only=True))
    for provider in (writer, reader):
        provider.set_calendar_dates([20260105])
    assert writer.load(20260105).empty
    assert reader.load(20260105).empty


def test_materialize_releases_each_date_and_does_not_load_labels(tmp_path):
    config = load_config(Path(__file__).parents[1] / "factors/logsig_alpha_v.toml")
    config = replace(config, features=replace(config.features, type="logsig_signature"))
    config = replace(config, dates=DatesConfig((1, 2), (3, 3), (4, 5)), sample=SampleConfig("daily", "daily"))
    config.features.params["feature_cache_dir"] = str(tmp_path / "features")
    previous = []
    visited = []

    def load(provider, date):
        assert not provider.read_only
        assert not previous or previous[-1]() is None
        frame = pd.DataFrame({"trade_date": [date]})
        previous.append(weakref.ref(frame))
        visited.append(date)
        return frame

    with mock.patch.object(TradingCalendar, "load", return_value=TradingCalendar([1, 2, 3, 4, 5])), \
         mock.patch.object(LogsigSignatureProvider, "load", load), \
         mock.patch.object(materialize, "DatasetBuilder", side_effect=AssertionError("must not load samples")):
        paths = materialize.run_config(config)
    assert visited == [1, 2, 3, 4, 5]
    assert len(paths) == 5
    assert all(ref() is None for ref in previous)
