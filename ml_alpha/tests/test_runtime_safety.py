from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config, DatesConfig, SampleConfig
from yq_ml_alpha.pipelines.runtime import build_windows
from yq_ml_alpha.output.artifacts import save_manifest, validate_manifest
from yq_ml_alpha.output.factor_metadata import write_factor_metadata, metadata_lock, METADATA_COLUMNS
from yq_ml_alpha.output.daily_wide_writer import DailyWideWriter
from yq_ml_alpha.data.tensor_storage import TensorSpool
from yq_ml_alpha.data.stores import discover_value_columns
from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider


def config():
    return load_config(Path(__file__).parents[1] / "factors/logsig_alpha_v.toml")


def test_static_label_maturity_and_separation():
    cfg = replace(config(), dates=DatesConfig((1, 30), (31, 50), (51, 80)), sample=SampleConfig("daily", "daily"))
    w, = build_windows(cfg, TradingCalendar(list(range(1, 81))))
    assert w.train_dates[-1] == 24
    assert w.valid_dates[-1] == 44
    assert w.predict_dates[0] == 51
    assert max(w.train_dates) + 6 < min(w.valid_dates)
    assert max(w.valid_dates) + 6 < min(w.predict_dates)


def test_all_e2e_dates_and_no_refits():
    calendar = TradingCalendar([int(date.strftime("%Y%m%d")) for date in pd.bdate_range("2011-01-01", "2026-04-24")])
    for path in (Path(__file__).parents[1] / "factors").glob("*.toml"):
        cfg = load_config(path)
        assert "deprecated" not in cfg.tags
        assert cfg.dates.train == (20110101, 20150930)
        assert cfg.dates.valid == (20151001, 20151231)
        w, = build_windows(cfg, calendar)
        assert w.window_id == "static"
        assert calendar.offset(w.valid_dates[-1], 6) < w.predict_dates[0]


def test_custom_label_requires_explicit_horizon():
    cfg = config()
    cfg = replace(cfg, label=replace(cfg.label, id="custom"))
    with pytest.raises(ValueError, match="lookahead_days"):
        build_windows(cfg, TradingCalendar([]))


def test_manifest_rejects_changes_but_allows_prediction_extension(tmp_path):
    cfg = config()
    window = SimpleNamespace(train_dates=[1, 2], valid_dates=[10])
    dataset = SimpleNamespace(feature_provider=SimpleNamespace(feature_columns=["x", "y"]))
    path = tmp_path / "model.pkl"
    path.write_bytes(b"test model")
    with pytest.raises(ValueError, match="manifest"):
        validate_manifest(path, cfg, window, dataset)
    save_manifest(path, cfg, window, dataset)
    validate_manifest(path, replace(cfg, dates=replace(cfg.dates, predict=(20160101, 20300101))), window, dataset)
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(path, replace(cfg, data_version="changed"), window, dataset)
    dataset.feature_provider.feature_columns.reverse()
    with pytest.raises(ValueError, match="changed"):
        validate_manifest(path, cfg, window, dataset)


def test_tensor_spool_matches_memory_and_closes_files(tmp_path):
    spool = TensorSpool(tmp_path)
    first = np.arange(24, dtype="float32").reshape(2, 3, 4)
    second = first + 100
    spool.append({"bar": first})
    spool.append({"bar": second})
    arrays = spool.finish()
    np.testing.assert_array_equal(arrays["bar"], np.concatenate([first, second]))
    directory = Path(spool.directory.name)
    assert isinstance(arrays["bar"], np.memmap)
    spool.close()
    assert not directory.exists()


def test_metadata_sources_and_collision(tmp_path):
    cfg = replace(config(), output=replace(config().output, root=tmp_path))
    native = {name: "" for name in METADATA_COLUMNS}
    native.update(factor_id="native", tags_json="[]", asset_class="stock", frequency="daily")
    pd.DataFrame([native]).to_parquet(tmp_path / "factor_metadata.parquet")
    write_factor_metadata(cfg)
    assert set(pd.read_parquet(tmp_path / "factor_metadata.parquet").factor_id) == {"native", cfg.factor_id}
    assert list(pd.read_parquet(tmp_path / "factor_metadata.ml_alpha.parquet").factor_id) == [cfg.factor_id]
    with metadata_lock(tmp_path):
        with pytest.raises(RuntimeError, match="lock"):
            write_factor_metadata(cfg)
    native["factor_id"] = cfg.factor_id
    pd.DataFrame([native]).to_parquet(tmp_path / "factor_metadata.rust.parquet")
    with pytest.raises(ValueError, match="collide"):
        write_factor_metadata(cfg)
    assert not (tmp_path / "factor_metadata.lock").exists()


def test_writer_reads_schema_once_per_date(tmp_path):
    writer = DailyWideWriter(tmp_path, "alpha")
    path = writer._path(20260105)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"trade_date": [20260105], "ts_code": ["000001.SZ"], "existing": [1.0]}).to_parquet(path)
    with mock.patch("yq_ml_alpha.output.daily_wide_writer.parquet_columns", return_value=["existing"]) as read:
        assert "existing" in writer._schema_columns([20260105])
        writer._schema_columns([20260105])
        assert read.call_count == 1
    frame = pd.read_parquet(path)
    frame["added_later"] = 3.0
    frame.to_parquet(path, index=False)
    writer.write(pd.DataFrame({"trade_date": [20260105], "ts_code": ["000001.SZ"], "score": [2.0]}))
    assert pd.read_parquet(path)["added_later"].iloc[0] == 3.0


def test_feature_schema_cutoff_and_exclusions(tmp_path):
    for date, column in [(20150101, "old"), (20160101, "future")]:
        path = tmp_path / str(date // 10000) / f"{date}.parquet"
        path.parent.mkdir(exist_ok=True)
        pd.DataFrame({"trade_date": [date], "ts_code": ["x"], column: [1], "ml": [2]}).to_parquet(path)
    assert discover_value_columns(tmp_path, 20150930, {"ml"}) == ["old"]


def test_logsig_cache_invalidates_source_change(tmp_path):
    root = tmp_path / "bars"
    path = root / "2026" / "20260105.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"trade_date": [20260105, 20260105], "ts_code": ["000001.SZ"] * 2,
                          "bar_index": [0, 1], "volume": [10.0, 20.0]})
    frame.to_parquet(path)
    provider = LogsigSignatureProvider(root, "__all__", {"lookback_days": 1, "bar_size": 120, "order": 2, "feature_cache_dir": tmp_path / "cache"})
    provider.set_calendar_dates([20260105])
    with mock.patch("yq_ml_alpha.features.logsig_signature._signature_batch_from_volume", return_value=(np.ones((1, 3)), "test")) as compute:
        a = provider.load(20260105)
        b = provider.load(20260105)
        pd.testing.assert_frame_equal(a, b)
        assert compute.call_count == 1
        frame["volume"] *= 2
        frame.to_parquet(path)
        provider.load(20260105)
        assert compute.call_count == 2


def test_logsig_prediction_independent_of_batches_and_other_dates():
    torch = pytest.importorskip("torch")
    from yq_ml_alpha.models.logsig_orthogonal_mlp_model import LogsigOrthogonalMLPAlphaModel
    torch.manual_seed(42)
    model = LogsigOrthogonalMLPAlphaModel()
    model.model = torch.nn.Linear(2, 3)
    model.feature_mean = np.zeros(2, dtype="float32")
    model.feature_std = np.ones(2, dtype="float32")
    model.params = {"device": "cpu", "batch_size": 2, "neutralize": "none"}
    data = pd.DataFrame({"trade_date": [1] * 5 + [2] * 6, "x": np.arange(11), "y": np.arange(11) ** 2})
    ctx = SimpleNamespace(feature_columns=["x", "y"])
    a = model.predict(data, ctx)
    model.params["batch_size"] = 100
    b = model.predict(data, ctx)
    c = model.predict(data.iloc[:5], ctx)
    np.testing.assert_allclose(a, b, atol=1e-5)
    np.testing.assert_allclose(a.iloc[:5], c, atol=1e-5)
