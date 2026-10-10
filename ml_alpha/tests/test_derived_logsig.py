from pathlib import Path
import os
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.dataset import make_feature_provider
from yq_ml_alpha.features.derived_logsig import DerivedLogsigProvider
from yq_ml_alpha.pipelines.materialize import run_config


def test_derived_logsig_reads_fixed_columns_and_rejects_missing_data(tmp_path):
    reader = DerivedLogsigProvider(tmp_path)
    assert len(reader.feature_columns) == 226
    with pytest.raises(FileNotFoundError, match="derive-logsig"):
        reader.load(20260105)
    path = tmp_path / "2026" / "20260105.parquet"
    path.parent.mkdir()
    frame = pd.DataFrame({"trade_date": [20260105], "ts_code": ["000001.SZ"],
                          **{c: [float(i)] for i, c in enumerate(reader.feature_columns)}})
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(reader.load(20260105), frame)
    frame.drop(columns="logsig_0226").to_parquet(path, index=False)
    with pytest.raises(Exception):
        reader.load(20260105)


def test_model_config_uses_derived_data_not_python_materialization():
    config = load_config(Path(__file__).resolve().parents[1] / "factors/logsig_alpha_v.toml")
    assert isinstance(make_feature_provider(config), DerivedLogsigProvider)
    with pytest.raises(ValueError, match="Rust derive-logsig"):
        run_config(config)


@pytest.mark.skipif(not os.environ.get("YQ_TEST_DERIVE_LOGSIG_EXE"), reason="opt-in Rust CLI integration")
def test_rust_derive_logsig_end_to_end(tmp_path):
    from yq_ml_alpha.features.logsig_signature import LogsigSignatureProvider

    dates = [int(d.strftime("%Y%m%d")) for d in pd.bdate_range("2025-12-08", periods=22)]
    root = tmp_path / "data"
    calendar = root / "calendar/trade_cal_SSE.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"cal_date": np.array(dates, dtype="int32"), "is_open": [1] * len(dates)}).to_parquet(calendar, index=False)
    raw = root / "stock_data/minute"
    for day, date in enumerate(dates):
        rows = []
        for stock in ["000002.SZ", "000001.SZ"]:
            for slot in range(48):
                minute = slot * 5
                clock = 571 + minute if minute < 120 else 781 + minute - 120
                rows.append((stock, f"{clock//60:02}:{clock%60:02}:00", 20. + (day + slot) % 13))
        path = raw / str(date // 10000) / f"{date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["ts_code", "trade_time", "vol"]).to_parquet(path, index=False)
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nbase_data_dir = "{root.as_posix()}"\n')
    command = [os.environ["YQ_TEST_DERIVE_LOGSIG_EXE"], "derive-logsig", "--asset", "stock",
               "--start-date", str(dates[-3]), "--end-date", str(dates[-1]), "--threads", "2",
               "--project-config", str(config)]
    subprocess.run(command, check=True, capture_output=True)
    derived = root / "derived/stock/logsig_v"
    assert len(list(derived.glob("*/*.parquet"))) == 3  # No warmup outputs.
    reader = DerivedLogsigProvider(derived)
    reference = LogsigSignatureProvider(raw, "__all__", dict(source="minute", lookback_days=20, bar_size=5, order=10))
    reference.set_calendar_dates(dates)
    for date in dates[-3:]:
        pd.testing.assert_frame_equal(reader.load(date), reference.load(date), check_exact=True)
        meta = pq.ParquetFile(derived / str(date // 10000) / f"{date}.parquet").metadata
        assert meta.row_group(0).column(0).compression == "SNAPPY"
    # A restarted process reconstructs warmup; overwriting produces identical results.
    before = reader.load(dates[-1])
    subprocess.run(command, check=True, capture_output=True)
    pd.testing.assert_frame_equal(reader.load(dates[-1]), before, check_exact=True)
    source = raw / str(dates[-1] // 10000) / f"{dates[-1]}.parquet"
    source.unlink()
    subprocess.run(command + ["--overwrite", "false"], check=True, capture_output=True)
    pd.testing.assert_frame_equal(reader.load(dates[-1]), before, check_exact=True)
    subprocess.run(command, check=True, capture_output=True)
    assert reader.load(dates[-1]).empty
