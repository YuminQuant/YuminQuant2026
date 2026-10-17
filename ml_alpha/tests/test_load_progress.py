from types import SimpleNamespace

import pandas as pd
import pytest

from yq_ml_alpha.data.dataset import DatasetBuilder


@pytest.mark.parametrize("split", ["train", "valid"])
def test_common_load_logs_filtered_rows_and_empty_dates(capsys, monkeypatch, split):
    builder = DatasetBuilder.__new__(DatasetBuilder)
    data = pd.DataFrame({"trade_date": [20220104, 20220104],
                         "ts_code": ["a", "b"], "x": [1., 2.]})
    builder.config = SimpleNamespace(label=SimpleNamespace(id="y", root="unused"))
    builder.feature_provider = SimpleNamespace(
        feature_columns=["x"], load=lambda date: data.copy() if date == 20220104 else data.iloc[:0].copy())
    builder.universe = SimpleNamespace(filter=lambda frame, date: frame)
    builder.filters = SimpleNamespace(apply=lambda frame, date: frame)
    builder._preprocess = lambda frame, include_label: frame
    builder._maybe_cache = lambda *args: None
    labels = data[["trade_date", "ts_code"]].assign(y=[1., float("nan")])
    monkeypatch.setattr("yq_ml_alpha.data.dataset.read_daily", lambda *args: labels)
    bundle = builder.load([20220104, 20220105], True, split_name=split)
    assert len(bundle.frame) == 1
    log = capsys.readouterr().out
    assert f"split={split} [1/2] date=20220104 step=start" in log
    assert "step=done rows=1 total_rows=1" in log
    assert "[2/2] date=20220105 step=done rows=0 total_rows=1" in log
    assert "date_elapsed_s=" in log
    assert "step=complete dates=2 total_rows=1 elapsed_s=" in log
    builder.load([], True, split_name=split)
    assert "step=complete dates=0 total_rows=0" in capsys.readouterr().out
