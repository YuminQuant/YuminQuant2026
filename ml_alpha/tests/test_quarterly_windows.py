from pathlib import Path
from dataclasses import replace

import pytest

import pandas as pd

from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.config import load_config
from yq_ml_alpha.data.sampler import sample_dates
from yq_ml_alpha.pipelines.runtime import build_windows, _actual_refit_dates, _build_windows


def test_quarterly_windows_cover_first_and_last_partial_quarters_without_leakage():
    config = load_config(Path(__file__).parents[1] / "models/mdl_000008.toml")
    calendar = TradingCalendar([int(d.strftime("%Y%m%d")) for d in
                                pd.bdate_range("2016-01-01", "2026-12-31")])
    config = replace(config, train_scheme=replace(config.train_scheme, start_from_features=False))
    positions = {date: i for i, date in enumerate(calendar.dates)}
    windows = build_windows(config, calendar)
    raw = _build_windows(config, calendar)
    assert len(windows) == 30
    assert windows[0].training_cutoff_date == 20181231
    assert windows[0].train_dates[0] == 20160101
    assert windows[0].activation_date == calendar.offset(20181231, 7)
    assert [d for w in windows for d in w.predict_dates] == calendar.between(windows[0].activation_date, config.dates.predict[1])
    for original, window in zip(raw, windows):
        assert len(original.valid_dates) == int(len(original.train_dates + original.valid_dates) * 0.2 + 0.5)
        assert window.activation_date == calendar.offset(window.training_cutoff_date, 7)
        assert window.valid_dates == original.valid_dates  # No prediction-boundary tail purge.
        assert positions[window.train_dates[-1]] + 6 < positions[window.valid_dates[0]]
        assert max(positions[d] + 6 for d in window.train_dates + window.valid_dates) < positions[window.predict_dates[0]]
        for dates in [window.train_dates, window.valid_dates]:
            assert all(positions[b] - positions[a] == 5 for a, b in zip(dates, dates[1:]))
    for old, new in zip(windows, windows[1:]):
        assert old.predict_dates[-1] == calendar.previous_open(new.activation_date)
        assert new.training_cutoff_date in old.predict_dates
        assert calendar.offset(new.training_cutoff_date, 6) in old.predict_dates
    # The first five prediction sessions after the next cutoff still use the old model.
    for step in range(1, 6):
        assert calendar.offset(windows[1].training_cutoff_date, step) in windows[0].predict_dates


def test_quarter_sampler_and_no_false_partial_quarter_anchor():
    calendar = TradingCalendar([20230330, 20230331, 20230403, 20230629, 20230630, 20230703, 20230704])
    assert sample_dates(calendar, (20230330, 20230704), "quarterly_end") == [20230331, 20230630, 20230704]
    assert _actual_refit_dates(calendar, (20230403, 20230704), "quarterly_end") == [20230331, 20230630]


@pytest.mark.parametrize("horizon", [0, 1, 6, 21])
def test_shared_policy_respects_custom_horizons_and_prediction_subranges(horizon):
    config = load_config(Path(__file__).parents[1] / "models/mdl_000008.toml")
    config = replace(config, train_scheme=replace(config.train_scheme, start_from_features=False),
                     label=replace(config.label, lookahead_days=horizon))
    calendar = TradingCalendar([int(d.strftime("%Y%m%d")) for d in
                                pd.bdate_range("2016-01-01", "2026-12-31")])
    windows = build_windows(config, calendar)
    assert windows[0].activation_date == calendar.offset(windows[0].training_cutoff_date, horizon + 1)
    start = windows[1].training_cutoff_date
    subset = replace(config, dates=replace(config.dates, predict=(start, 20200701)))
    clipped = build_windows(subset, calendar)
    assert clipped[0].window_id == windows[0].window_id
    assert clipped[0].predict_dates[0] == start


def test_feature_start_skips_empty_and_all_missing_files(tmp_path):
    from yq_ml_alpha.pipelines.runtime import _feature_start_month
    config = load_config(Path(__file__).parents[1] / "models/mdl_000008.toml")
    root = tmp_path / "factors"
    (root / "2016").mkdir(parents=True)
    for date, values in [(20160104, []), (20160105, [float("nan")]), (20160201, [1.0])]:
        pd.DataFrame({"trade_date": [date] * len(values), "ts_code": ["a"] * len(values),
                      "x": values}).to_parquet(root / "2016" / (str(date) + ".parquet"))
    config = replace(config, features=replace(config.features, root=root))
    calendar = TradingCalendar([20160104, 20160105, 20160201])
    assert _feature_start_month(config, calendar) == 20160201


def test_invalid_activation_combinations_are_rejected():
    from yq_ml_alpha.config import _validate_train_scheme
    config = load_config(Path(__file__).parents[1] / "models/mdl_000008.toml")
    for changes in [{"type": "static"}, {"validation_ratio": None},
                    {"refit_frequency": "daily"}, {"prediction_activation": "unknown"}]:
        with pytest.raises(ValueError):
            _validate_train_scheme(replace(config.train_scheme, **changes))


def test_manifest_records_cutoff_and_activation():
    from types import SimpleNamespace
    from yq_ml_alpha.output.artifacts import artifact_identity
    config = load_config(Path(__file__).parents[1] / "models/mdl_000008.toml")
    config = replace(config, train_scheme=replace(config.train_scheme, start_from_features=False))
    calendar = TradingCalendar([int(d.strftime("%Y%m%d")) for d in
                                pd.bdate_range("2016-01-01", "2026-12-31")])
    window = build_windows(config, calendar)[0]
    dataset = SimpleNamespace(feature_provider=SimpleNamespace(feature_columns=["x"]))
    identity = artifact_identity(config, window, dataset)
    assert identity["prediction_activation"]["training_cutoff_date"] == window.training_cutoff_date
    assert identity["prediction_activation"]["activation_date"] == window.activation_date
