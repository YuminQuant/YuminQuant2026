from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.config import load_config
from yq_ml_alpha.calendar import TradingCalendar
from yq_ml_alpha.pipelines.runtime import build_windows


@pytest.mark.parametrize('backend,class_name', [
    ('lgbm', 'LightGBMOptunaAlphaModel'), ('xgb', 'XGBoostOptunaAlphaModel'),
])
@pytest.mark.parametrize('refit', [False, True])
def test_validation_only_enters_final_fit(monkeypatch, tmp_path, backend, class_name, refit):
    module = import_module('yq_ml_alpha.models.' + backend + '_optuna_model')
    fits = []

    class Regressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y, **kwargs):
            fits.append(y.copy())

        def predict(self, x):
            return np.zeros(len(x))

    class Study:
        best_params = {}
        best_value = 1.0
        trials = [1]

        def optimize(self, objective, **kwargs):
            objective(None)

    monkeypatch.setattr(module, '_lightgbm_module' if backend == 'lgbm' else '_xgboost_module',
                        lambda: SimpleNamespace(LGBMRegressor=Regressor, XGBRegressor=Regressor))
    monkeypatch.setattr(module, '_optuna_module', lambda: SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=lambda **kw: None), create_study=lambda **kw: Study()))
    monkeypatch.setattr(module, 'suggest_params', lambda *args: {})
    context = ModelContext('test', 'test', ['x'], 'y', tmp_path, {},
                           {'n_trials': 1, 'refit_on_validation': refit})
    model = getattr(module, class_name)()
    model.fit(pd.DataFrame({'x': [1., 2.], 'y': [10., 20.]}),
              pd.DataFrame({'x': [3.], 'y': [30.]}), context)
    assert fits[0].tolist() == [10., 20.]
    assert fits[1].tolist() == ([10., 20., 30.] if refit else [10., 20.])


@pytest.mark.parametrize('model_id', ['mdl_000007', 'mdl_000008'])
def test_fixed_gpu_windows_have_mature_labels(model_id):
    config = load_config(Path(__file__).parents[1] / 'models' / (model_id + '.toml'))
    calendar = TradingCalendar([int(d.strftime('%Y%m%d')) for d in
                                pd.bdate_range('2016-01-01', '2026-04-24')])
    window, = build_windows(config, calendar)
    assert window.window_id == 'static'
    assert window.train_dates[0] >= 20160101
    assert window.valid_dates[0] >= 20220101
    assert window.predict_dates[0] >= 20230101
    assert calendar.offset(window.train_dates[-1], 6) < window.valid_dates[0]
    assert calendar.offset(window.valid_dates[-1], 6) < window.predict_dates[0]
    assert config.model.search['refit_on_validation']
    if model_id == 'mdl_000007':
        assert config.model.params['device_type'] == 'gpu'
    else:
        assert config.model.params['tree_method'] == 'gpu_hist'
