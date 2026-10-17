from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from yq_ml_alpha.models.base import ModelContext

optuna = pytest.importorskip("optuna")


@pytest.mark.parametrize("backend", ["lgbm", "xgb"])
@pytest.mark.parametrize("values,expected", [
    ([1.0] * 20, 6),
    ([1.0, 0.99996, 0.99992, 0.99988] + [0.99988] * 16, 9),
    ([1.0 - i * 0.01 for i in range(20)], 20),
    ([0.0] * 20, 6),
])
def test_real_study_stops_and_keeps_absolute_best(backend, values, expected):
    module = import_module("yq_ml_alpha.models." + backend + "_optuna_model")
    callback = module._SearchEarlyStopping(5, 0.0001)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: values[trial.number], n_trials=20, callbacks=[callback])
    assert len(study.trials) == expected
    assert study.best_value == min(values[:expected])


@pytest.mark.parametrize("backend,name", [
    ("lgbm", "LightGBMOptunaAlphaModel"), ("xgb", "XGBoostOptunaAlphaModel"),
])
def test_model_stops_search_but_still_refits(monkeypatch, backend, name):
    module = import_module("yq_ml_alpha.models." + backend + "_optuna_model")
    fits = []

    class Regressor:
        def __init__(self, **params):
            pass

        def fit(self, x, y, **kwargs):
            fits.append(len(y))

        def predict(self, x):
            return np.zeros(len(x))

    monkeypatch.setattr(module, "_lightgbm_module" if backend == "lgbm" else "_xgboost_module",
                        lambda: SimpleNamespace(LGBMRegressor=Regressor, XGBRegressor=Regressor))
    model = getattr(module, name)()
    context = ModelContext("test", "test", ["x"], "y", Path("unused"), {},
                           {"n_trials": 20, "refit_on_validation": True})
    model.fit(pd.DataFrame({"x": [1., 2.], "y": [1., 2.]}),
              pd.DataFrame({"x": [3.], "y": [3.]}), context)
    assert fits == [2] * 6 + [3]
    assert model.study_summary_["stopped_early"]
    assert model.study_summary_["n_trials"] == 6
    assert model.best_value_ == 9.0
