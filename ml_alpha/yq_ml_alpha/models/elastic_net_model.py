from __future__ import annotations

from yq_ml_alpha.models.regularized_linear_training import _RegularizedLinearAlphaModel


class ElasticNetAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "ElasticNet"
    default_search_space = {
        "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "l1_ratio": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
        "fit_intercept": [True, False],
        "max_iter": [3000, 5000, 10000],
        "tol": [1e-5, 1e-4, 1e-3],
        "selection": ["cyclic", "random"],
        "positive": [False, True],
    }
