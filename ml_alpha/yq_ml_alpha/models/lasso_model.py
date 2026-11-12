from __future__ import annotations

from yq_ml_alpha.models.regularized_linear_training import _RegularizedLinearAlphaModel


class LassoAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "Lasso"
    default_search_space = {
        "alpha": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
        "fit_intercept": [True, False],
        "max_iter": [3000, 5000, 10000],
        "tol": [1e-5, 1e-4, 1e-3],
        "selection": ["cyclic", "random"],
        "positive": [False, True],
    }
