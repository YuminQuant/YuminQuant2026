from __future__ import annotations

from yq_ml_alpha.models.regularized_linear_training import _RegularizedLinearAlphaModel


class RidgeAlphaModel(_RegularizedLinearAlphaModel):
    estimator_name = "Ridge"
    default_search_space = {
        "alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0],
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "saga"],
        "tol": [1e-5, 1e-4, 1e-3],
        "max_iter": [None, 3000, 10000],
    }
