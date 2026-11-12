from __future__ import annotations

import inspect

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class AdaBoostAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "adaboost"

    def _make_estimator(self, params):
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.tree import DecisionTreeRegressor

        defaults = dict(n_estimators=100, learning_rate=0.05, loss="linear", random_state=42)
        options = {**defaults, **params}
        base = DecisionTreeRegressor(
            max_depth=int(options.pop("base_max_depth", 3)),
            min_samples_leaf=int(options.pop("base_min_samples_leaf", 10)),
            random_state=options["random_state"],
        )
        # Support both scikit-learn's old and new constructor parameter names.
        if "estimator" in params or "base_estimator" in params:
            raise ValueError("configure AdaBoost trees with base_max_depth/base_min_samples_leaf")
        key = "estimator" if "estimator" in inspect.signature(AdaBoostRegressor).parameters else "base_estimator"
        options[key] = base
        return AdaBoostRegressor(**options)
