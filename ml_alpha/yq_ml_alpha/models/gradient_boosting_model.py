from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class GradientBoostingAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "gbdt"

    def _make_estimator(self, params):
        from sklearn.ensemble import GradientBoostingRegressor

        defaults = dict(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )
        return GradientBoostingRegressor(**{**defaults, **params})
