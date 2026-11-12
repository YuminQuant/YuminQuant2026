from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class SVRAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "svr"

    def _make_estimator(self, params):
        from sklearn.svm import SVR

        defaults = dict(kernel='rbf', C=1.0, gamma=0.01, epsilon=0.001, cache_size=512)
        return SVR(**{**defaults, **params})
