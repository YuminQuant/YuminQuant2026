from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class GaussianNBAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "gaussian_nb"
    classification = True

    def _make_estimator(self, params):
        from sklearn.naive_bayes import GaussianNB

        defaults = dict(var_smoothing=1e-09)
        return GaussianNB(**{**defaults, **params})
