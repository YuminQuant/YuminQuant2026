from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class QDAAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "qda"
    classification = True

    def _make_estimator(self, params):
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        defaults = dict(reg_param=0.1)
        return QuadraticDiscriminantAnalysis(**{**defaults, **params})
