from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class LDAAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "lda"
    classification = True

    def _make_estimator(self, params):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        defaults = dict(solver='lsqr', shrinkage='auto')
        return LinearDiscriminantAnalysis(**{**defaults, **params})
