from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class LogisticRegressionAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "logistic"
    classification = True

    def _make_estimator(self, params):
        from sklearn.linear_model import LogisticRegression

        defaults = dict(C=0.001, solver='lbfgs', max_iter=1000, random_state=42)
        return LogisticRegression(**{**defaults, **params})
