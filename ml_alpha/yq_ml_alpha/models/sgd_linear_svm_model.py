from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class SGDLinearSVMAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "sgd_hinge"
    classification = True
    score_with_margin = True

    def _make_estimator(self, params):
        from sklearn.linear_model import SGDClassifier

        defaults = dict(loss='hinge', alpha=0.1, penalty='l2', max_iter=10000, tol=0.001, random_state=42)
        if params.get("loss", defaults["loss"]) != defaults["loss"]:
            raise ValueError("select the matching SGD model class instead of changing its loss")
        return SGDClassifier(**{**defaults, **params})
