from __future__ import annotations

from yq_ml_alpha.models.classical_training import _ClassicalAlphaModel


class SGDModifiedHuberAlphaModel(_ClassicalAlphaModel):
    estimator_kind = "sgd_modified_huber"
    classification = True

    def _make_estimator(self, params):
        from sklearn.linear_model import SGDClassifier

        defaults = dict(
            loss='modified_huber',
            alpha=1.0,
            penalty='l2',
            max_iter=10000,
            tol=0.001,
            random_state=42,
        )
        if params.get("loss", defaults["loss"]) != defaults["loss"]:
            raise ValueError("select the matching SGD model class instead of changing its loss")
        return SGDClassifier(**{**defaults, **params})
