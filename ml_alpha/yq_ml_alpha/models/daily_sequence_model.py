"""Shared lazy daily sequence validation and label-free prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.torch_support import TorchAlphaModel, _device, _torch_modules


def full_date_ic_loss(prediction, target, eps):
    if prediction.ndim != 1 or target.shape != prediction.shape or prediction.numel() < 2:
        raise ValueError("full-date IC requires equally sized one-dimensional vectors with at least two stocks")
    prediction, target = prediction - prediction.mean(), target - target.mean()
    denominator = prediction.norm() * target.norm()
    return 1 - (prediction * target).sum() / denominator.clamp_min(eps)


class DailySequenceAlphaModel(TorchAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError(f"{type(self).__name__} requires the lazy daily sequence bundle interface")

    def predict(self, data, context):
        raise ValueError(f"{type(self).__name__} requires the lazy daily sequence bundle interface")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or source.length != self.params["sequence_length"]
                or source.columns != self.feature_columns or bundle.feature_columns != self.feature_columns):
            raise ValueError(f"{type(self).__name__} requires matching daily sequence histories")
        return source

    def _inputs(self, bundle, date, frame):
        return self._source(bundle).get(int(date), frame.ts_code.tolist())

    def predict_bundle(self, bundle, context):
        return self._predict_bundle(bundle)

    def _predict_bundle(self, bundle, *, with_components=False):
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} has not been fitted")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        output = np.empty(len(bundle.frame), dtype="float32")
        components = np.empty((len(bundle.frame), self.params["base_factors"]), dtype=np.float32) if with_components else None
        try:
            with torch.no_grad():
                for date, indices in bundle.frame.groupby("trade_date", sort=True).indices.items():
                    frame = bundle.frame.iloc[indices]
                    if with_components:
                        scores, factors = self._scores_and_components(torch, bundle, date, frame, device)
                        components[indices] = factors.cpu().numpy()
                    else:
                        scores = self._scores(torch, bundle, date, frame, device, training=False)
                    if not torch.isfinite(scores).all():
                        raise ValueError(f"nonfinite daily sequence prediction on {date}")
                    output[indices] = scores.cpu().numpy()
            scores = pd.Series(output, index=bundle.frame.index, dtype="float32")
            return (scores, components) if with_components else scores
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")
