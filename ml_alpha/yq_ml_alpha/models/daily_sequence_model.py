"""Shared lazy daily sequence validation and label-free prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _device, _torch_modules


class DailySequenceAlphaModel(MLPAlphaModel):
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
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} has not been fitted")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        output = np.empty(len(bundle.frame), dtype="float32")
        try:
            with torch.no_grad():
                for date, indices in bundle.frame.groupby("trade_date", sort=True).indices.items():
                    scores = self._scores(torch, bundle, date, bundle.frame.iloc[indices], device, training=False)
                    if not torch.isfinite(scores).all():
                        raise ValueError(f"nonfinite daily sequence prediction on {date}")
                    output[indices] = scores.cpu().numpy()
            return pd.Series(output, index=bundle.frame.index, dtype="float32")
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")
