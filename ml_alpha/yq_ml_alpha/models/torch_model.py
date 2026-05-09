from __future__ import annotations

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class TorchMLPAlphaModel(AlphaModel):
    """Small optional Torch adapter intended for v1.5 raw-panel experiments."""

    def __init__(self) -> None:
        self.model = None

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        import torch
        from torch import nn

        params = dict(context.model_params)
        hidden = int(params.get("hidden", 64))
        epochs = int(params.get("epochs", 10))
        lr = float(params.get("lr", 1e-3))
        x = _features(train_data, context.feature_columns)
        y = train_data[context.label_column].astype("float32").to_numpy().reshape(-1, 1)
        self.model = nn.Sequential(
            nn.Linear(x.shape[1], hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(self.model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        import torch

        with torch.no_grad():
            score = self.model(torch.from_numpy(_features(data, context.feature_columns))).numpy().reshape(-1)
        return pd.Series(score, index=data.index, dtype="float32")


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
