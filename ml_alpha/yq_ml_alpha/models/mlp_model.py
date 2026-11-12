from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.tabular_training import TabularTorchRegressor


class MLPAlphaModel(TabularTorchRegressor):
    """PyTorch MLP regressor for factor-frame alpha combination."""

    def _build_model(self, nn):
        return _build_network(nn, self.input_dim, self.params)


def _build_network(nn, input_dim: int, params: dict[str, Any]):
    layers = []
    current_dim = input_dim
    for hidden_dim in params["hidden_layers"]:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.Tanh() if params["activation"] == "tanh" else nn.ReLU())
        dropout = float(params["dropout"])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)
