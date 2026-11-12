from __future__ import annotations

from yq_ml_alpha.models.rawbar_lstm_model import RawBarLSTMAlphaModel as RawBarLSTMAlphaModel
from yq_ml_alpha.models.rawbar_training import RawBarRecurrentRegressor, _RawBarAlphaModel


class RawBarAGRUAlphaModel(_RawBarAlphaModel):
    """Single-output attention GRU over a normalized daily rawbar panel."""

    architecture = "agru"

    def _build_regressor(self, nn):
        import torch

        hidden_size = self.params["hidden_size"]

        def attention():
            return nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                                 nn.Linear(hidden_size, 1, bias=False))

        def readout(network, output, state):
            weights = torch.softmax(network.attention(output), dim=1)
            return torch.sum(weights * output, dim=1)

        return RawBarRecurrentRegressor(
            nn, self.params, nn.GRU, readout, getattr(self, "_panel_stats", None), attention=attention,
        )
