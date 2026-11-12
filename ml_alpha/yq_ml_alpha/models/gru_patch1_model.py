from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel
from yq_ml_alpha.models.gru_patch_inputs import _gru, _params, _patches


class GRUPatch1AlphaModel(BarGRUAlphaModel):
    """Shared intraday GRU, then attention across trading-day patches."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _params(raw, "patch1")

    def _build_regressor(self, nn):
        return _GRUPatch1Regressor(nn, self.params)


class _GRUPatch1Regressor:
    def __new__(cls, nn, params: dict[str, Any]):
        import torch

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # One shared GRU, not one independently parameterized GRU per day.
                self.gru = _gru(nn, params["input_size"], params)
                self.attention = nn.Linear(params["hidden_size"], 1)
                self.batch_norm = nn.BatchNorm1d(params["hidden_size"])
                self.head = nn.Linear(params["hidden_size"], 1)

            def forward(self, x):
                patches = _patches(x, params)
                batch = patches.shape[0]
                _, hidden = self.gru(patches.reshape(
                    batch * params["patch_count"], params["patch_length"], params["input_size"],
                ))
                daily = hidden[-1].reshape(batch, params["patch_count"], params["hidden_size"])
                # Report p.6: softmax(tanh(w h_i + b)), normalized over days.
                weights = torch.softmax(torch.tanh(self.attention(daily)), dim=1)
                pooled = torch.sum(weights * daily, dim=1)
                return self.head(self.batch_norm(pooled)).reshape(-1)

        return Model()
