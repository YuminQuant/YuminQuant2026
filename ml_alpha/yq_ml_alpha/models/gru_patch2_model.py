from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel
from yq_ml_alpha.models.gru_patch_inputs import _gru, _params, _patches


class GRUPatch2AlphaModel(BarGRUAlphaModel):
    """Independent per-feature GRUs over daily patches, then concatenation."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _params(raw, "patch2")

    def _build_regressor(self, nn):
        return _GRUPatch2Regressor(nn, self.params)


class _GRUPatch2Regressor:
    def __new__(cls, nn, params: dict[str, Any]):
        import torch

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.grus = nn.ModuleList([
                    _gru(nn, params["patch_length"], params) for _ in range(params["input_size"])
                ])
                width = params["input_size"] * params["hidden_size"]
                self.batch_norm = nn.BatchNorm1d(width)
                self.head = nn.Linear(width, 1)

            def forward(self, x):
                patches = _patches(x, params)
                features = []
                for channel, gru in enumerate(self.grus):
                    # [stock, day, intraday slot]; each original field has its own weights.
                    _, hidden = gru(patches[:, :, :, channel].contiguous())
                    features.append(hidden[-1])
                combined = torch.cat(features, dim=1)
                return self.head(self.batch_norm(combined)).reshape(-1)

        return Model()
