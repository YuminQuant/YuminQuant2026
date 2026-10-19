from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel, _params as _bar_params


class GRUPatch1AlphaModel(BarGRUAlphaModel):
    """Shared intraday GRU, then attention across trading-day patches."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _params(raw, "patch1")

    def _build_regressor(self, nn):
        return _GRUPatch1Regressor(nn, self.params)


class GRUPatch2AlphaModel(BarGRUAlphaModel):
    """Independent per-feature GRUs over daily patches, then concatenation."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _params(raw, "patch2")

    def _build_regressor(self, nn):
        return _GRUPatch2Regressor(nn, self.params)


def _params(raw: dict[str, Any], architecture: str) -> dict[str, Any]:
    params = _bar_params({"lr": 0.005, **raw})
    stored_architecture = params.get("patch_architecture", architecture)
    if stored_architecture != architecture:
        raise ValueError(f"expected {architecture} checkpoint, got {stored_architecture}")
    params["patch_architecture"] = architecture
    params["patch_length"] = int(params.get("patch_length", 16))
    if params["patch_length"] <= 0 or params["sequence_length"] % params["patch_length"]:
        raise ValueError("sequence_length must be a positive multiple of patch_length")
    params["patch_count"] = params["sequence_length"] // params["patch_length"]
    if params["hidden_size"] <= 0 or params["num_layers"] <= 0 or params["batch_size"] <= 0:
        raise ValueError("hidden_size, num_layers and batch_size must be positive")
    return params


def _patches(x, params):
    expected = (params["sequence_length"], params["input_size"])
    if x.ndim != 3 or tuple(x.shape[1:]) != expected:
        raise ValueError(f"GRUPatch expects [N,{expected[0]},{expected[1]}], got {tuple(x.shape)}")
    # Provider order is oldest day first, then intraday bar, then feature.
    return x.reshape(x.shape[0], params["patch_count"], params["patch_length"], params["input_size"])


def _gru(nn, input_size, params):
    return nn.GRU(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        batch_first=True,
        dropout=params["dropout"] if params["num_layers"] > 1 else 0.0,
    )


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
