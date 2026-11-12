from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_gru_model import _params as _bar_params


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
