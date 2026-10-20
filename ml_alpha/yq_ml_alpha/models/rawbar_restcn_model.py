"""DFZQ ResNet + temporal convolution over the existing 30-session rawbar input."""
from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.rawbar_model import RawBarAGRUAlphaModel, _normalize_panel, _register_panel_stats


class RawBarRESTCNAlphaModel(RawBarAGRUAlphaModel):
    """Single-output RESTCN using the shared date-wise IC training lifecycle."""

    architecture = "restcn"

    def _normalize_params(self, raw):
        params = super()._normalize_params({"hidden_size": 64, "num_layers": 4, "dropout": 0.1, **raw})
        params["residual_blocks"] = int(params.get("residual_blocks", 3))
        params["kernel_size"] = int(params.get("kernel_size", 3))
        for key in ("hidden_size", "num_layers", "residual_blocks", "kernel_size", "batch_size"):
            if params[key] <= 0:
                raise ValueError(f"RESTCN {key} must be positive")
        if not np.isfinite(params["dropout"]) or not 0 <= params["dropout"] < 1:
            raise ValueError("RESTCN dropout must be finite and in [0, 1)")
        # Two convolutions per residual block; TCN dilation doubles at each layer.
        receptive_field = 1 + 2 * (params["kernel_size"] - 1) * (
            params["residual_blocks"] + sum(2 ** level for level in range(params["num_layers"]))
        )
        if receptive_field < params["sequence_length"]:
            raise ValueError("RESTCN receptive field must cover the complete input sequence")
        params["receptive_field"] = receptive_field
        return params

    def _build_regressor(self, nn):
        return _RESTCNRegressor(nn, self.params, getattr(self, "_panel_stats", None))

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            residual_blocks=self.params["residual_blocks"], kernel_size=self.params["kernel_size"],
            tcn_dilations=[2 ** level for level in range(self.params["num_layers"])],
            receptive_field=self.params["receptive_field"],
            temporal_padding="left_only_all_convolutions", aggregation="last_step_linear_scalar",
        )


class _RESTCNRegressor:
    def __new__(cls, nn, params, stats=None):
        import torch
        from torch.nn import functional as functional

        class CausalConv(nn.Module):
            def __init__(self, inputs, outputs, dilation, weight_norm):
                super().__init__()
                self.left = (params["kernel_size"] - 1) * dilation
                conv = nn.Conv1d(inputs, outputs, params["kernel_size"], dilation=dilation)
                self.conv = nn.utils.weight_norm(conv) if weight_norm else conv

            def forward(self, x):
                return self.conv(functional.pad(x, (self.left, 0)))

        class ResidualBlock(nn.Module):
            def __init__(self, inputs, dilation, temporal):
                super().__init__()
                width = params["hidden_size"]
                dropout = params["dropout"] if temporal else 0.0
                self.first = CausalConv(inputs, width, dilation, temporal)
                self.second = CausalConv(width, width, dilation, temporal)
                self.skip = nn.Identity() if inputs == width else nn.Conv1d(inputs, width, 1)
                self.dropout = nn.Dropout(dropout)
                self.temporal = temporal

            def forward(self, x):
                hidden = self.dropout(torch.relu(self.first(x)))
                hidden = self.second(hidden)
                if self.temporal:
                    hidden = self.dropout(torch.relu(hidden))
                return torch.relu(hidden + self.skip(x))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                _register_panel_stats(self, stats)
                self.local = nn.Sequential(*[
                    ResidualBlock(params["input_size"] if level == 0 else params["hidden_size"], 1, False)
                    for level in range(params["residual_blocks"])
                ])
                self.temporal = nn.Sequential(*[
                    ResidualBlock(params["hidden_size"], 2 ** level, True)
                    for level in range(params["num_layers"])
                ])
                self.head = nn.Linear(params["hidden_size"], 1)

            def encode(self, x):
                if x.ndim != 3 or tuple(x.shape[1:]) != (params["sequence_length"], params["input_size"]):
                    raise ValueError("RESTCN expects the configured [stocks, sessions, features] tensor")
                x = _normalize_panel(torch, self, x).transpose(1, 2)
                return self.temporal(self.local(x))[:, :, -1]

            def forward(self, x):
                return self.head(self.encode(x)).reshape(-1)

        return Model()
