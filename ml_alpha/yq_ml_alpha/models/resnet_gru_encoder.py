"""Convolutional and recurrent encoder for fixed-layout bar panels."""
from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.bar_panel_model import normalize_bar_params


def normalize_resnet_gru_params(raw):
    params = normalize_bar_params({"hidden_size": 64, "num_layers": 2, "dropout": 0.1, **raw})
    params["dataset"] = str(params.get("dataset", "week"))
    layout = {"week": (150, 6, 5), "msnew": (240, 5, 8)}
    if params["dataset"] not in layout:
        raise ValueError("ResNet GRU dataset must be week or msnew")
    steps, channels, width = layout[params["dataset"]]
    if (params["sequence_length"], params["input_size"]) != (steps, channels):
        raise ValueError(f"{params['dataset']} requires sequence_length={steps}, input_size={channels}")
    params["patch_width"] = width
    params["conv_channels"] = int(params.get("conv_channels", 32))
    params["kernel_size"] = int(params.get("kernel_size", 3))
    params["base_factors"] = int(params.get("base_factors", 64))
    params["orthogonal_lambda"] = float(params.get("orthogonal_lambda", 1.0))
    if params["kernel_size"] < 1 or params["kernel_size"] > width or params["kernel_size"] % 2 != 1:
        raise ValueError("kernel_size must be odd and no larger than the patch width")
    if params["conv_channels"] < 1 or params["base_factors"] < 2 or params["batch_size"] < 2:
        raise ValueError("invalid conv_channels, base_factors or batch_size")
    if not np.isfinite(params["orthogonal_lambda"]) or params["orthogonal_lambda"] < 0:
        raise ValueError("orthogonal_lambda must be finite and nonnegative")
    return params


def resnet_gru_encoder_type(nn):
    import torch

    class Encoder(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.width = params["patch_width"]
            self.channels = params["input_size"]
            self.crop = (params["kernel_size"] - 1) // 2
            self.conv = nn.Conv1d(self.channels, params["conv_channels"], params["kernel_size"])
            self.shortcut = nn.Conv1d(self.channels, params["conv_channels"], 1)
            encoded_size = params["conv_channels"] * (self.width - 2 * self.crop)
            self.gru = nn.GRU(encoded_size, params["hidden_size"], num_layers=params["num_layers"],
                              batch_first=True, dropout=params["dropout"] if params["num_layers"] > 1 else 0.0)

        def encode_images(self, x):
            # The provider is chronological: N x (30 * width) x channels.
            images = x.reshape(-1, self.width, self.channels).transpose(1, 2)
            shortcut = images[:, :, self.crop:-self.crop] if self.crop else images
            return torch.relu(self.conv(images)) + self.shortcut(shortcut)

        def encode(self, x):
            features = self.encode_images(x).reshape(x.shape[0], 30, -1)
            _, hidden = self.gru(features)
            return hidden[-1]

        def forward(self, inputs):
            return self.encode(inputs)

    return Encoder
