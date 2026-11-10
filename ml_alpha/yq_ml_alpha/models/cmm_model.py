"""Characteristic-managed momentum and reversal on the raw weekly bar panel."""
from __future__ import annotations

import math

from yq_ml_alpha.features.transforms import apply_cross_section_transform
from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel, _params
from yq_ml_alpha.models.resnet_gru_model import _ResNetGRURegressor


class CMMWeekAlphaModel(BarGRUAlphaModel):
    def _normalize_params(self, raw):
        params = _params(raw)
        if (params["sequence_length"], params["input_size"]) != (150, 7):
            raise ValueError("CMM week requires 150 sessions, six bar features and one daily return")
        params["return_lookback"] = int(params["return_lookback"])
        params["half_life"] = float(params["half_life"])
        params["conv_channels"] = int(params["conv_channels"])
        params["kernel_size"] = int(params["kernel_size"])
        if not 1 <= params["return_lookback"] <= params["sequence_length"]:
            raise ValueError("CMM return_lookback must fit the bar history")
        if not math.isfinite(params["half_life"]) or params["half_life"] <= 0:
            raise ValueError("CMM half_life must be finite and positive")
        if params["conv_channels"] < 1 or params["kernel_size"] not in (1, 3, 5):
            raise ValueError("CMM requires positive conv_channels and kernel_size of 1, 3 or 5")
        return params

    def _build_regressor(self, nn):
        import torch

        params = self.params
        lookback = params["return_lookback"]

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _ResNetGRURegressor(
                    nn, {**params, "input_size": 6, "patch_width": 5}, encoder_only=True,
                )
                self.head = nn.Linear(params["hidden_size"], 3)
                ages = torch.arange(lookback - 1, -1, -1, dtype=torch.float32)
                self.register_buffer("decay", torch.exp2(-ages / params["half_life"]))

            def return_weights(self, inputs):
                returns = inputs[:, -lookback:, 6]
                hidden = self.encoder.encode(inputs[:, :, :6])
                momentum, reversal, gate_logit = self.head(hidden).split(1, dim=-1)
                gate = torch.sigmoid(gate_logit)
                positive = torch.softmax(momentum * returns + self.decay, dim=-1)
                negative = torch.softmax(reversal * returns + self.decay, dim=-1)
                return gate * positive - (1.0 - gate) * negative

            def forward(self, inputs):
                return (self.return_weights(inputs) * inputs[:, -lookback:, 6]).sum(dim=-1)

        return Model()

    def _training_loss(self, torch, inputs, target):
        valid = torch.isfinite(target)
        if int(valid.sum()) < 2:
            return None
        return (self.model(inputs[valid]) - target[valid]).square().mean()

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        # Standardize the entire labeled date before the training loop slices stock batches.
        train_data = apply_cross_section_transform(train_data, "zscore", [], [context.label_column])
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(loss="cross_section_standardized_label_mse",
                               validation_loss="negative_datewise_pearson_ic",
                               return_lookback=self.params["return_lookback"],
                               half_life=self.params["half_life"])
