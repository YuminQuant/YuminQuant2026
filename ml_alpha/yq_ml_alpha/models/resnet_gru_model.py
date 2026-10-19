from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel, _params


class ResNetGRUAlphaModel(BarGRUAlphaModel):
    """Shared image ResNet, temporal GRU and 64-factor equal-weight head."""

    def _normalize_params(self, raw):
        params = _params({"hidden_size": 64, "num_layers": 2, "dropout": 0.1, **raw})
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

    def _build_regressor(self, nn):
        return _ResNetGRURegressor(nn, self.params)

    def _training_loss(self, torch, inputs, target):
        valid = torch.isfinite(target)
        if int(valid.sum()) < 2:
            return None
        target = target[valid]
        scale = target.std(unbiased=False)
        if float(scale.detach()) <= 1e-8:
            return None
        base = self.model.forward_factors(inputs[valid])
        score = self.model.combine(base)
        target = (target - target.mean()) / scale
        n, k = base.shape
        # Report formula: lambda / (N K^2) * ||Z'Z||_F, including the diagonal.
        penalty = torch.norm(base.T @ base, p="fro") / (n * k * k)
        return (score - target).square().mean() + self.params["orthogonal_lambda"] * penalty

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            dataset=self.params["dataset"],
            image_count=30,
            patch_width=self.params["patch_width"],
            conv_channels=self.params["conv_channels"],
            kernel_size=self.params["kernel_size"],
            base_factors=self.params["base_factors"],
            orthogonal_lambda=self.params["orthogonal_lambda"],
            loss="standardized_mse_plus_gram_frobenius_penalty",
            validation_loss="negative_datewise_pearson_ic",
            aggregation="fixed_equal_weight_mean_then_batch_norm",
        )


class _ResNetGRURegressor:
    def __new__(cls, nn, params):
        import torch

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.width = params["patch_width"]
                self.channels = params["input_size"]
                self.crop = (params["kernel_size"] - 1) // 2
                self.conv = nn.Conv1d(self.channels, params["conv_channels"], params["kernel_size"])
                self.shortcut = nn.Conv1d(self.channels, params["conv_channels"], 1)
                encoded_size = params["conv_channels"] * (self.width - 2 * self.crop)
                self.gru = nn.GRU(encoded_size, params["hidden_size"], num_layers=params["num_layers"],
                                  batch_first=True, dropout=params["dropout"] if params["num_layers"] > 1 else 0.0)
                self.factor_head = nn.Linear(params["hidden_size"], params["base_factors"])
                self.factor_norm = nn.BatchNorm1d(params["base_factors"])
                self.output_norm = nn.BatchNorm1d(1)
                self.register_buffer("equal_weights", torch.full((params["base_factors"],), 1.0 / params["base_factors"]))

            def encode_images(self, x):
                # The provider is chronological: N x (30 * width) x channels.
                images = x.reshape(-1, self.width, self.channels).transpose(1, 2)
                shortcut = images[:, :, self.crop:-self.crop] if self.crop else images
                return torch.relu(self.conv(images)) + self.shortcut(shortcut)

            def forward_factors(self, x):
                features = self.encode_images(x).reshape(x.shape[0], 30, -1)
                _, hidden = self.gru(features)
                return self.factor_norm(self.factor_head(hidden[-1]))

            def combine(self, base):
                score = base @ self.equal_weights
                return self.output_norm(score[:, None]).reshape(-1)

            def forward(self, x):
                return self.combine(self.forward_factors(x))

        return Model()
