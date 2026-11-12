from __future__ import annotations

from yq_ml_alpha.models.bar_training import DatewiseICBarAlphaModel
from yq_ml_alpha.models.resnet_gru_encoder import normalize_resnet_gru_params, resnet_gru_encoder_type


class ResNetGRUAlphaModel(DatewiseICBarAlphaModel):
    """Shared image ResNet, temporal GRU and 64-factor equal-weight head."""

    def _normalize_params(self, raw):
        return normalize_resnet_gru_params(raw)

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

        class Model(resnet_gru_encoder_type(nn)):
            def __init__(self):
                super().__init__(params)
                self.factor_head = nn.Linear(params["hidden_size"], params["base_factors"])
                self.factor_norm = nn.BatchNorm1d(params["base_factors"])
                self.output_norm = nn.BatchNorm1d(1)
                self.register_buffer("equal_weights", torch.full((params["base_factors"],), 1.0 / params["base_factors"]))


            def forward_factors(self, x):
                return self.factor_norm(self.factor_head(self.encode(x)))

            def combine(self, base):
                score = base @ self.equal_weights
                return self.output_norm(score[:, None]).reshape(-1)

            def forward(self, x):
                return self.combine(self.forward_factors(x))

        return Model()
