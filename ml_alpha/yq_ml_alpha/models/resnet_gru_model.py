from __future__ import annotations

import math

from yq_ml_alpha.models.adversarial import batchnorm_without_running_updates, linf_adversarial_inputs
from yq_ml_alpha.models.bar_training import DatewiseICBarAlphaModel
from yq_ml_alpha.models.resnet_gru_encoder import normalize_resnet_gru_params, resnet_gru_encoder_type


class ResNetGRUAlphaModel(DatewiseICBarAlphaModel):
    """Shared image ResNet, temporal GRU and 64-factor equal-weight head."""

    def _normalize_params(self, raw):
        params = normalize_resnet_gru_params(raw)
        if "adversarial" in params:
            options = params["adversarial"]
            if not isinstance(options, dict) or set(options) != {"steps", "epsilon", "step_size", "weight"}:
                raise ValueError("adversarial training requires steps, epsilon, step_size and weight")
            if type(options["steps"]) is not int or options["steps"] < 1:
                raise ValueError("adversarial steps must be a positive integer")
            for key in ("epsilon", "step_size", "weight"):
                value = options[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"adversarial {key} must be finite and nonnegative")
            if not 0 < options["step_size"] <= options["epsilon"]:
                raise ValueError("adversarial step_size must be positive and no larger than epsilon")
        return params

    def _build_regressor(self, nn):
        return _ResNetGRURegressor(nn, self.params)

    @staticmethod
    def _standardized_batch(torch, inputs, target):
        valid = torch.isfinite(target)
        if int(valid.sum()) < 2:
            return None
        target = target[valid]
        scale = target.std(unbiased=False)
        if float(scale.detach()) <= 1e-8:
            return None
        return inputs[valid], (target - target.mean()) / scale

    def _clean_loss(self, torch, inputs, target):
        base = self.model.forward_factors(inputs)
        score = self.model.combine(base)
        n, k = base.shape
        # Report formula: lambda / (N K^2) * ||Z'Z||_F, including the diagonal.
        penalty = torch.norm(base.T @ base, p="fro") / (n * k * k)
        return (score - target).square().mean() + self.params["orthogonal_lambda"] * penalty

    def _training_loss(self, torch, inputs, target):
        batch = self._standardized_batch(torch, inputs, target)
        return None if batch is None else self._clean_loss(torch, *batch)

    def _backward_batch(self, torch, inputs, target):
        options = self.params.get("adversarial")
        if not options or options["weight"] == 0:
            return super()._backward_batch(torch, inputs, target)
        batch = self._standardized_batch(torch, inputs, target)
        if batch is None:
            return None
        inputs, target = batch
        clean_loss = self._clean_loss(torch, inputs, target)
        clean_loss.backward()
        # Release the clean activations before the attack and adversarial
        # passes, then accumulate both parameter gradients for one Adam step.
        with batchnorm_without_running_updates(self.model.factor_norm, self.model.output_norm):
            perturbed = linf_adversarial_inputs(torch, inputs, self.model, target,
                steps=options["steps"], epsilon=options["epsilon"], step_size=options["step_size"])
            adv_loss = options["weight"] * (self.model(perturbed) - target).square().mean()
            adv_loss.backward()
        return clean_loss.detach() + adv_loss.detach()

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
        if self.params.get("adversarial", {}).get("weight", 0) > 0:
            self.model_info.update(
                loss="standardized_mse_plus_gram_frobenius_penalty_and_adversarial_mse",
                adversarial=self.params["adversarial"],
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
