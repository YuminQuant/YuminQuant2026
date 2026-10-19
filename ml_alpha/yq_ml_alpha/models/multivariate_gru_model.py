from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.bar_gru_model import _negative_ic_loss
from yq_ml_alpha.models.rawbar_model import RawBarAGRUAlphaModel, _RawBarRegressor


class MultivariateGRUAlphaModel(RawBarAGRUAlphaModel):
    """Daily GRU with decorrelated base factors and a fixed equal-weight output."""

    architecture = "gru"

    def _normalize_params(self, raw):
        params = super()._normalize_params({"hidden_size": 64, "num_layers": 2, "dropout": 0.1, **raw})
        params["base_factors"] = int(params.get("base_factors", 64))
        params["orthogonal_lambda"] = float(params.get("orthogonal_lambda", 0.05))
        if params["base_factors"] < 2:
            raise ValueError("multivariate GRU requires at least two base factors")
        if not np.isfinite(params["orthogonal_lambda"]) or params["orthogonal_lambda"] < 0:
            raise ValueError("orthogonal_lambda must be finite and nonnegative")
        if params["batch_size"] < 2:
            raise ValueError("multivariate GRU BatchNorm requires batch_size >= 2")
        return params

    def _build_regressor(self, nn):
        return _MultivariateGRURegressor(nn, self.params, getattr(self, "_panel_stats", None))

    def _training_loss(self, torch, inputs, target):
        base = self.model.forward_factors(inputs)
        score = self.model.aggregate(base)
        loss = _negative_ic_loss(torch, score, target)
        if loss is None:
            return None
        valid = torch.isfinite(target.reshape(-1)) & torch.isfinite(base).all(dim=1)
        return loss + self.params["orthogonal_lambda"] * _correlation_penalty(torch, base[valid])

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            base_factors=self.params["base_factors"],
            orthogonal_lambda=self.params["orthogonal_lambda"],
            aggregation="fixed_equal_weight_mean",
            loss="negative_datewise_pearson_ic_plus_base_factor_correlation_penalty",
            validation_loss="negative_datewise_pearson_ic",
        )


def _correlation_penalty(torch, base):
    """RMS off-diagonal correlation, computed within the current date batch."""
    centered = base - base.mean(dim=0, keepdim=True)
    normalized = centered / centered.square().mean(dim=0, keepdim=True).clamp_min(1e-12).sqrt()
    corr = normalized.T @ normalized / len(base)
    mask = ~torch.eye(base.shape[1], device=base.device, dtype=torch.bool)
    return (corr[mask].square().mean() + 1e-12).sqrt()


class _MultivariateGRURegressor:
    def __new__(cls, nn, params, stats=None):
        import torch

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _RawBarRegressor(nn, params, stats)
                self.encoder.head = nn.Identity()
                count = params["base_factors"]
                self.factor_head = nn.Linear(params["hidden_size"], count)
                self.factor_norm = nn.BatchNorm1d(count)
                # A checkpointed buffer, not an optimizer parameter or a learned head.
                self.register_buffer("equal_weights", torch.full((count,), 1.0 / count))

            def forward_factors(self, x):
                return self.factor_norm(self.factor_head(self.encoder.encode(x)))

            def aggregate(self, base):
                return base @ self.equal_weights

            def forward(self, x):
                return self.aggregate(self.forward_factors(x))

        return Model()
