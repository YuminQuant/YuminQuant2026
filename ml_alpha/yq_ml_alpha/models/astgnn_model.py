from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.cross_section_bar_model import CrossSectionBarAlphaModel
from yq_ml_alpha.models.resnet_gru_model import ResNetGRUAlphaModel, _ResNetGRURegressor


class ASTGNNAlphaModel(CrossSectionBarAlphaModel, ResNetGRUAlphaModel):
    """Data-driven graph over ResNet-GRU embeddings, with an equal-weight head."""

    def _normalize_params(self, raw):
        params = super()._normalize_params(raw)
        params["graph_chunk_size"] = int(params.get("graph_chunk_size", 512))
        params["gradient_checkpointing"] = params.get("gradient_checkpointing", True)
        params["grad_clip"] = float(params.get("grad_clip", 5.0))
        if not isinstance(params["gradient_checkpointing"], bool):
            raise ValueError("gradient_checkpointing must be a boolean")
        if not 2 <= params["batch_size"] <= 5000 or params["graph_chunk_size"] < 1:
            raise ValueError("ASTGNN requires encoder chunks of 2..5000 and positive graph chunks")
        if any(params[key] < 1 for key in ("hidden_size", "num_layers", "epochs")) or params["patience"] < 0:
            raise ValueError("invalid ASTGNN training dimensions or epoch/patience settings")
        if not 0 <= params["dropout"] < 1:
            raise ValueError("dropout must be in [0,1)")
        for key in ("lr", "grad_clip", "weight_decay"):
            if not np.isfinite(params[key]) or params[key] < 0 or (key != "weight_decay" and params[key] == 0):
                raise ValueError(f"invalid {key}")
        return params

    def _build_regressor(self, nn):
        return _ASTGNNRegressor(nn, self.params)

    def _training_loss_encoded(self, torch, encoded, target):
        valid = torch.isfinite(target)
        target = target[valid]
        scale = target.std(unbiased=False)
        if float(scale.detach()) <= 1e-8:
            return None
        # Unlabeled nodes remain in the graph and BN; only supervision is masked.
        base = self.model.forward_factors_encoded(encoded)
        score = self.model.combine(base)[valid]
        base = base[valid]
        target = (target - target.mean()) / scale
        n, k = base.shape
        penalty = torch.norm(base.T @ base, p="fro") / (n * k * k)
        return (score - target).square().mean() + self.params["orthogonal_lambda"] * penalty

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            dataset=self.params["dataset"], sequence_length=self.params["sequence_length"],
            hidden_size=self.params["hidden_size"], num_layers=self.params["num_layers"],
            base_factors=self.params["base_factors"], orthogonal_lambda=self.params["orthogonal_lambda"],
            adjacency="row_softmax_relu_gru_hidden_gram_including_self",
            graph_convolution="relu((H + A @ H) @ W)",
            loss="full_date_standardized_mse_plus_gram_frobenius_penalty",
            aggregation="fixed_equal_weight_mean_then_batch_norm",
        )


class _ASTGNNRegressor:
    def __new__(cls, nn, params):
        import torch
        from torch.utils.checkpoint import checkpoint

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _ResNetGRURegressor(nn, params, encoder_only=True)
                self.graph_projection = nn.Linear(params["hidden_size"], params["hidden_size"], bias=False)
                self.factor_head = nn.Linear(params["hidden_size"], params["base_factors"])
                self.factor_norm = nn.BatchNorm1d(params["base_factors"])
                self.output_norm = nn.BatchNorm1d(1)
                self.register_buffer("equal_weights", torch.full((params["base_factors"],), 1.0 / params["base_factors"]))

            def encode(self, x):
                return self.encoder.encode(x)

            def graph_features(self, hidden):
                values = self.graph_projection(hidden)
                messages = []

                def aggregate(query, keys, projected):
                    # Exact full graph; no cosine/scaling/top-k/detached neighbors.
                    weights = torch.softmax(torch.relu(query @ keys.T), dim=1)
                    return weights @ projected

                for start in range(0, len(hidden), params["graph_chunk_size"]):
                    query = hidden[start:start + params["graph_chunk_size"]]
                    if self.training and params["gradient_checkpointing"]:
                        message = checkpoint(aggregate, query, hidden, values, preserve_rng_state=True)
                    else:
                        message = aggregate(query, hidden, values)
                    messages.append(message)
                # The report's identity term is in addition to A's self edges.
                return torch.relu(values + torch.cat(messages, dim=0))

            def forward_factors_encoded(self, hidden):
                return self.factor_norm(self.factor_head(self.graph_features(hidden)))

            def combine(self, base):
                score = base @ self.equal_weights
                return self.output_norm(score[:, None]).reshape(-1)

            def forward_encoded(self, hidden):
                return self.combine(self.forward_factors_encoded(hidden))

            def forward(self, x):
                return self.forward_encoded(self.encode(x))

        return Model()
