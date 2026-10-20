"""DFZQ Time2Vec encoder with a learned temporal summary and last-step readout."""
from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.rawbar_model import RawBarAGRUAlphaModel, _normalize_panel, _register_panel_stats


class RawBarTransformerAlphaModel(RawBarAGRUAlphaModel):
    """Single-output Transformer over the same rawbar data as AGRU/LSTM."""

    architecture = "transformer"

    def _normalize_params(self, raw):
        params = super()._normalize_params({"hidden_size": 64, "num_layers": 2, "dropout": 0.1, **raw})
        params["attention_heads"] = int(params.get("attention_heads", 4))
        params["feedforward_size"] = int(params.get("feedforward_size", 128))
        params["time2vec_periodic_dim"] = int(params.get("time2vec_periodic_dim", 1))
        params["time_coordinate"] = str(params.get("time_coordinate", "relative_position"))
        for key in ("hidden_size", "num_layers", "attention_heads", "feedforward_size", "time2vec_periodic_dim", "batch_size"):
            if params[key] <= 0:
                raise ValueError(f"Transformer {key} must be positive")
        if params["hidden_size"] % params["attention_heads"]:
            raise ValueError("Transformer hidden_size must be divisible by attention_heads")
        if not np.isfinite(params["dropout"]) or not 0 <= params["dropout"] < 1:
            raise ValueError("Transformer dropout must be finite and in [0, 1)")
        if params["time_coordinate"] != "relative_position":
            raise ValueError("rawbar Transformer supports time_coordinate=relative_position only")
        return params

    def _build_regressor(self, nn):
        return _TransformerRegressor(nn, self.params, getattr(self, "_panel_stats", None))

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            attention_heads=self.params["attention_heads"], feedforward_size=self.params["feedforward_size"],
            time_coordinate=self.params["time_coordinate"], time2vec_periodic_dim=self.params["time2vec_periodic_dim"],
            time_encoding="concatenate_linear_and_learned_sine",
            aggregation="time_linear_concat_last_step_linear_scalar",
            attention_scope="full_observed_history_within_each_stock",
        )


class _TransformerRegressor:
    def __new__(cls, nn, params, stats=None):
        import torch

        class EncoderBlock(nn.Module):
            def __init__(self):
                super().__init__()
                width = params["hidden_size"]
                self.attention = nn.MultiheadAttention(
                    width, params["attention_heads"], dropout=params["dropout"], batch_first=True,
                )
                self.attention_norm = nn.LayerNorm(width)
                self.feedforward = nn.Sequential(
                    nn.Linear(width, params["feedforward_size"]), nn.ReLU(), nn.Dropout(params["dropout"]),
                    nn.Linear(params["feedforward_size"], width),
                )
                self.feedforward_norm = nn.LayerNorm(width)
                self.dropout = nn.Dropout(params["dropout"])

            def forward(self, x):
                # Attention is over the 30 observed sessions, never over other stocks.
                attention, _ = self.attention(x, x, x, need_weights=False)
                x = self.attention_norm(x + self.dropout(attention))
                return self.feedforward_norm(x + self.dropout(self.feedforward(x)))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                _register_panel_stats(self, stats)
                self.register_buffer("time_positions", torch.linspace(0.0, 1.0, params["sequence_length"]).reshape(1, -1, 1))
                self.time_linear = nn.Linear(1, 1)
                self.time_periodic = nn.Linear(1, params["time2vec_periodic_dim"])
                self.input_projection = nn.Linear(params["input_size"] + 1 + params["time2vec_periodic_dim"],
                                                  params["hidden_size"])
                # Construct each block independently rather than clone identical initial weights.
                self.blocks = nn.Sequential(*[EncoderBlock() for _ in range(params["num_layers"])])
                self.time_summary = nn.Linear(params["sequence_length"], 1)
                self.head = nn.Linear(2 * params["hidden_size"], 1)

            def encode(self, x):
                if x.ndim != 3 or tuple(x.shape[1:]) != (params["sequence_length"], params["input_size"]):
                    raise ValueError("Transformer expects the configured [stocks, sessions, features] tensor")
                x = _normalize_panel(torch, self, x)
                time_features = torch.cat([
                    self.time_linear(self.time_positions), torch.sin(self.time_periodic(self.time_positions)),
                ], dim=-1).expand(x.shape[0], -1, -1)
                hidden = self.blocks(self.input_projection(torch.cat([x, time_features], dim=-1)))
                summary = self.time_summary(hidden.transpose(1, 2)).squeeze(-1)
                return torch.cat([summary, hidden[:, -1]], dim=-1)

            def forward(self, x):
                return self.head(self.encode(x)).reshape(-1)

        return Model()
