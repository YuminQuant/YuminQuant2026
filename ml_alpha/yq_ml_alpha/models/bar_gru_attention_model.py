from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel, _params as _bar_params


class BarGRUAttentionAlphaModel(BarGRUAlphaModel):
    """Huatai report 68: temporal attention and last-state GRU concatenation."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        params = _bar_params(raw)
        if params.get("architecture", "gru_attention") != "gru_attention":
            raise ValueError("bar GRU attention checkpoint architecture mismatch")
        params["architecture"] = "gru_attention"
        if params["hidden_size"] <= 0 or params["num_layers"] <= 0:
            raise ValueError("bar GRU attention requires positive hidden_size and num_layers")
        return params

    def _build_regressor(self, nn):
        return _BarGRUAttentionRegressor(nn, self.params)

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
        self.model_info.update(
            architecture="gru_attention",
            attention="scalar_tanh_linear_temporal_softmax",
            readout="concat_last_hidden_and_attention_summary_then_batchnorm_linear",
            readout_size=2 * int(self.params["hidden_size"]),
        )


class _BarGRUAttentionRegressor:
    def __new__(cls, nn, params: dict[str, Any]):
        import torch

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden_size = int(params["hidden_size"])
                self.gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=hidden_size,
                    num_layers=int(params["num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["num_layers"]) > 1 else 0.0,
                )
                # Report p6: u_i=tanh(W*z_i+b), gamma=softmax(u) over time.
                self.attention = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())
                self.batch_norm = nn.BatchNorm1d(2 * hidden_size)
                self.head = nn.Linear(2 * hidden_size, 1)

            def forward(self, x):
                sequence, hidden = self.gru(x)
                weights = torch.softmax(self.attention(sequence), dim=1)
                summary = torch.sum(weights * sequence, dim=1)
                joined = torch.cat((hidden[-1], summary), dim=1)
                return self.head(self.batch_norm(joined)).reshape(-1)

        return Model()
