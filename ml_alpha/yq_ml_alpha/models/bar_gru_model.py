from __future__ import annotations

from typing import Any

from yq_ml_alpha.models.bar_training import DatewiseICBarAlphaModel


class BarGRUAlphaModel(DatewiseICBarAlphaModel):
    """End-to-end bar panel GRU trained with date-wise negative IC loss."""

    def _build_regressor(self, nn):
        return _BarGRURegressor(nn, self.params)


class _BarGRURegressor:
    def __new__(cls, nn, params: dict[str, Any]):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["num_layers"]) > 1 else 0.0,
                )
                self.batch_norm = nn.BatchNorm1d(int(params["hidden_size"]))
                self.head = nn.Linear(int(params["hidden_size"]), 1)

            def forward(self, x):
                _, hidden = self.gru(x)
                last = hidden[-1]
                normalized = self.batch_norm(last)
                return self.head(normalized).reshape(-1)

        return Model()
