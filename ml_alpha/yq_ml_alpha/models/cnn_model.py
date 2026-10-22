from __future__ import annotations

from typing import Any

import pandas as pd

from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.mlp_model import MLPAlphaModel


class CNNAlphaModel(MLPAlphaModel):
    """Huatai factor-by-time CNN with a scalar return-regression head.

    Replaces the legacy single-date Conv1d model. The shared sequence loader
    supplies time-major flattened columns; each minibatch is reshaped here.
    Training, early stopping, diagnostics and checkpoints reuse the MLP engine.
    """

    def _build_model(self, nn):
        if "sequence_length" not in self.params:
            raise ValueError(
                "CNNAlphaModel now requires factor-time sequences; legacy 1D CNN "
                "configs/checkpoints are incompatible and must be retrained"
            )
        return _CNNRegressor(nn, self.input_dim, self.params)

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        super().fit(train_data, valid_data, context)
        sequence_length = int(self.params["sequence_length"])
        self.model_info.update(
            architecture="huatai_factor_time_cnn",
            factor_count=self.input_dim // sequence_length,
            sequence_length=sequence_length,
            sequence_frequency=self.params.get("sequence_frequency", "daily"),
            conv_channels=int(self.params.get("conv_channels", 10)),
            kernel_size=[int(self.params.get("factor_kernel_size", 5)),
                         int(self.params.get("time_kernel_size", 5))],
            pooling=False,
            dense_init_std=float(self.params.get("dense_init_std", 0.1)),
        )


class _CNNRegressor:
    def __new__(cls, nn, input_dim: int, params: dict[str, Any]):
        sequence_length = int(params["sequence_length"])
        if sequence_length <= 0 or input_dim % sequence_length:
            raise ValueError("CNN input width must be divisible by positive sequence_length")
        factor_count = input_dim // sequence_length
        factor_kernel = int(params.get("factor_kernel_size", 5))
        time_kernel = int(params.get("time_kernel_size", 5))
        channels = int(params.get("conv_channels", 10))
        if not 1 <= factor_kernel <= factor_count or not 1 <= time_kernel <= sequence_length:
            raise ValueError("CNN kernel must fit both the factor and time dimensions")
        hidden_layers = params["hidden_layers"]
        if channels <= 0 or not hidden_layers or any(value <= 0 for value in hidden_layers):
            raise ValueError("CNN channels and hidden widths must be positive")
        if params["activation"] != "relu":
            raise ValueError("Huatai CNN uses relu activation")
        dropout = float(params["dropout"])
        init_std = float(params.get("dense_init_std", 0.1))
        if not 0.0 <= dropout < 1.0 or not 0.0 < init_std < float("inf"):
            raise ValueError("CNN requires dropout in [0, 1) and finite positive dense_init_std")
        flattened_dim = channels * (factor_count - factor_kernel + 1) * (sequence_length - time_kernel + 1)

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, channels, kernel_size=(factor_kernel, time_kernel), padding=0)
                nn.init.xavier_uniform_(self.conv.weight)
                nn.init.zeros_(self.conv.bias)
                self.activation = nn.ReLU()
                layers = []
                width = flattened_dim
                for hidden in hidden_layers:
                    layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
                    width = hidden
                layers.append(nn.Linear(width, 1))
                self.head = nn.Sequential(*layers)
                for layer in self.head:
                    if isinstance(layer, nn.Linear):
                        nn.init.trunc_normal_(layer.weight, mean=0.0, std=init_std,
                                              a=-2.0 * init_std, b=2.0 * init_std)
                        nn.init.zeros_(layer.bias)

            def forward(self, x):
                # Loader order: [t0/f0..fF, t1/f0..fF, ...]; CNN axes: [N, 1, F, T].
                if x.ndim != 2 or x.shape[1] != input_dim:
                    raise ValueError("CNN expects flattened time-major factor sequences")
                panel = x.reshape(x.shape[0], sequence_length, factor_count).transpose(1, 2).unsqueeze(1)
                encoded = self.activation(self.conv(panel))
                return self.head(encoded.flatten(start_dim=1))

        return Model()
