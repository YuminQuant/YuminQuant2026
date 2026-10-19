from __future__ import annotations

import numpy as np

from yq_ml_alpha.models.bar_gru_model import BarGRUAlphaModel, _bar_tensor_batch, _params as _bar_params


class RawBarAGRUAlphaModel(BarGRUAlphaModel):
    """Single-output attention GRU over a normalized daily rawbar panel."""

    architecture = "agru"

    def _normalize_params(self, raw):
        params = _bar_params({"sequence_length": 30, "hidden_size": 30, **raw})
        if params.get("architecture", self.architecture) != self.architecture:
            raise ValueError("rawbar checkpoint architecture mismatch")
        params["architecture"] = self.architecture
        params["panel_clip_sigma"] = float(params.get("panel_clip_sigma", 5.0))
        if params["input_size"] != 6 or not np.isfinite(params["panel_clip_sigma"]) or params["panel_clip_sigma"] <= 0:
            raise ValueError("rawbar requires six inputs and positive finite panel_clip_sigma")
        return params

    def _fit(self, train_data, valid_data, context, *, train_tensor, valid_tensor):
        params = self._normalize_params(context.model_params)
        self._panel_stats = _fit_panel_stats(train_data, context.feature_columns, params, train_tensor)
        try:
            super()._fit(train_data, valid_data, context, train_tensor=train_tensor, valid_tensor=valid_tensor)
            self.model_info["architecture"] = self.architecture
            self.model_info["panel_normalization"] = "train_only_per_feature_across_stocks_and_lags"
        finally:
            del self._panel_stats

    def _build_regressor(self, nn):
        return _RawBarRegressor(nn, self.params, getattr(self, "_panel_stats", None))


class RawBarLSTMAlphaModel(RawBarAGRUAlphaModel):
    """Single-output LSTM over the same normalized daily rawbar panel."""

    architecture = "lstm"


def _fit_panel_stats(frame, columns, params, tensor):
    """Two streaming passes, train only; no transformed dataset copy or disk cache."""
    if frame.empty:
        raise ValueError("rawbar panel normalization requires training rows")
    rows = frame.index.to_numpy()
    chunk_size = int(params["batch_size"])
    if chunk_size <= 0:
        raise ValueError("batch_size must be positive")

    def moments(bounds=None):
        count, mean, m2 = 0, np.zeros(6), np.zeros(6)
        for start in range(0, len(rows), chunk_size):
            block = _bar_tensor_batch(frame, columns, rows[start:start + chunk_size], params, tensor).astype("float64").reshape(-1, 6)
            if bounds is not None:
                block = np.clip(block, bounds[0], bounds[1])
            n = len(block)
            block_mean = block.mean(axis=0)
            delta = block_mean - mean
            total = count + n
            m2 += ((block - block_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / total
            mean += delta * n / total
            count = total
        return mean, np.sqrt(m2 / count)

    center, scale = moments()
    lower = center - params["panel_clip_sigma"] * scale
    upper = center + params["panel_clip_sigma"] * scale
    mean, std = moments((lower, upper))
    std = np.where(std > 1e-8, std, 1.0)
    return lower, upper, mean, std


class _RawBarRegressor:
    def __new__(cls, nn, params, stats=None):
        import torch

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                defaults = (np.full(6, -np.inf), np.full(6, np.inf), np.zeros(6), np.ones(6))
                for name, values in zip(("panel_lower", "panel_upper", "panel_mean", "panel_std"),
                                        defaults if stats is None else stats):
                    self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
                recurrent = nn.GRU if params["architecture"] == "agru" else nn.LSTM
                self.recurrent = recurrent(input_size=6, hidden_size=params["hidden_size"],
                                           num_layers=params["num_layers"], batch_first=True,
                                           dropout=params["dropout"] if params["num_layers"] > 1 else 0.0)
                if params["architecture"] == "agru":
                    self.attention = nn.Sequential(nn.Linear(params["hidden_size"], params["hidden_size"]),
                                                   nn.Tanh(), nn.Linear(params["hidden_size"], 1, bias=False))
                self.head = nn.Linear(params["hidden_size"], 1)

            def forward(self, x):
                x = torch.minimum(torch.maximum(x, self.panel_lower), self.panel_upper)
                x = (x - self.panel_mean) / self.panel_std
                output, state = self.recurrent(x)
                if params["architecture"] == "agru":
                    weights = torch.softmax(self.attention(output), dim=1)
                    summary = torch.sum(weights * output, dim=1)
                else:
                    summary = state[0][-1]
                return self.head(summary).reshape(-1)

        return Model()
