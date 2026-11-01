"""Causal memory-spectrum and delay-coordinate primitives for SWSC LSG."""
from __future__ import annotations

import numpy as np


def fractional_difference_matrices(length, orders):
    """GL convolution matrices; truncate each series at its observed left edge."""
    if type(length) is not int or length < 1:
        raise ValueError("fractional difference length must be a positive integer")
    orders = np.asarray(orders, dtype=np.float64)
    if orders.ndim != 1 or not len(orders) or not np.isfinite(orders).all():
        raise ValueError("fractional difference orders must be a finite vector")
    if (orders < 0).any() or (orders > 1).any() or len(np.unique(orders)) != len(orders):
        raise ValueError("fractional difference orders must be distinct and in [0, 1]")
    weights = np.ones((len(orders), length), dtype=np.float64)
    for lag in range(1, length):
        weights[:, lag] = weights[:, lag - 1] * (lag - 1 - orders) / lag
    lags = np.arange(length)[:, None] - np.arange(length)[None, :]
    return np.where(lags[None] >= 0, weights[:, lags.clip(min=0)], 0.0).astype("float32")


def delay_coordinates(torch, values, *, steps, delay, dimension):
    """Return [stocks, steps, features, dimension], newest coordinate first."""
    if any(type(v) is not int or v < 1 for v in (steps, delay, dimension)):
        raise ValueError("delay-coordinate dimensions must be positive integers")
    required = steps + (dimension - 1) * delay
    if values.ndim != 3 or values.shape[1] != required:
        raise ValueError(f"delay coordinates require {required} observed sessions")
    return torch.stack([
        values[:, required - steps - lag * delay:required - lag * delay]
        for lag in range(dimension)
    ], dim=-1)


def build_lsg(nn, params):
    """One jointly trained FDS/FSDE/RAB/LSG network, with configurable widths."""
    import math
    import torch
    import torch.nn.functional as functional

    width = params["d_model"]
    features = params["input_size"]
    steps = params["temporal_steps"]

    class MemorySpectrum(nn.Module):
        def __init__(self):
            super().__init__()
            orders = params["spectrum_orders"]
            self.register_buffer("orders", torch.tensor(orders, dtype=torch.float32))
            for name, length in (("time_operator", steps), ("geometry_operator", params["sequence_length"])):
                self.register_buffer(name, torch.from_numpy(fractional_difference_matrices(length, orders)))
            self.raw_mean = nn.Parameter(torch.full((features,), math.log(params["mu_init"] / (1 - params["mu_init"]))))
            self.raw_sigma = nn.Parameter(torch.full((features,), math.log(math.expm1(params["sigma_init"]))))

        def forward(self, values, *, geometry):
            mean = self.raw_mean.sigmoid()
            sigma = functional.softplus(self.raw_sigma) + params["eps"]
            logits = -(self.orders[None] - mean[:, None]).square() / (2 * sigma[:, None].square())
            weights = logits.softmax(dim=-1)
            matrix = self.geometry_operator if geometry else self.time_operator
            # Linear fusion before convolution is equivalent to building all
            # five spectra, without a [stocks, sessions, features, orders] copy.
            operator = torch.einsum("kts,fk->fts", matrix, weights)
            return torch.einsum("fts,bsf->btf", operator, values)

    class RecurrenceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(width, 3 * width)
            self.output = nn.Linear(width, width)
            self.attention_norm = nn.LayerNorm(width)
            self.feedforward_norm = nn.LayerNorm(width)
            self.feedforward = nn.Sequential(
                nn.Linear(width, params["ffn_size"]), nn.ReLU(), nn.Dropout(params["dropout"]),
                nn.Linear(params["ffn_size"], width),
            )
            self.dropout = nn.Dropout(params["dropout"])
            self.distance_weight = nn.Parameter(torch.tensor(params["lambda_init"], dtype=torch.float32))

        def forward(self, values, distances):
            heads = params["attention_heads"]
            qkv = self.qkv(values).reshape(len(values), steps, 3, heads, width // heads)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            logits = query @ key.transpose(-1, -2) / math.sqrt(width // heads)
            logits = logits - self.distance_weight.clamp_min(0) * distances[:, None]
            attention = self.dropout(logits.softmax(dim=-1))
            summary = (attention @ value).transpose(1, 2).reshape(len(values), steps, width)
            hidden = self.attention_norm(values + self.dropout(self.output(summary)))
            return self.feedforward_norm(hidden + self.dropout(self.feedforward(hidden)))

    class LSG(nn.Module):
        def __init__(self):
            super().__init__()
            self.spectrum = MemorySpectrum()
            self.time_projection = nn.Linear(features, width)
            positions = torch.arange(steps, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
            encoding = torch.zeros(steps, width)
            encoding[:, 0::2] = torch.sin(positions * frequency)
            encoding[:, 1::2] = torch.cos(positions * frequency[:width // 2])
            self.register_buffer("position_encoding", encoding)
            self.blocks = nn.ModuleList([RecurrenceBlock() for _ in range(params["transformer_layers"])])
            self.shape_encoder = nn.Conv1d(1, params["conv_channels"], params["kernel_size"])
            self.geometry_gru = nn.GRU(params["conv_channels"], params["gru_hidden"],
                                       params["gru_layers"], batch_first=True,
                                       dropout=params["dropout"] if params["gru_layers"] > 1 else 0.0)
            self.geometry_projection = nn.Sequential(nn.Linear(features * params["gru_hidden"], width), nn.ReLU())
            self.gate_weight = nn.Parameter(torch.tensor(params["gate_init"], dtype=torch.float32))
            self.head = nn.Sequential(nn.Linear(2 * width, params["head_hidden"]), nn.ReLU(), nn.Linear(params["head_hidden"], 1))

        def distances(self, coordinates):
            # Accumulate one feature at a time instead of retaining the much
            # larger [stocks, features, steps, steps, delay_dimension] tensor.
            distances = coordinates.new_zeros((len(coordinates), steps, steps))
            for feature in range(features):
                points = coordinates[:, :, feature].contiguous()
                distances = distances + torch.cdist(points, points)
            distances = distances / features
            diagonal = torch.eye(steps, dtype=torch.bool, device=distances.device)
            distances = distances.masked_fill(diagonal[None], 0.0)
            if params["distance_normalization"] == "mean":
                distances = distances / distances.mean(dim=(-2, -1), keepdim=True).clamp_min(params["eps"])
            return distances

        def project_constraints(self):
            with torch.no_grad():
                for block in self.blocks:
                    block.distance_weight.clamp_(min=0)

        def forward(self, values):
            if values.ndim != 3 or tuple(values.shape[1:]) != (params["sequence_length"], features):
                raise ValueError("LSG expects the configured complete observed daily sequence")
            geometry = (self.spectrum(values, geometry=True)
                        if params["geometry_input"] == "spectrum" else values)
            coordinates = delay_coordinates(torch, geometry, steps=steps, delay=params["delay"], dimension=params["embedding_dim"])
            distances = self.distances(coordinates)
            shapes = coordinates.reshape(-1, 1, params["embedding_dim"])
            shapes = functional.relu(self.shape_encoder(shapes)).mean(dim=-1)
            shapes = shapes.reshape(len(values), steps, features, params["conv_channels"])
            shapes = shapes.permute(0, 2, 1, 3).reshape(len(values) * features, steps, params["conv_channels"])
            summaries = self.geometry_gru(shapes)[1][-1].reshape(len(values), features * params["gru_hidden"])
            geometry = self.geometry_projection(summaries)
            rarity = distances[:, -1, :-1].topk(params["nearest_k"], largest=False, dim=-1).values.mean(dim=-1)
            gate = 1 + functional.softplus(self.gate_weight) * functional.softplus(rarity)
            geometry = geometry * gate[:, None]
            temporal = self.spectrum(values[:, -steps:], geometry=False)
            temporal = self.time_projection(temporal) + self.position_encoding
            for block in self.blocks:
                temporal = block(temporal, distances)
            return self.head(torch.cat([temporal[:, -1], geometry], dim=-1)).reshape(-1)

    return LSG()
