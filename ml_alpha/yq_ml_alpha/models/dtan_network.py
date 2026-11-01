"""Transformer generator with output/latent heads and two independent critics."""
from __future__ import annotations


def build_generator(nn, params):
    import math
    import torch

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            width, length = params["d_model"], params["sequence_length"]
            self.input_projection = nn.Linear(params["input_size"], width)
            positions = torch.arange(length, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
            encoding = torch.zeros(length, width)
            encoding[:, 0::2] = torch.sin(positions * frequency)
            encoding[:, 1::2] = torch.cos(positions * frequency[:width // 2])
            self.register_buffer("position_encoding", encoding)
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, params["attention_heads"], params["ffn_size"],
                    dropout=params["dropout"], activation="relu", layer_norm_eps=params["norm_eps"], batch_first=True)
                for _ in range(params["transformer_layers"])
            ])
            self.factor_head = nn.Sequential(nn.Linear(width, params["head_hidden"]), nn.ReLU(), nn.Linear(params["head_hidden"], 1))
            self.latent_projection = nn.Linear(length * width, params["latent_size"])

        def encode(self, values):
            if values.ndim != 3 or tuple(values.shape[1:]) != (params["sequence_length"], params["input_size"]):
                raise ValueError("DTAN expects the configured observed market/style sequence")
            hidden = self.input_projection(values) + self.position_encoding
            for block in self.blocks:
                hidden = block(hidden)
            return hidden

        def forward(self, values):
            hidden = self.encode(values)
            return self.factor_head(hidden[:, -1]).reshape(-1), self.latent_projection(hidden.flatten(1))

        def score(self, values):
            # Production inference has no dependency on critics or references.
            return self.factor_head(self.encode(values)[:, -1]).reshape(-1)

    return Generator()


def build_critic(nn, input_size, params):
    layers = []
    current = input_size
    for index, width in enumerate(params["critic_hidden_layers"]):
        layers.append(nn.Linear(current, width))
        if index == 0:
            layers.append(nn.LayerNorm(width, eps=params["norm_eps"]))
        layers.append(nn.LeakyReLU(params["leaky_slope"]))
        current = width
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)
