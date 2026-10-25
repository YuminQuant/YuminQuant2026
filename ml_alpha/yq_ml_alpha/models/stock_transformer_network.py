"""Huatai AI69: full-date stock tokens with bounded attention query blocks."""
from __future__ import annotations

import math
from inspect import signature


def build_stock_transformer(nn, input_size, task_count, params):
    """Build a post-norm Transformer whose tokens are all stocks on one date.

    No input projection or position encoding is used. Zero channels only make
    the feature width divisible by the attention head count. Query chunks keep
    every stock as a key/value and never form independent attention groups.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    def positive_integer(value, name):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"stock Transformer {name} must be a positive integer")
        return int(value)

    input_size = positive_integer(input_size, "input_size")
    task_count = positive_integer(task_count, "task_count")
    heads = positive_integer(params.get("attention_heads", 2), "attention_heads")
    layers = positive_integer(params.get("num_layers", 2), "num_layers")
    feedforward = positive_integer(params.get("feedforward_size", 64), "feedforward_size")
    head_width = positive_integer(params.get("hidden_size", 256), "hidden_size")
    stock_chunk = positive_integer(params.get("batch_size", 3000), "batch_size")
    query_chunk = positive_integer(params.get("attention_chunk_size", 512), "attention_chunk_size")
    if stock_chunk > 3000 or query_chunk > stock_chunk:
        raise ValueError("stock Transformer attention chunks must not exceed stock chunks, capped at 3000")
    dropout = float(params.get("dropout", 0.1))
    slope = float(params.get("negative_slope", 0.01))
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("stock Transformer dropout must be finite and in [0, 1)")
    if not math.isfinite(slope) or slope < 0:
        raise ValueError("stock Transformer negative_slope must be finite and nonnegative")
    checkpointing = params.get("gradient_checkpointing", True)
    if not isinstance(checkpointing, bool):
        raise ValueError("stock Transformer gradient_checkpointing must be boolean")
    width = ((input_size + heads - 1) // heads) * heads
    head_dimension = width // heads
    checkpoint_options = dict(preserve_rng_state=True)
    if "use_reentrant" in signature(checkpoint).parameters:
        checkpoint_options["use_reentrant"] = True

    class EncoderBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Retain standard MHA parameter shapes and initialization while
            # computing its quadratic attention only for one query block.
            self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.attention_norm = nn.LayerNorm(width)
            self.attention_dropout = nn.Dropout(dropout)
            self.feedforward = nn.Sequential(
                nn.Linear(width, feedforward), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(feedforward, width), nn.Dropout(dropout),
            )
            self.feedforward_norm = nn.LayerNorm(width)

        def _attention_block(self, query, keys, values):
            # Shapes [heads, queries/ALL stocks, head dimension].
            queries = query.reshape(len(query), heads, head_dimension).transpose(0, 1)
            keys = keys.reshape(len(keys), heads, head_dimension).transpose(0, 1)
            values = values.reshape(len(values), heads, head_dimension).transpose(0, 1)
            logits = queries @ keys.transpose(1, 2) / math.sqrt(head_dimension)
            weights = functional.dropout(torch.softmax(logits, dim=-1), dropout, training=self.training)
            context = (weights @ values).transpose(0, 1).contiguous().reshape(len(query), width)
            return self.attention.out_proj(context)

        def forward(self, hidden):
            projected = [functional.linear(
                hidden[start:start + stock_chunk], self.attention.in_proj_weight,
                self.attention.in_proj_bias,
            ) for start in range(0, len(hidden), stock_chunk)]
            query, keys, values = torch.cat(projected, dim=0).chunk(3, dim=-1)
            del projected
            messages = []
            for start in range(0, len(hidden), query_chunk):
                query_rows = query[start:start + query_chunk]
                if self.training and checkpointing and torch.is_grad_enabled():
                    # Backward recomputes each score matrix and its dropout
                    # using the saved RNG state, rather than retaining N x N.
                    message = checkpoint(
                        self._attention_block, query_rows, keys, values, **checkpoint_options,
                    )
                else:
                    message = self._attention_block(query_rows, keys, values)
                messages.append(message)
            attended = torch.cat(messages, dim=0)
            hidden = self.attention_norm(hidden + self.attention_dropout(attended))
            # Finish this layer for ALL stocks before the next layer sees it.
            return torch.cat([
                self.feedforward_norm(rows + self.feedforward(rows))
                for rows in hidden.split(stock_chunk, dim=0)
            ], dim=0)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_size = input_size
            self.model_dim = width
            self.padding_features = width - input_size
            self.task_count = task_count
            self.blocks = nn.ModuleList([EncoderBlock() for _ in range(layers)])
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(width, head_width), nn.LeakyReLU(slope),
                    nn.Dropout(dropout), nn.Linear(head_width, 1),
                ) for _ in range(task_count)
            ])

        def forward(self, host_values, device):
            if (host_values.ndim != 2 or host_values.shape[1] != input_size
                    or len(host_values) == 0 or host_values.device.type != "cpu"):
                raise ValueError("stock Transformer requires a nonempty CPU [all stocks, factors] snapshot")
            dtype = self.blocks[0].attention.in_proj_weight.dtype
            inputs = []
            for rows in host_values.split(stock_chunk, dim=0):
                rows = rows.to(device=device, dtype=dtype)
                inputs.append(functional.pad(rows, (0, self.padding_features)))
            hidden = torch.cat(inputs, dim=0)
            del inputs
            for block in self.blocks:
                hidden = block(hidden)
            return torch.cat([
                torch.cat([head(rows) for head in self.heads], dim=1)
                for rows in hidden.split(stock_chunk, dim=0)
            ], dim=0)

    return Model()
