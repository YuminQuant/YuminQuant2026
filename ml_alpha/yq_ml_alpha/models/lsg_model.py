"""Full LSG factor: memory spectrum, delay geometry, recurrence and state gate."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.market_style_model import MarketStyleAlphaModel, full_date_ic_loss
from yq_ml_alpha.models.mlp_model import _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    integers = {"input_size", "sequence_length", "temporal_steps", "delay", "embedding_dim", "conv_channels",
                "kernel_size", "gru_hidden", "gru_layers", "d_model", "attention_heads", "transformer_layers",
                "ffn_size", "head_hidden", "nearest_k", "epochs", "batch_size", "patience"}
    scalars = {"dropout", "mu_init", "sigma_init", "lambda_init", "gate_init", "eps", "lr", "weight_decay",
               "grad_clip", "min_delta"}
    keys = integers | scalars | {"seed", "device", "spectrum_orders", "geometry_input",
                                "distance_normalization", "gradient_checkpointing"}
    if set(raw) != keys:
        raise ValueError(f"LSG parameter keys: missing={sorted(keys - set(raw))}, unknown={sorted(set(raw) - keys)}")
    params = dict(raw)
    for key in integers:
        value = params[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"LSG {key} must be a positive integer")
        params[key] = int(value)
    if type(params["seed"]) is not int:
        raise ValueError("LSG seed must be an integer")
    for key in scalars:
        params[key] = float(params[key])
        if not np.isfinite(params[key]):
            raise ValueError(f"LSG {key} must be finite")
    if not 0 <= params["dropout"] < 1 or not 0 < params["mu_init"] < 1:
        raise ValueError("LSG requires dropout in [0,1) and mu_init in (0,1)")
    if any(params[key] <= 0 for key in ("sigma_init", "eps", "lr", "grad_clip")):
        raise ValueError("LSG scales, learning rate and gradient clip must be positive")
    if any(params[key] < 0 for key in ("lambda_init", "weight_decay", "min_delta")):
        raise ValueError("LSG recurrence weight, decay and minimum improvement must be nonnegative")
    if params["d_model"] % params["attention_heads"]:
        raise ValueError("LSG model width must be divisible by attention heads")
    if params["sequence_length"] != params["temporal_steps"] + (params["embedding_dim"] - 1) * params["delay"]:
        raise ValueError("LSG history must cover every delay coordinate of the temporal window")
    if params["kernel_size"] > params["embedding_dim"] or params["nearest_k"] >= params["temporal_steps"]:
        raise ValueError("LSG kernel and nearest-neighbor count exceed observed coordinates")
    if params["geometry_input"] not in {"spectrum", "raw"}:
        raise ValueError("LSG geometry_input must be spectrum or raw")
    if params["distance_normalization"] not in {"mean", "none"}:
        raise ValueError("LSG distance_normalization must be mean or none")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("LSG gradient_checkpointing must be boolean")
    from yq_ml_alpha.models.lsg_network import fractional_difference_matrices
    fractional_difference_matrices(params["temporal_steps"], params["spectrum_orders"])
    params["spectrum_orders"] = [float(d) for d in params["spectrum_orders"]]
    return params


class LSGAlphaModel(MarketStyleAlphaModel):
    def _scores(self, torch, bundle, date, frame, device, *, training):
        from torch.utils.checkpoint import checkpoint
        values = self._inputs(bundle, date, frame)
        scores = []
        for start in range(0, len(values), self.params["batch_size"]):
            inputs = torch.from_numpy(values[start:start + self.params["batch_size"]]).to(device)
            if training and self.params["gradient_checkpointing"]:
                scores.append(checkpoint(self.model, inputs, use_reentrant=False, preserve_rng_state=True))
            else:
                scores.append(self.model(inputs))
        return torch.cat(scores)

    def _epoch(self, torch, bundle, context, device, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        losses = []
        for date, frame in bundle.frame.groupby("trade_date", sort=True):
            target = pd.to_numeric(frame[context.label_column], errors="coerce").to_numpy(dtype="float32")
            keep = np.isfinite(target)
            if keep.sum() < 2 or target[keep].std() <= self.params["eps"]:
                continue
            with torch.set_grad_enabled(training):
                if training:
                    optimizer.zero_grad(set_to_none=True)
                scores = self._scores(torch, bundle, date, frame, device, training=training)
                if not torch.isfinite(scores).all():
                    raise ValueError(f"nonfinite LSG scores on {date}")
                observed = scores[torch.as_tensor(keep, device=device)]
                labels = torch.as_tensor(target[keep], device=device)
                loss = full_date_ic_loss(observed, labels, self.params["eps"])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite LSG IC loss on {date}")
                if training:
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"])
                    if not torch.isfinite(norm):
                        raise ValueError(f"nonfinite LSG gradients on {date}")
                    optimizer.step()
                    self.model.project_constraints()
                losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("LSG has no usable labeled date cross-sections")
        return float(np.mean(losses)), len(losses)

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        if train.frame.empty or valid.frame.empty or context.model_search:
            raise ValueError("LSG requires separate nonempty train/valid bundles and no hyperparameter search")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("LSG train and validation labels must match the configured target")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("LSG train and validation dates must be chronological and disjoint")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if self.input_dim != self.params["input_size"]:
            raise ValueError("LSG input_size must match market/style feature columns")
        source = self._source(train)
        self._source(valid)
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_lsg(nn, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        best, best_state, stale, best_epoch = np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_started = perf_counter()
            before = source.cache_info()
            train_loss, train_dates = self._epoch(torch, train, context, device, optimizer)
            valid_loss, valid_dates = self._epoch(torch, valid, context, device)
            improved = valid_loss < best - self.params["min_delta"]
            if improved:
                best, best_state, stale, best_epoch = valid_loss, _cpu_state_dict(self.model), 0, epoch
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch,
                train_loss=train_loss, valid_loss=valid_loss, train_ic=1 - train_loss, valid_ic=1 - valid_loss,
                best_loss=best, is_best=improved, stale_epochs=stale, train_date_count=train_dates,
                valid_date_count=valid_dates, epoch_elapsed_seconds=perf_counter() - epoch_started,
                elapsed_seconds=perf_counter() - started))
            if context.diagnostics.get("enabled", False) and context.diagnostics.get("print_epoch", False):
                print(f"lsg epoch={epoch} dates={train_dates} train_ic={1 - train_loss:.6f} "
                      f"valid_ic={1 - valid_loss:.6f} stale={stale}/{self.params['patience']} "
                      f"epoch_elapsed_s={perf_counter() - epoch_started:.2f}", flush=True)
                source.log_cache(f"lsg_epoch_{epoch}", before=before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(window_id=context.artifact_dir.name, params=self.params,
            feature_columns=self.feature_columns, best_epoch=best_epoch, best_valid_ic=1 - best,
            epochs_run=len(self.loss_history), train_rows=len(train.frame), valid_rows=len(valid.frame),
            loss="one_minus_full_date_pearson_ic", date_selection="all_fixed_sampled_dates_every_epoch",
            normalization=source.normalization, geometry_input=self.params["geometry_input"],
            distance_scope="within_each_stock_observed_history", gate="one_plus_softplus_weight_times_softplus_knn_distance")

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted LSG")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="lsg", params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "lsg":
            raise ValueError("unsupported LSG checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        if model.input_dim != model.params["input_size"]:
            raise ValueError("LSG checkpoint feature dimension mismatch")
        model.model = build_lsg(nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model


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
