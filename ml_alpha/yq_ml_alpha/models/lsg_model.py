"""Full LSG factor: memory spectrum, delay geometry, recurrence and state gate."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.lsg_network import build_lsg
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
