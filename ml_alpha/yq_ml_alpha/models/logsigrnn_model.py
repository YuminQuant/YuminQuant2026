"""Logsig-RNN factor using derived multivariate logsignatures."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel, full_date_ic_loss
from yq_ml_alpha.models.torch_support import _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    integers = {"input_size", "sequence_length", "hidden_size", "num_layers", "batch_size"}
    scalars = {"dropout", "lr", "weight_decay", "eps", "norm_eps", "bn_momentum"}
    if "epochs_per_draw" in raw:
        integers.add("epochs_per_draw")
        if "date_draws" in raw:
            integers.add("date_draws")
    else:
        integers.update(("epochs", "patience"))
        scalars.add("min_delta")
    keys = integers | scalars | {"seed", "device", "gradient_checkpointing"}
    if set(raw) != keys:
        raise ValueError(f"Logsig-RNN parameters: missing={sorted(keys - set(raw))}, unknown={sorted(set(raw) - keys)}")
    params = dict(raw)
    if any(type(params[k]) is not int or params[k] < 1 for k in integers):
        raise ValueError("Logsig-RNN dimensions, batch_size and epoch limits must be positive integers")
    for key in scalars:
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"invalid Logsig-RNN {key}")
    if (any(params[k] <= 0 for k in ("eps", "norm_eps", "lr")) or params["dropout"] >= 1
            or not 0 < params["bn_momentum"] <= 1):
        raise ValueError("invalid Logsig-RNN numerical scales, dropout or BatchNorm momentum")
    if type(params["seed"]) is not int or not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("Logsig-RNN requires integer seed and boolean gradient_checkpointing")
    if params["input_size"] != 728 or params["sequence_length"] != 21:
        raise ValueError("Logsig-RNN derived protocol is fixed at 21 x 728")
    if "epochs_per_draw" in params and params["batch_size"] < 3:
        raise ValueError("Logsig-RNN mini-batches need batch_size of at least three")
    return params


def build_network(nn, params):
    import torch
    from torch.utils.checkpoint import checkpoint

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = nn.BatchNorm1d(params["input_size"], eps=params["norm_eps"], momentum=params["bn_momentum"])
            self.recurrent = nn.LSTM(params["input_size"], params["hidden_size"], params["num_layers"],
                batch_first=True, dropout=params["dropout"] if params["num_layers"] > 1 else 0.0)
            self.output_norm = nn.BatchNorm1d(params["hidden_size"], eps=params["norm_eps"], momentum=params["bn_momentum"])
            self.head = nn.Linear(params["hidden_size"], 1)

        def encode(self, values):
            return self.recurrent(values)[0][:, -1]

        def forward(self, values):
            # BatchNorm uses the supplied batch. Encoding chunks do not change its statistics.
            normalized = self.input_norm(values.transpose(1, 2)).transpose(1, 2)
            blocks = []
            for start in range(0, len(values), params["batch_size"]):
                block = normalized[start:start + params["batch_size"]]
                if self.training and params["gradient_checkpointing"]:
                    blocks.append(checkpoint(self.encode, block, use_reentrant=False, preserve_rng_state=True))
                else:
                    blocks.append(self.encode(block))
            encoded = torch.cat(blocks)
            return self.head(self.output_norm(encoded)).flatten()

    return Network()


class LogsigRNNAlphaModel(DailySequenceAlphaModel):
    def _source(self, bundle):
        from yq_ml_alpha.features.derived_logsigrnn import LogsigRnnHistory
        source = super()._source(bundle)
        if not isinstance(source, LogsigRnnHistory):
            raise ValueError("Logsig-RNN requires the derived seven-channel logsignature protocol")
        return source

    def _scores(self, torch, bundle, date, frame, device, *, training):
        source = self._source(bundle)
        if sorted(frame.ts_code.tolist()) != source.symbols(int(date)):
            raise ValueError("Logsig-RNN BatchNorm requires the complete eligible date cross-section")
        values = self._inputs(bundle, date, frame)
        return self.model(torch.from_numpy(values).to(device))

    def _epoch(self, torch, bundle, context, device, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        losses = []
        with torch.set_grad_enabled(training):
            for date, frame in bundle.frame.groupby("trade_date", sort=True):
                target = pd.to_numeric(frame[context.label_column], errors="coerce").to_numpy(dtype="float32")
                keep = np.isfinite(target)
                if keep.sum() < 2 or target[keep].std() <= self.params["eps"]:
                    continue
                if training:
                    optimizer.zero_grad(set_to_none=True)
                prediction = self._scores(torch, bundle, date, frame, device, training=training)
                loss = full_date_ic_loss(prediction[torch.as_tensor(keep, device=device)],
                                        torch.as_tensor(target[keep], device=device), self.params["eps"])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite Logsig-RNN loss on {date}")
                if training:
                    loss.backward()
                    optimizer.step()
                losses.append(float(loss.detach()))
        if not losses:
            raise ValueError("Logsig-RNN has no usable labeled date cross-sections")
        return float(np.mean(losses)), len(losses)

    def _fit_cross_sections(self, torch, train, valid, context, device, optimizer, source):
        groups = {}
        for date, indices in train.frame.groupby("trade_date", sort=True).indices.items():
            target = pd.to_numeric(train.frame.iloc[indices][context.label_column],
                                   errors="coerce").to_numpy(dtype="float32")
            keep = np.isfinite(target)
            if keep.sum() >= 2 and target[keep].std() > self.params["eps"]:
                groups[date] = indices[keep], target[keep]
        if not groups:
            raise ValueError("Logsig-RNN has no eligible training dates")

        rng = np.random.default_rng(self.params["seed"])
        dates = list(groups)
        date_draws = self.params.get("date_draws")
        progress_key = "draw" if date_draws is not None else "section"
        if date_draws is not None:
            dates = rng.choice(dates, size=date_draws, replace=True).tolist()
        batch_size = self.params["batch_size"]
        started = perf_counter()
        self.model.train()
        for draw, date in enumerate(dates, start=1):
            draw_started, before = perf_counter(), source.cache_info()
            indices, target = groups[date]
            values = self._inputs(train, date, train.frame.iloc[indices])
            boundaries = list(range(0, len(target), batch_size))
            if len(boundaries) > 1 and len(target) - boundaries[-1] == 1:
                # Move one stock from the previous batch so both batches support BN and IC.
                boundaries[-1] -= 1
            boundaries.append(len(target))
            losses = []
            for _ in range(self.params["epochs_per_draw"]):
                order = rng.permutation(len(target))
                for start, end in zip(boundaries, boundaries[1:]):
                    batch = order[start:end]
                    label = target[batch]
                    if label.std() <= self.params["eps"]:
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(torch.from_numpy(values[batch]).to(device))
                    loss = full_date_ic_loss(prediction, torch.from_numpy(label).to(device), self.params["eps"]) - 1
                    if not torch.isfinite(loss):
                        raise ValueError(f"nonfinite Logsig-RNN mini-batch loss on {date}")
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
            if not losses:
                raise ValueError(f"no usable Logsig-RNN mini-batches on {date}")
            train_loss = float(np.mean(losses))
            self.loss_history.append(dict(window_id=context.artifact_dir.name, trade_date=int(date),
                train_loss=train_loss, train_ic=-train_loss, optimizer_steps=len(losses),
                draw_elapsed_seconds=perf_counter() - draw_started, elapsed_seconds=perf_counter() - started,
                **{progress_key: draw}))
            if context.diagnostics.get("enabled", False) and context.diagnostics.get("print_epoch", False):
                print(f"logsigrnn window={context.artifact_dir.name} {progress_key}={draw}/{len(dates)} date={date} "
                      f"stocks={len(target)} steps={len(losses)} train_ic={-train_loss:.6f} "
                      f"draw_elapsed_s={perf_counter() - draw_started:.2f}", flush=True)
                source.log_cache(f"logsigrnn_{progress_key}_{draw}", before=before)
            del values

        valid_loss, valid_dates = self._epoch(torch, valid, context, device)
        self.loss_history[-1].update(valid_loss=valid_loss - 1, valid_ic=1 - valid_loss)
        self.model.to("cpu").eval()
        self.model_info = dict(window_id=context.artifact_dir.name, params=self.params, feature_columns=self.feature_columns,
            sections_run=len(dates), train_date_count=len(groups), distinct_dates_trained=len(set(dates)),
            valid_date_count=valid_dates, final_valid_ic=1 - valid_loss, train_rows=len(train.frame),
            valid_rows=len(valid.frame), date_selection="uniform_with_replacement" if date_draws is not None else "fixed_chronological",
            loss="negative_minibatch_pearson_ic",
            batch_norm="training_minibatch", prediction="saved_batchnorm_statistics_no_labels")
        print(f"logsigrnn final valid_dates={valid_dates} valid_ic={1 - valid_loss:.6f}", flush=True)

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        if train.frame.empty or valid.frame.empty or context.model_search:
            raise ValueError("Logsig-RNN requires separate nonempty train/valid bundles and no model.search")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("Logsig-RNN labels must match the configured target")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("Logsig-RNN training and validation must be chronological and disjoint")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if self.input_dim != self.params["input_size"]:
            raise ValueError("Logsig-RNN input dimension mismatch")
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_network(nn, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        self.loss_history = []
        if "epochs_per_draw" in self.params:
            self._fit_cross_sections(torch, train, valid, context, device, optimizer, source)
            return
        best, best_state, best_epoch, stale = np.inf, None, 0, 0
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_started, before = perf_counter(), source.cache_info()
            train_loss, train_dates = self._epoch(torch, train, context, device, optimizer)
            valid_loss, valid_dates = self._epoch(torch, valid, context, device)
            improved = valid_loss < best - self.params["min_delta"]
            if improved:
                best, best_state, best_epoch, stale = valid_loss, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch,
                train_loss=train_loss, valid_loss=valid_loss, train_ic=1 - train_loss, valid_ic=1 - valid_loss,
                best_loss=best, is_best=improved, stale_epochs=stale, train_date_count=train_dates,
                valid_date_count=valid_dates, epoch_elapsed_seconds=perf_counter() - epoch_started,
                elapsed_seconds=perf_counter() - started))
            if context.diagnostics.get("enabled", False) and context.diagnostics.get("print_epoch", False):
                print(f"logsigrnn window={context.artifact_dir.name} epoch={epoch} dates={train_dates}/{valid_dates} "
                      f"train_ic={1 - train_loss:.6f} valid_ic={1 - valid_loss:.6f} "
                      f"stale={stale}/{self.params['patience']} epoch_elapsed_s={perf_counter() - epoch_started:.2f}", flush=True)
                source.log_cache(f"logsigrnn_epoch_{epoch}", before=before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(window_id=context.artifact_dir.name, params=self.params, feature_columns=self.feature_columns,
            best_epoch=best_epoch, best_valid_ic=1 - best, epochs_run=len(self.loss_history),
            train_rows=len(train.frame), valid_rows=len(valid.frame), train_date_count=train_dates, valid_date_count=valid_dates,
            date_selection="all_fixed_sampled_dates_every_epoch", loss="full_labeled_date_pearson_ic",
            batch_norm="full_observed_cross_section_input_NT_output_N", prediction="saved_batchnorm_statistics_no_labels")

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted Logsig-RNN")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="logsigrnn", params=self.params,
            feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
            loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "logsigrnn":
            raise ValueError("unsupported Logsig-RNN checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        if model.input_dim != model.params["input_size"]:
            raise ValueError("Logsig-RNN checkpoint feature dimension mismatch")
        model.model = build_network(nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
