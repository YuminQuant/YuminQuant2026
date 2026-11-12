"""Dual-track adversarial factor with training-only latent reference snapshots."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.data.latent_reference import LatentReferenceBank, reference_refresh_due
from yq_ml_alpha.features.transforms import _mad5_zscore_series
from yq_ml_alpha.models.adversarial import critic_objective, frozen_critics, noisy_samples
from yq_ml_alpha.models.market_style_model import MarketStyleAlphaModel, full_date_ic_loss
from yq_ml_alpha.models.mlp_model import _cpu_state_dict, _device, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io


def _params(raw):
    integers = {"input_size", "sequence_length", "d_model", "attention_heads", "transformer_layers", "ffn_size",
                "head_hidden", "latent_size", "memory_capacity", "memory_refill_epochs", "critic_steps",
                "epochs", "patience", "batch_size"}
    scalars = {"dropout", "norm_eps", "eps", "lr", "critic_lr", "weight_decay", "grad_clip", "min_delta",
               "top_fraction", "output_weight", "latent_weight", "ic_weight", "gp_weight", "output_noise_std",
               "latent_noise_std", "leaky_slope"}
    keys = integers | scalars | {"seed", "device", "gradient_checkpointing", "memory_warmup_epochs",
                                "critic_hidden_layers", "generator_betas", "critic_betas"}
    if set(raw) != keys:
        raise ValueError(f"DTAN parameters: missing={sorted(keys - set(raw))}, unknown={sorted(set(raw) - keys)}")
    params = dict(raw)
    for key in integers | {"memory_warmup_epochs"}:
        value = params[key]
        if type(value) is not int or value < (0 if key == "memory_warmup_epochs" else 1):
            raise ValueError(f"DTAN {key} must be an integer within its valid range")
    for key in scalars:
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"DTAN {key} must be finite and nonnegative")
    if any(params[k] == 0 for k in ("norm_eps", "eps", "lr", "critic_lr", "grad_clip", "gp_weight", "ic_weight")):
        raise ValueError("DTAN numerical scales, rates, gradient penalty and IC weight must be positive")
    if not 0 < params["top_fraction"] <= 1 or not 0 <= params["dropout"] < 1:
        raise ValueError("DTAN requires top_fraction in (0,1] and dropout in [0,1)")
    if params["d_model"] % params["attention_heads"] or params["memory_capacity"] < 2:
        raise ValueError("DTAN requires divisible attention heads and a reference capacity >= 2")
    if not isinstance(params["gradient_checkpointing"], bool) or type(params["seed"]) is not int:
        raise ValueError("DTAN requires boolean checkpointing and an integer seed")
    layers = params["critic_hidden_layers"]
    if not isinstance(layers, list) or not layers or any(type(n) is not int or n < 1 for n in layers):
        raise ValueError("DTAN critic_hidden_layers must list positive widths")
    for key in ("generator_betas", "critic_betas"):
        betas = params[key]
        if len(betas) != 2 or any(not np.isfinite(x) or not 0 <= x < 1 for x in betas):
            raise ValueError(f"invalid Adam {key}")
        params[key] = [float(x) for x in betas]
    return params


def _target(frame, label):
    values = pd.to_numeric(frame[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _mad5_zscore_series(values).to_numpy(dtype="float32")


def _step(torch, loss, network, optimizer, grad_clip):
    if not torch.isfinite(loss):
        raise ValueError("nonfinite DTAN loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    if not torch.isfinite(norm):
        raise ValueError("nonfinite DTAN gradients")
    optimizer.step()


class DTANAlphaModel(MarketStyleAlphaModel):
    def __init__(self):
        super().__init__()
        self.bank_history = []
        self.critic_state_dicts = {}

    def _forward_values(self, torch, values, device, *, training, latent):
        from torch.utils.checkpoint import checkpoint
        outputs, latents = [], []
        function = self.model if latent else self.model.score
        for start in range(0, len(values), self.params["batch_size"]):
            inputs = torch.from_numpy(values[start:start + self.params["batch_size"]]).to(device)
            if training and self.params["gradient_checkpointing"]:
                result = checkpoint(function, inputs, use_reentrant=False, preserve_rng_state=True)
            else:
                result = function(inputs)
            prediction, embedding = result if latent else (result, None)
            outputs.append(prediction)
            if latent:
                latents.append(embedding)
        return torch.cat(outputs), torch.cat(latents) if latent else None

    def _scores(self, torch, bundle, date, frame, device, *, training):
        return self._forward_values(torch, self._inputs(bundle, date, frame), device,
                                    training=training, latent=False)[0]

    def _fill_bank(self, torch, train, context, bank, epoch, device):
        # Candidate selection reads ONLY training labels already admitted by the
        # public maturity/split policy. Validation never enters this interface.
        groups = {int(date): frame for date, frame in train.frame.groupby("trade_date", sort=True)}
        candidates = []
        for date, frame in groups.items():
            target = _target(frame, context.label_column)
            keep = np.flatnonzero(np.isfinite(target))
            if len(keep) < 2 or target[keep].std() <= self.params["eps"]:
                continue
            count = max(1, int(np.ceil(len(keep) * self.params["top_fraction"])))
            selected = keep[np.argsort(target[keep], kind="stable")[-count:]]
            candidates.extend((date, int(index)) for index in selected)
        if len(candidates) < 2:
            raise ValueError("DTAN needs at least two labeled training references")
        rng = np.random.default_rng(self.params["seed"] + epoch)
        chosen = rng.choice(len(candidates), size=min(bank.capacity, len(candidates)), replace=False)
        selected_by_date = {}
        for index in chosen:
            date, row = candidates[index]
            selected_by_date.setdefault(date, []).append(row)
        was_training = self.model.training
        self.model.eval()
        references = []
        try:
            with torch.no_grad():
                for date, indices in sorted(selected_by_date.items()):
                    values = self._inputs(train, date, groups[date])
                    _, latent = self._forward_values(torch, values[indices], device, training=False, latent=True)
                    references.append(latent.cpu().numpy())
        finally:
            self.model.train(was_training)
        bank.replace(np.concatenate(references), version=epoch)
        self.bank_history.append(dict(epoch=epoch, candidates=len(candidates), entries=len(bank.values),
            encoded_dates=len(selected_by_date), first_source_date=min(selected_by_date), last_source_date=max(selected_by_date),
            bytes=int(bank.values.nbytes), source="training_true_return_top_fraction", refresh="uniform_snapshot_current_generator_eval"))
        print(f"dtan reference_bank epoch={epoch} candidates={len(candidates)} entries={len(bank.values)} "
              f"dates={len(selected_by_date)} version={bank.version}", flush=True)

    def _critic_update(self, torch, critic, optimizer, real, fake, noise):
        critic.train()
        optimizer.zero_grad(set_to_none=True)
        real, fake = noisy_samples(torch, real, noise), noisy_samples(torch, fake, noise)
        loss, gap, penalty = critic_objective(torch, critic, real, fake, gp_weight=self.params["gp_weight"])
        _step(torch, loss, critic, optimizer, self.params["grad_clip"])
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), float(gap.detach()), float(penalty.detach())

    def _train_epoch(self, torch, train, context, device, generator_optimizer, critics, critic_optimizers, bank, rng):
        self.model.train()
        active = bank.version is not None
        totals = []
        for date, frame in train.frame.groupby("trade_date", sort=True):
            target = _target(frame, context.label_column)
            keep = np.isfinite(target)
            if keep.sum() < 2 or target[keep].std() <= self.params["eps"]:
                continue
            values = self._inputs(train, date, frame)
            labels = torch.as_tensor(target[keep], device=device).reshape(-1, 1)
            mask = torch.as_tensor(keep, device=device)
            generator_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake, latent = self._forward_values(torch, values, device, training=False, latent=active)
            current_indices = None
            real_latent = None
            if active:
                size = min(len(values), len(bank.values))
                current_indices = torch.as_tensor(rng.choice(len(values), size=size, replace=False), device=device)
                real_latent = torch.from_numpy(bank.sample(size, rng)).to(device)
            out_stats, latent_stats = [], []
            for _ in range(self.params["critic_steps"]):
                out_stats.append(self._critic_update(torch, critics[0], critic_optimizers[0], labels,
                    fake[mask, None], self.params["output_noise_std"]))
                if active:
                    latent_stats.append(self._critic_update(torch, critics[1], critic_optimizers[1], real_latent,
                        latent[current_indices], self.params["latent_noise_std"]))
            del fake, latent, real_latent
            with frozen_critics(critics):
                prediction, latent = self._forward_values(torch, values, device, training=True, latent=active)
                ic_loss = full_date_ic_loss(prediction[mask], labels.flatten(), self.params["eps"])
                output_loss = -critics[0](noisy_samples(torch, prediction[mask, None], self.params["output_noise_std"])).mean()
                latent_loss = prediction.new_zeros(())
                if active:
                    latent_loss = -critics[1](noisy_samples(torch, latent[current_indices], self.params["latent_noise_std"])).mean()
                loss = (self.params["ic_weight"] * ic_loss + self.params["output_weight"] * output_loss
                        + self.params["latent_weight"] * latent_loss)
                _step(torch, loss, self.model, generator_optimizer, self.params["grad_clip"])
            totals.append([float(v.detach()) for v in (loss, ic_loss, output_loss, latent_loss)]
                          + np.mean(out_stats, axis=0).tolist()
                          + (np.mean(latent_stats, axis=0).tolist() if active else [np.nan] * 3))
            del prediction, latent, loss, ic_loss, output_loss, latent_loss
        if not totals:
            raise ValueError("DTAN has no usable training date cross-sections")
        result = np.asarray(totals, dtype=np.float64)
        values = np.mean(result[:, :7], axis=0).tolist()
        values += np.mean(result[:, 7:], axis=0).tolist() if active else [None] * 3
        names = ["train_loss", "ic_loss", "output_adversarial_loss", "latent_adversarial_loss",
                 "critic_output_loss", "output_wasserstein_gap", "output_gradient_penalty",
                 "critic_latent_loss", "latent_wasserstein_gap", "latent_gradient_penalty"]
        return dict(zip(names, values)), len(totals)

    def _validate(self, torch, valid, context, device):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for date, frame in valid.frame.groupby("trade_date", sort=True):
                target = _target(frame, context.label_column)
                keep = np.isfinite(target)
                if keep.sum() < 2 or target[keep].std() <= self.params["eps"]:
                    continue
                prediction = self._scores(torch, valid, date, frame, device, training=False)
                labels = torch.as_tensor(target[keep], device=device)
                loss = full_date_ic_loss(prediction[torch.as_tensor(keep, device=device)], labels, self.params["eps"])
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite DTAN validation loss on {date}")
                losses.append(float(loss))
        if not losses:
            raise ValueError("DTAN validation has no usable labeled cross-sections")
        return float(np.mean(losses))

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        if train.frame.empty or valid.frame.empty or context.model_search:
            raise ValueError("DTAN requires separate nonempty train/valid bundles and no model.search")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("DTAN labels must match the configured target")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("DTAN training and validation must be chronological and disjoint")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if self.input_dim != self.params["input_size"]:
            raise ValueError("DTAN input size must match market/style columns")
        source = self._source(train)
        self._source(valid)
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_generator(nn, self.params).to(device)
        critics = [build_critic(nn, width, self.params).to(device) for width in (1, self.params["latent_size"])]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
            betas=tuple(self.params["generator_betas"]), weight_decay=self.params["weight_decay"])
        critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=self.params["critic_lr"],
            betas=tuple(self.params["critic_betas"]), weight_decay=self.params["weight_decay"]) for critic in critics]
        bank = LatentReferenceBank(self.params["memory_capacity"], self.params["latent_size"])
        rng = np.random.default_rng(self.params["seed"])
        best, best_state, stale, best_epoch = np.inf, None, 0, 0
        self.loss_history, self.bank_history, self.critic_state_dicts = [], [], {}
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_started, before = perf_counter(), source.cache_info()
            if reference_refresh_due(epoch, self.params["memory_warmup_epochs"], self.params["memory_refill_epochs"]):
                self._fill_bank(torch, train, context, bank, epoch, device)
            metrics, dates = self._train_epoch(torch, train, context, device, optimizer, critics, critic_optimizers, bank, rng)
            valid_loss = self._validate(torch, valid, context, device)
            improved = valid_loss < best - self.params["min_delta"]
            if improved:
                best, best_epoch, stale, best_state = valid_loss, epoch, 0, _cpu_state_dict(self.model)
                self.critic_state_dicts = {name: _cpu_state_dict(critic) for name, critic in zip(("output", "latent"), critics)}
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch, **metrics,
                train_ic=1 - metrics["ic_loss"], valid_loss=valid_loss, valid_ic=1 - valid_loss, best_loss=best,
                is_best=improved, stale_epochs=stale, train_date_count=dates, reference_entries=len(bank.values),
                reference_version=bank.version, epoch_elapsed_seconds=perf_counter() - epoch_started,
                elapsed_seconds=perf_counter() - started))
            if context.diagnostics.get("enabled", False) and context.diagnostics.get("print_epoch", False):
                print(f"dtan epoch={epoch} dates={dates} generator_loss={metrics['train_loss']:.6f} "
                      f"train_ic={1 - metrics['ic_loss']:.6f} valid_ic={1 - valid_loss:.6f} "
                      f"critic_out={metrics['critic_output_loss']:.6f} critic_lat={metrics['critic_latent_loss']} "
                      f"adv_out={metrics['output_adversarial_loss']:.6f} adv_lat={metrics['latent_adversarial_loss']:.6f} "
                      f"gp_out={metrics['output_gradient_penalty']:.6f} gp_lat={metrics['latent_gradient_penalty']} "
                      f"bank={len(bank.values)} version={bank.version} stale={stale}/{self.params['patience']} "
                      f"epoch_elapsed_s={perf_counter() - epoch_started:.2f}", flush=True)
                source.log_cache(f"dtan_epoch_{epoch}", before=before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(window_id=context.artifact_dir.name, params=self.params, feature_columns=self.feature_columns,
            best_epoch=best_epoch, best_valid_ic=1 - best, epochs_run=len(self.loss_history),
            train_rows=len(train.frame), valid_rows=len(valid.frame), bank_history=self.bank_history,
            reference_scope="training_true_return_top_fraction_only", critic_normalization="first_hidden_layer_layernorm",
            label_transform="date_mad5_zscore_observed_labels_only", date_selection="all_fixed_sampled_dates_every_epoch",
            prediction="generator_clean_score_only_no_bank_no_critics_no_labels")

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted DTAN")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="dtan", params=self.params, feature_columns=self.feature_columns,
            state_dict=_cpu_state_dict(self.model), critic_state_dicts=self.critic_state_dicts,
            loss_history=self.loss_history, bank_history=self.bank_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "dtan":
            raise ValueError("unsupported DTAN checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        if model.input_dim != model.params["input_size"]:
            raise ValueError("DTAN checkpoint feature dimension mismatch")
        model.model = build_generator(nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        model.bank_history, model.critic_state_dicts = saved["bank_history"], saved["critic_state_dicts"]
        return model


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
