"""DFQ-FactorVAE with local factor sequences and a label-free prediction path."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _mad5_zscore_series
from yq_ml_alpha.models.mlp_model import (
    MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules,
)


def _params(raw):
    params = dict(
        sequence_length=30, hidden_size=32, num_layers=1,
        num_portfolio=100, num_factor=48, epochs=200, patience=20,
        average_epochs=5, dates_per_epoch=100, batch_size=3000,
        lr=0.0003, weight_decay=0.0, kl_weight=1.0, min_delta=1e-8,
        min_std=1e-6, seed=42, device="auto", gradient_checkpointing=True,
    )
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown FactorVAE parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "hidden_size", "num_layers", "num_portfolio",
                "num_factor", "epochs", "patience", "average_epochs", "dates_per_epoch", "batch_size"):
        value = params[key]
        params[key] = int(value)
        if params[key] <= 0 or params[key] != value:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "weight_decay", "kl_weight", "min_delta", "min_std"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if params["lr"] == 0 or params["min_std"] == 0:
        raise ValueError("lr and min_std must be positive")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    params["seed"] = int(params["seed"])
    return params


def build_factor_vae(nn, input_size, params):
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    hidden = params["hidden_size"]
    factors = params["num_factor"]

    class GlobalAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.key = nn.Linear(hidden, hidden)
            self.value = nn.Linear(hidden, hidden)
            self.query = nn.Parameter(torch.randn(hidden) / hidden ** 0.5)

        def forward(self, encoded):
            keys = functional.normalize(self.key(encoded), dim=-1, eps=1e-8)
            query = functional.normalize(self.query, dim=0, eps=1e-8)
            # Report figure 11: positive cosine / sum of positive cosines.
            # A head with no positive match contributes zero, never NaN.
            weights = (keys @ query).clamp_min(0)
            weights = weights / weights.sum().clamp_min(1e-8)
            return weights @ self.value(encoded)

    class FactorVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature = nn.Sequential(
                nn.LayerNorm(input_size), nn.Linear(input_size, hidden), nn.LeakyReLU(),
            )
            self.gru = nn.GRU(hidden, hidden, params["num_layers"], batch_first=True)
            self.portfolio = nn.Linear(hidden, params["num_portfolio"])
            self.posterior_mean = nn.Linear(params["num_portfolio"], factors)
            self.posterior_std = nn.Linear(params["num_portfolio"], factors)
            self.attention = nn.ModuleList([GlobalAttention() for _ in range(factors)])
            self.prior_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
            self.prior_mean = nn.Linear(hidden, 1)
            self.prior_std = nn.Linear(hidden, 1)
            self.alpha_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
            self.alpha_mean = nn.Linear(hidden, 1)
            self.alpha_std = nn.Linear(hidden, 1)
            self.beta = nn.Linear(hidden, factors)

        def positive_std(self, value):
            return functional.softplus(value) + params["min_std"]

        def encode(self, host_values, device):
            def run(values, token):
                values = self.feature(values.to(device))
                return self.gru(values)[1][-1]

            if self.training and params["gradient_checkpointing"]:
                token = torch.zeros((), device=device, requires_grad=True)
                return checkpoint(run, host_values, token)
            return run(host_values, None)

        def posterior(self, encoded, target, mask):
            # Only observed labels enter virtual portfolio returns. Prior
            # attention and feature normalization still see all eligible stocks.
            weights = torch.softmax(self.portfolio(encoded[mask]), dim=0)
            returns = weights.T @ target
            return self.posterior_mean(returns), self.positive_std(self.posterior_std(returns))

        def prior(self, encoded):
            summaries = torch.stack([head(encoded) for head in self.attention])
            hidden_state = self.prior_hidden(summaries)
            return (self.prior_mean(hidden_state).squeeze(-1),
                    self.positive_std(self.prior_std(hidden_state).squeeze(-1)))

        def decode(self, encoded, factor_mean, factor_std):
            alpha = self.alpha_hidden(encoded)
            beta = self.beta(encoded)
            mean = self.alpha_mean(alpha).squeeze(-1) + beta @ factor_mean
            variance = self.positive_std(self.alpha_std(alpha).squeeze(-1)).square()
            variance = variance + beta.square() @ factor_std.square()
            return mean, variance.clamp_min(params["min_std"] ** 2).sqrt()

        def forward(self, encoded):
            # Inference accepts no target and cannot invoke the posterior.
            return self.decode(encoded, *self.prior(encoded))

        def training_loss(self, encoded, target, mask):
            post_mean, post_std = self.posterior(encoded, target, mask)
            prior_mean, prior_std = self.prior(encoded)
            mean, std = self.decode(encoded[mask], post_mean, post_std)
            reconstructed = mean + std * torch.randn_like(std)
            mse = (reconstructed - target).square().mean()
            # KL(q_posterior || p_prior) for diagonal Gaussians: sum over factors.
            kl = (torch.log(prior_std) - torch.log(post_std)
                  + 0.5 * ((post_std / prior_std).square()
                           + ((post_mean - prior_mean) / prior_std).square() - 1)).sum()
            return mse + params["kl_weight"] * kl, mse, kl

    return FactorVAE()


def _target(rows, label):
    """Date-local return winsorization/zscore; absent labels stay absent."""
    raw = pd.to_numeric(rows[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _mad5_zscore_series(raw).to_numpy(dtype=np.float32)


class FactorVAEAlphaModel(MLPAlphaModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("FactorVAE requires features.type=factor_sequence and fit_bundle")

    def predict(self, data, context):
        raise ValueError("FactorVAE requires a complete date factor-sequence bundle")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or list(bundle.feature_columns) != self.feature_columns
                or source.columns != self.feature_columns
                or source.length != self.params["sequence_length"]):
            raise ValueError("FactorVAE requires matching factor sequence columns and length")
        return source

    def _encode(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        if sorted(symbols) != source.symbols(date):
            raise ValueError("FactorVAE requires the complete eligible date cross-section, including unlabeled stocks")
        encoded = []
        for start in range(0, len(symbols), self.params["batch_size"]):
            host = torch.from_numpy(source.get(date, symbols[start:start + self.params["batch_size"]]))
            encoded.append(self.model.encode(host, device))
        return torch.cat(encoded)

    def _evaluate(self, bundle, device):
        torch, _ = _torch_modules()
        source = self._source(bundle)
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for date, rows in bundle.frame.groupby("trade_date", sort=True):
                target = _target(rows, bundle.label_column)
                keep = np.isfinite(target)
                if keep.sum() < 2 or target[keep].std() <= 1e-8:
                    continue
                # Deterministic prior mean; posterior and sampling are training-only.
                score = self.model(self._encode(int(date), rows, source, device))[0].cpu().numpy()[keep]
                if not np.isfinite(score).all():
                    raise ValueError(f"nonfinite FactorVAE validation output on {date}")
                metrics.append(float(np.corrcoef(score, target[keep])[0, 1]) if score.std() > 1e-8 else 0.0)
        if not metrics:
            raise ValueError("FactorVAE validation has no usable labeled cross-sections")
        return float(np.mean(metrics))

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        if context.model_search:
            raise ValueError("FactorVAE does not use model.search")
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if not self.input_dim:
            raise ValueError("FactorVAE requires at least one input factor")
        source = self._source(train)
        self._source(valid)
        if train.frame.empty or valid.frame.empty:
            raise ValueError("FactorVAE requires nonempty separate train and validation data")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("FactorVAE label columns must match the training context")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("FactorVAE train/validation dates must be chronological and disjoint")
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_factor_vae(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        grouped = train.frame.groupby("trade_date", sort=True)
        days = np.asarray(sorted(grouped.groups))
        rng = np.random.RandomState(self.params["seed"])
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, stale, best_epoch = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        for epoch in range(1, self.params["epochs"] + 1):
            self.model.train()
            selected = np.sort(rng.choice(days, min(len(days), self.params["dates_per_epoch"]), replace=False))
            totals = np.zeros(3, dtype=np.float64)
            used = 0
            for date in selected:
                rows = grouped.get_group(date)
                y = _target(rows, context.label_column)
                keep = np.isfinite(y)
                if keep.sum() < 2 or y[keep].std() <= 1e-8:
                    continue
                optimizer.zero_grad(set_to_none=True)
                encoded = self._encode(int(date), rows, source, device)
                target = torch.as_tensor(y[keep], device=device)
                mask = torch.as_tensor(keep, device=device)
                loss, mse, kl = self.model.training_loss(encoded, target, mask)
                if not torch.isfinite(loss):
                    raise ValueError(f"nonfinite FactorVAE loss on {date}")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                if not torch.isfinite(norm):
                    raise ValueError(f"nonfinite FactorVAE gradients on {date}")
                optimizer.step()
                totals += [float(value.detach()) for value in (loss, mse, kl)]
                used += 1
                del encoded, loss, mse, kl
            if not used:
                raise ValueError("FactorVAE epoch has no usable labeled cross-section")
            live = _cpu_state_dict(self.model)
            recent.append(live)
            averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
            self.model.load_state_dict(averaged)
            try:
                metric = self._evaluate(valid, device)
            finally:
                self.model.load_state_dict(live)
            improved = metric > best + self.params["min_delta"]
            if improved:
                best, best_state, stale, best_epoch = metric, averaged, 0, epoch
            else:
                stale += 1
            train_loss, reconstruction_mse, kl_loss = (totals / used).tolist()
            self.loss_history.append(dict(
                window_id=context.artifact_dir.name, epoch=epoch, train_loss=train_loss,
                reconstruction_mse=reconstruction_mse, kl_loss=kl_loss,
                valid_ic=metric, best_ic=best, is_best=improved, stale_epochs=stale,
                sampled_dates=used, averaged_epochs=len(recent),
                elapsed_seconds=perf_counter() - started,
            ))
            if context.diagnostics.get("print_epoch", False):
                print(f"factorvae epoch={epoch} dates={used} loss={train_loss:.6f} "
                      f"mse={reconstruction_mse:.6f} kl={kl_loss:.6f} valid_ic={metric:.6f} "
                      f"average={len(recent)} stale={stale}", flush=True)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(
            best_epoch=best_epoch, best_valid_ic=best, feature_count=self.input_dim,
            feature_columns=self.feature_columns, params=self.params, device=str(device),
            train_rows=len(train.frame), valid_rows=len(valid.frame), epochs_run=len(self.loss_history),
            loss="sampled_reconstruction_mse_plus_kl_posterior_prior_sum",
            label_transform="date_mad5_zscore_observed_labels_only",
            prediction="prior_return_mean_no_labels_no_sampling",
            attention="positive_cosine_normalized_over_complete_date",
            validation="date_mean_pearson_ic",
            checkpoint="best_validation_trailing_epoch_parameter_mean",
        )

    def predict_bundle(self, bundle, context):
        torch, _ = _torch_modules()
        if self.model is None:
            raise RuntimeError("FactorVAE is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("FactorVAE prediction feature order changed")
        source = self._source(bundle)
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    scores = self.model(self._encode(int(date), rows, source, device))[0].cpu().numpy()
                    if not np.isfinite(scores).all():
                        raise ValueError(f"nonfinite FactorVAE predictions on {date}")
                    result[rows.index.to_numpy()] = scores
            return pd.Series(result, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")

    def save(self, path):
        torch, _ = _torch_modules()
        if self.model is None:
            raise RuntimeError("cannot save an unfitted FactorVAE")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(version=1, model_kind="factor_vae", params=self.params,
                        feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
                        loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = torch.load(path, map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "factor_vae":
            raise ValueError("unsupported FactorVAE checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = saved["feature_columns"]
        model.input_dim = len(model.feature_columns)
        model.model = build_factor_vae(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
