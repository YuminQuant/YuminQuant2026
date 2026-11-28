"""DFQ-FactorVAE-pro with dynamic feature masking and date-conditioned training."""
from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _mad5_zscore_series
from yq_ml_alpha.models.torch_support import (
    TorchAlphaModel,
    _cpu_state_dict,
    _device,
    _set_seed,
    _torch_modules,
)
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(
        sequence_length=30, hidden_size=32, num_layers=1,
        num_portfolio=100, num_factor=48, epochs=100, patience=10,
        average_epochs=5, batch_size=5000, grad_clip=5.0,
        lr=0.0003, weight_decay=0.0, kl_weight=1.0, min_delta=1e-8,
        min_std=1e-6, seed=42, device="auto", gradient_checkpointing=True,
        feature_reconstruction_weight=1.0, prior_alignment_weight=1.0,
    )
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown FactorVAE-pro parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "hidden_size", "num_layers", "num_portfolio",
                "num_factor", "epochs", "patience", "average_epochs", "batch_size"):
        value = params[key]
        params[key] = int(value)
        if params[key] <= 0 or params[key] != value:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lr", "weight_decay", "kl_weight", "min_delta", "min_std",
                "grad_clip", "feature_reconstruction_weight", "prior_alignment_weight"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if params["lr"] == 0 or params["min_std"] == 0 or params["grad_clip"] == 0:
        raise ValueError("lr, min_std and grad_clip must be positive")
    if not isinstance(params["gradient_checkpointing"], bool):
        raise ValueError("gradient_checkpointing must be boolean")
    params["seed"] = int(params["seed"])
    return params


def _gaussian_kl(source, destination):
    source_mean, source_std = source
    destination_mean, destination_std = destination
    return (destination_std.log() - source_std.log()
            + 0.5 * ((source_std / destination_std).square()
                     + ((source_mean - destination_mean) / destination_std).square() - 1)).sum()


def build_factor_vae_pro(nn, input_size, environment_size, params):
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    hidden = params["hidden_size"]
    factors = params["num_factor"]
    portfolios = params["num_portfolio"]
    min_std = params["min_std"]

    def positive_std(value):
        return functional.softplus(value) + min_std

    class FeatureMask(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Sequential(nn.Linear(input_size, hidden), nn.LeakyReLU())
            self.gru = nn.GRU(hidden, hidden, params["num_layers"], batch_first=True)
            self.logit = nn.Linear(hidden, input_size)

        def forward(self, values):
            sequence, _ = self.gru(self.projection(values))
            # Softmax over [drop, keep] depends only on the logit difference.
            logits = self.logit(sequence)
            keep = (logits > 0).to(values.dtype)
            if self.training:
                probability = logits.sigmoid()
                keep = keep + (probability - probability.detach())
            return values * keep

    class FactorBranch(nn.Module):
        def __init__(self, conditioned=False):
            super().__init__()
            self.feature = nn.Sequential(nn.LayerNorm(input_size), nn.Linear(input_size, hidden))
            self.environment = nn.Linear(environment_size, hidden, bias=False) if conditioned else None
            self.gru = nn.GRU(hidden, hidden, params["num_layers"], batch_first=True)
            self.portfolio = nn.Linear(hidden, portfolios)
            self.posterior_mean = nn.Linear(portfolios, factors)
            self.posterior_std = nn.Linear(portfolios, factors)
            self.keys = nn.Linear(hidden, factors * hidden)
            self.values = nn.Linear(hidden, factors * hidden)
            self.queries = nn.Parameter(torch.randn(factors, hidden) / hidden ** 0.5)
            self.prior_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
            self.prior_mean = nn.Linear(hidden, 1)
            self.prior_std = nn.Linear(hidden, 1)

        def encode(self, masked, environment=None):
            values = self.feature(masked)
            if self.environment is not None:
                # The sample-date code is shared by all stocks and time steps.
                values = values + self.environment(environment)
            return self.gru(functional.leaky_relu(values))[1][-1]

        def posterior(self, encoded, target, labeled_rows):
            weights = self.portfolio(encoded.index_select(0, labeled_rows)).softmax(dim=0)
            returns = weights.T @ target
            return self.posterior_mean(returns), positive_std(self.posterior_std(returns))

        def prior(self, encoded):
            keys = self.keys(encoded).reshape(-1, factors, hidden)
            queries = functional.normalize(self.queries, dim=-1, eps=1e-8)
            scores = torch.einsum("nfh,fh->nf", keys, queries)
            scores = (scores / keys.norm(dim=-1).clamp_min(1e-8)).clamp_min(0)
            weights = scores / scores.sum(dim=0).clamp_min(1e-8)
            values = self.values(encoded).reshape(-1, factors, hidden)
            summaries = torch.einsum("nf,nfh->fh", weights, values)
            state = self.prior_hidden(summaries)
            return self.prior_mean(state).squeeze(-1), positive_std(self.prior_std(state).squeeze(-1))

    class ReturnDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha_hidden = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())
            self.alpha_mean = nn.Linear(hidden, 1)
            self.alpha_std = nn.Linear(hidden, 1)
            self.beta = nn.Linear(hidden, factors)

        def forward(self, encoded, factor, *, sample=False):
            factor_mean, factor_std = factor
            alpha = self.alpha_hidden(encoded)
            beta = self.beta(encoded)
            mean = self.alpha_mean(alpha).squeeze(-1) + beta @ factor_mean
            if not sample:
                return mean
            variance = positive_std(self.alpha_std(alpha).squeeze(-1)).square()
            variance = variance + beta.square() @ factor_std.square()
            std = variance.clamp_min(min_std ** 2).sqrt()
            return mean + std * torch.randn_like(std)

    class FactorVAEPro(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask = FeatureMask()
            self.main = FactorBranch()
            self.environment = FactorBranch(conditioned=True)
            self.decoder = ReturnDecoder()
            self.reconstructor = nn.Sequential(
                nn.Linear(input_size, hidden), nn.LeakyReLU(),
                nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, input_size),
            )

        def encode(self, host_values, device):
            masked = self.mask(host_values.to(device))
            return self.main.encode(masked)

        def encode_training(self, host_values, device, environment):
            def run(batch):
                values = batch.to(device)
                masked = self.mask(values)
                squared_error = functional.mse_loss(self.reconstructor(masked), values, reduction="sum")
                return self.main.encode(masked), self.environment.encode(masked, environment), squared_error

            if params["gradient_checkpointing"]:
                # Retain the host batch and the small outputs; rebuild activations on backward.
                return checkpoint(run, host_values, use_reentrant=False, preserve_rng_state=False)
            return run(host_values)

        def forward(self, encoded):
            return self.decoder(encoded, self.main.prior(encoded))

        def training_loss(self, main, conditioned, target, labeled_rows, reconstruction):
            main_prior = self.main.prior(main)
            env_prior = self.environment.prior(conditioned)
            main_post = self.main.posterior(main, target, labeled_rows)
            env_post = self.environment.posterior(conditioned, target, labeled_rows)
            main_return = self.decoder(main.index_select(0, labeled_rows), main_post, sample=True)
            env_return = self.decoder(conditioned.index_select(0, labeled_rows), env_post, sample=True)
            main_mse = functional.mse_loss(main_return, target)
            env_mse = functional.mse_loss(env_return, target)
            main_kl = _gaussian_kl(main_post, main_prior)
            env_kl = _gaussian_kl(env_post, env_prior)
            alignment = _gaussian_kl(env_prior, main_prior)
            loss = (main_mse + env_mse + params["kl_weight"] * (main_kl + env_kl)
                    + params["feature_reconstruction_weight"] * reconstruction
                    + params["prior_alignment_weight"] * alignment)
            return torch.stack((loss, reconstruction, main_mse, main_kl, env_mse, env_kl, alignment))

    return FactorVAEPro()


def _observed_targets(frame, label):
    targets = {}
    for date, rows in frame.groupby("trade_date", sort=True):
        raw = pd.to_numeric(rows[label], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = _mad5_zscore_series(raw).to_numpy(dtype=np.float32)
        labeled = np.flatnonzero(np.isfinite(values))
        observed = values[labeled]
        if len(labeled) >= 2 and observed.std() > 1e-8:
            targets[date] = labeled, observed
    return targets


class FactorVAEProAlphaModel(TorchAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or list(bundle.feature_columns) != self.feature_columns
                or source.columns != self.feature_columns
                or source.length != self.params["sequence_length"]):
            raise ValueError("FactorVAE-pro requires matching factor sequence columns and length")
        return source

    def _batches(self, date, rows, source, torch):
        symbols = rows.ts_code.tolist()
        if sorted(symbols) != source.symbols(date):
            raise ValueError(f"FactorVAE-pro requires the complete stock cross-section on {date}")
        size = self.params["batch_size"]
        for start in range(0, len(symbols), size):
            yield torch.from_numpy(source.get(date, symbols[start:start + size]))

    def _score(self, date, rows, source, device, torch):
        encoded = [self.model.encode(host, device) for host in self._batches(date, rows, source, torch)]
        return self.model(torch.cat(encoded)).cpu().numpy()

    def _evaluate(self, groups, targets, source, device, torch):
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for date, (labeled, target) in targets.items():
                score = self._score(int(date), groups.get_group(date), source, device, torch)[labeled]
                if not np.isfinite(score).all():
                    raise ValueError(f"nonfinite FactorVAE-pro validation output on {date}")
                metrics.append(float(np.corrcoef(score, target)[0, 1]) if score.std() > 1e-8 else 0.0)
        return float(np.mean(metrics))

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if context.model_search or not self.input_dim:
            raise ValueError("FactorVAE-pro needs input factors and does not use model.search")
        train_source, valid_source = self._source(train), self._source(valid)
        if train.frame.empty or valid.frame.empty or train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("FactorVAE-pro requires nonempty chronological, disjoint train and validation sets")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("FactorVAE-pro train and validation labels must match the context")
        train_groups = train.frame.groupby("trade_date", sort=True)
        valid_groups = valid.frame.groupby("trade_date", sort=True)
        train_targets = _observed_targets(train.frame, context.label_column)
        valid_targets = _observed_targets(valid.frame, context.label_column)
        if not train_targets or not valid_targets:
            raise ValueError("FactorVAE-pro requires usable labeled train and validation cross-sections")
        self.environment_years = sorted({int(date) // 10000 for date in train_groups.groups})
        year_index = {year: i for i, year in enumerate(self.environment_years)}
        environment_size = len(year_index) + 12
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        date_codes = np.zeros((len(train_targets), environment_size), dtype=np.float32)
        for i, date in enumerate(train_targets):
            year, month_day = divmod(int(date), 10000)
            date_codes[i, year_index[year]] = 1
            date_codes[i, len(year_index) + month_day // 100 - 1] = 1
        dates = dict(zip(train_targets, torch.from_numpy(date_codes).to(device).unbind()))

        self.model = build_factor_vae_pro(nn, self.input_dim, environment_size, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, stale, best_epoch = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        train_source.log_cache("fit_start")
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_started = perf_counter()
            cache_before = train_source.cache_info()
            self.model.train()
            totals = torch.zeros(7, device=device, dtype=torch.float64)
            for date, (labeled, observed) in train_targets.items():
                rows = train_groups.get_group(date)
                optimizer.zero_grad(set_to_none=True)
                main, conditioned, errors = [], [], []
                for host in self._batches(int(date), rows, train_source, torch):
                    stock, environment, error = self.model.encode_training(host, device, dates[date])
                    main.append(stock)
                    conditioned.append(environment)
                    errors.append(error)
                reconstruction = torch.stack(errors).sum() / (len(rows) * self.params["sequence_length"] * self.input_dim)
                metrics = self.model.training_loss(
                    torch.cat(main), torch.cat(conditioned),
                    torch.as_tensor(observed, device=device), torch.as_tensor(labeled, device=device),
                    reconstruction,
                )
                metrics[0].backward()
                norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params["grad_clip"])
                if not torch.isfinite(torch.stack((metrics[0].detach(), norm))).all():
                    raise ValueError(f"nonfinite FactorVAE-pro loss or gradients on {date}")
                optimizer.step()
                totals.add_(metrics.detach())
                del main, conditioned, errors, stock, environment, error, reconstruction, metrics

            live = _cpu_state_dict(self.model)
            recent.append(live)
            averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
            self.model.load_state_dict(averaged)
            metric = self._evaluate(valid_groups, valid_targets, valid_source, device, torch)
            self.model.load_state_dict(live)
            improved = metric > best + self.params["min_delta"]
            if improved:
                best, best_state, stale, best_epoch = metric, averaged, 0, epoch
            else:
                stale += 1
            used = len(train_targets)
            loss, recon, main_mse, main_kl, env_mse, env_kl, alignment = (totals / used).tolist()
            self.loss_history.append(dict(
                window_id=context.artifact_dir.name, epoch=epoch, train_loss=loss,
                feature_reconstruction_mse=recon, main_return_mse=main_mse, main_kl=main_kl,
                environment_return_mse=env_mse, environment_kl=env_kl, prior_alignment_kl=alignment,
                valid_ic=metric, best_ic=best, is_best=improved, stale_epochs=stale,
                train_date_count=used, averaged_epochs=len(recent), elapsed_seconds=perf_counter() - started,
            ))
            if context.diagnostics.get("print_epoch", False):
                print(f"factorvae-pro epoch={epoch} dates={used} loss={loss:.6f} "
                      f"feature_mse={recon:.6f} main_mse={main_mse:.6f} main_kl={main_kl:.6f} "
                      f"env_mse={env_mse:.6f} env_kl={env_kl:.6f} alignment_kl={alignment:.6f} "
                      f"valid_ic={metric:.6f} average={len(recent)} stale={stale} "
                      f"epoch_elapsed_s={perf_counter() - epoch_started:.2f}", flush=True)
                train_source.log_cache(f"epoch_{epoch}", before=cache_before)
            if stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.to("cpu").eval()
        self.model_info = dict(
            best_epoch=best_epoch, best_valid_ic=best, feature_count=self.input_dim,
            feature_columns=self.feature_columns, params=self.params, device=str(device),
            environment_years=self.environment_years, train_rows=len(train.frame),
            valid_rows=len(valid.frame), epochs_run=len(self.loss_history),
            label_transform="date_mad5_zscore_observed_labels_only",
            environment="sample_year_and_month_one_hot_training_only",
            prior_alignment="kl_environment_prior_to_main_prior",
            prediction="hard_feature_mask_and_main_prior_return_mean",
            validation="date_mean_pearson_ic",
        )

    def predict_bundle(self, bundle, context):
        torch, _ = _torch_modules()
        if self.model is None:
            raise RuntimeError("FactorVAE-pro is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("FactorVAE-pro prediction feature order changed")
        source = self._source(bundle)
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    scores = self._score(int(date), rows, source, device, torch)
                    if not np.isfinite(scores).all():
                        raise ValueError(f"nonfinite FactorVAE-pro predictions on {date}")
                    result[rows.index.to_numpy()] = scores
            return pd.Series(result, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.to("cpu")

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted FactorVAE-pro")
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(
            version=1, model_kind="factor_vae_pro", params=self.params,
            feature_columns=self.feature_columns, environment_years=self.environment_years,
            state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history,
            model_info=self.model_info,
        ), path)

    @classmethod
    def load(cls, path):
        _, nn = _torch_modules()
        saved = artifact_io.torch_load(path, map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "factor_vae_pro":
            raise ValueError("unsupported FactorVAE-pro checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = saved["feature_columns"]
        model.input_dim = len(model.feature_columns)
        model.environment_years = saved["environment_years"]
        model.model = build_factor_vae_pro(nn, model.input_dim, len(model.environment_years) + 12, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
