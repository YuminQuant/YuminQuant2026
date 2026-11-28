"""ABCM1 jointly learns an alpha score and twelve daily risk exposures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import scale_market_window, zscore_observed
from yq_ml_alpha.models.daily_sequence_model import DailySequenceAlphaModel
from yq_ml_alpha.models.torch_support import _cpu_state_dict, _device, _diagnostics, _set_seed, _torch_modules
from yq_ml_alpha.output import artifact_io

if TYPE_CHECKING:
    from torch import Tensor


def _params(raw):
    params = dict(raw)
    for key in ("input_size", "sequence_length", "hidden_size", "gru_layers", "beta_size",
                "batch_size", "query_block_size", "turnover_lag", "epochs", "patience"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"ABCM1 {key} must be a positive integer")
    for key in ("lr", "grad_clip", "ridge", "eps", "risk_weight", "orthogonal_weight", "turnover_weight"):
        if not np.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"ABCM1 {key} must be positive and finite")
    if params["input_size"] != 15 or params["beta_size"] != 12:
        raise ValueError("ABCM1 requires 15 daily inputs and 12 learned beta outputs")
    if type(params["gradient_checkpointing"]) is not bool:
        raise ValueError("ABCM1 gradient_checkpointing must be boolean")
    return params


def _network(torch, nn, params):
    from torch.utils.checkpoint import checkpoint

    class ABCM(nn.Module):
        def __init__(self):
            super().__init__()
            width, beta_size = params["hidden_size"], params["beta_size"]
            self.gru = nn.GRU(params["input_size"], width, num_layers=params["gru_layers"], batch_first=True)
            self.alpha = nn.Linear(width, 1)
            self.risk_input = nn.Linear(width, beta_size)
            self.query = nn.Linear(beta_size, beta_size)
            self.key = nn.Linear(beta_size, beta_size)
            self.value = nn.Linear(beta_size, beta_size)
            self.risk_output = nn.Linear(beta_size, beta_size)

        def _encode(self, values):
            return self.gru(values)[1][-1]

        def encode(self, values):
            recompute = self.training and torch.is_grad_enabled() and params["gradient_checkpointing"]
            return torch.cat([
                checkpoint(self._encode, block, use_reentrant=False) if recompute else self._encode(block)
                for block in values.split(params["batch_size"])
            ])

        def beta(self, hidden):
            risk = self.risk_input(hidden)
            query, keys, values = self.query(risk), self.key(risk), self.value(risk)
            nodes = torch.arange(len(risk), device=risk.device)
            recompute = self.training and torch.is_grad_enabled() and params["gradient_checkpointing"]

            def aggregate(block, all_keys, all_values, query_nodes):
                # Mask the diagonal after ReLU so softmax assigns self messages zero weight.
                logits = (block @ all_keys.T).relu()
                logits = logits.masked_fill(query_nodes[:, None] == nodes[None, :], -torch.inf)
                return logits.softmax(dim=1) @ all_values

            if len(risk) < 2:
                return self.risk_output(risk)
            messages = []
            for block, indices in zip(query.split(params["query_block_size"]),
                                      nodes.split(params["query_block_size"])):
                messages.append(checkpoint(aggregate, block, keys, values, indices, use_reentrant=False)
                                if recompute else aggregate(block, keys, values, indices))
            return self.risk_output(risk + torch.cat(messages, dim=0))

    return ABCM()


def _unit_centered(values, eps):
    centered = values - values.mean(dim=0, keepdim=True)
    return centered / centered.norm(dim=0, keepdim=True).clamp_min(eps)


def _unexplained_return(torch, factors, returns, ridge, eps):
    # Normalize exposure columns for a stable small ridge system. The report's
    # projection has no intercept and preserves the raw return target.
    x = factors / factors.norm(dim=0, keepdim=True).clamp_min(eps)
    identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    coeff = torch.linalg.solve_ex(x.T @ x + ridge * identity, x.T @ returns).result
    return (returns - x @ coeff).square().sum() / returns.square().sum().clamp_min(eps)


@dataclass
class Section:
    values: Tensor
    alpha_rows: Tensor
    alpha_target: Tensor
    beta_rows: Tensor
    beta_target: Tensor
    previous: tuple[Tensor, Tensor, Tensor] | None


class ABCM1AlphaModel(DailySequenceAlphaModel):
    def _inputs(self, bundle, date, frame):
        values = super()._inputs(bundle, date, frame)
        scale_market_window(values[:, :, :6], price_columns=5, eps=self.params["eps"])
        # Each (history date, field) is a column. No label enters these statistics.
        return zscore_observed(values.reshape(len(values), -1), mad5=True).reshape(values.shape)

    def _scores(self, torch, bundle, date, frame, device, *, training):
        values = torch.from_numpy(self._inputs(bundle, date, frame)).to(device)
        return self.model.alpha(self.model.encode(values)).squeeze(1)

    def _prepare(self, torch, bundle, labels, *, training):
        source = self._source(bundle)
        prior_sections = {}
        sections = []
        for date, frame in bundle.frame.groupby("trade_date", sort=True):
            alpha = frame[labels[0]].to_numpy(dtype=np.float32, na_value=np.nan)
            alpha_rows = np.flatnonzero(np.isfinite(alpha))
            if len(alpha_rows) < 2:
                continue
            beta = frame[labels[1]].to_numpy(dtype=np.float32, na_value=np.nan)
            beta_rows = np.flatnonzero(np.isfinite(beta))
            if training and (len(beta_rows) <= self.params["beta_size"] + 1
                             or np.linalg.norm(beta[beta_rows]) <= self.params["eps"]):
                continue
            values = torch.from_numpy(self._inputs(bundle, int(date), frame))
            previous = None
            if training:
                position = source.positions[int(date)]
                symbols = pd.Index(frame.ts_code)
                prior = prior_sections.get(position - self.params["turnover_lag"])
                if prior is not None:
                    prior_symbols, prior_values = prior
                    aligned = prior_symbols.get_indexer(symbols)
                    rows = np.flatnonzero(aligned >= 0)
                    if len(rows) >= 2:
                        previous = (prior_values, torch.from_numpy(rows), torch.from_numpy(aligned[rows]))
                prior_sections[position] = (symbols, values)
            sections.append(Section(values, torch.from_numpy(alpha_rows), torch.from_numpy(alpha[alpha_rows]),
                                    torch.from_numpy(beta_rows), torch.from_numpy(beta[beta_rows]), previous))
        if not sections:
            raise ValueError("ABCM1 split has no usable labeled cross-sections")
        return sections

    def _beta_losses(self, torch, hidden, alpha, sample, device):
        params = self.params
        risk = self.model.beta(hidden)
        exposures = torch.cat((alpha[:, None], risk), dim=1)
        residual = _unexplained_return(torch, exposures.index_select(0, sample.beta_rows.to(device)),
                                       sample.beta_target.to(device), params["ridge"], params["eps"])
        unit = _unit_centered(exposures, params["eps"])
        corr = unit.T @ unit
        orthogonal = (corr - corr.diag().diag()).square().mean()
        turnover = alpha.new_zeros(())
        if sample.previous is not None:
            prior_values, current_rows, prior_rows = sample.previous
            # Use the current weights for both dates. The earlier exposure is a
            # fixed reference for this step, so no second backward graph is kept.
            with torch.no_grad():
                prior = self.model.beta(self.model.encode(prior_values.to(device)))
                prior = _unit_centered(prior.index_select(0, prior_rows.to(device)), params["eps"])
            current = _unit_centered(risk.index_select(0, current_rows.to(device)), params["eps"])
            turnover = 1 - (current * prior).sum(dim=0).mean()
        return residual, orthogonal, turnover

    def _epoch(self, torch, sections, device, optimizer=None):
        training = optimizer is not None
        self.model.train(training)
        totals = torch.zeros(5, device=device)
        with torch.set_grad_enabled(training):
            for sample in sections:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                hidden = self.model.encode(sample.values.to(device))
                alpha = self.model.alpha(hidden).squeeze(1)
                mse = (alpha.index_select(0, sample.alpha_rows.to(device)) - sample.alpha_target.to(device)).square().mean()
                if not training:
                    totals[1] += mse
                    continue
                residual, orthogonal, turnover = self._beta_losses(torch, hidden, alpha, sample, device)
                params = self.params
                loss = (mse + params["risk_weight"] * residual + params["orthogonal_weight"] * orthogonal
                        + params["turnover_weight"] * turnover)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), params["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
                totals += torch.stack((loss, mse, residual, orthogonal, turnover)).detach()
        return (totals / len(sections)).tolist()

    def fit_bundle(self, train, valid, context):
        torch, nn = _torch_modules()
        params = self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        labels = context.label_columns
        if (train.frame.empty or valid.frame.empty or train.frame.trade_date.max() >= valid.frame.trade_date.min()
                or self.input_dim != params["input_size"] or context.model_search
                or labels != ["future_vwap_return_10d", "future_vwap_return_20d"]):
            raise ValueError("ABCM1 requires chronological splits, 10/20-day labels and fixed daily inputs")
        diagnostics = _diagnostics(context)
        started = perf_counter()
        training = self._prepare(torch, train, labels, training=True)
        validation = self._prepare(torch, valid, labels, training=False)
        if diagnostics["enabled"]:
            print(f"abcm1 prepared train_dates={len(training)} valid_dates={len(validation)} "
                  f"elapsed_s={perf_counter() - started:.2f}", flush=True)
        _set_seed(torch, params["seed"])
        device = _device(torch, params["device"])
        self.model = _network(torch, nn, params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        best, state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        for epoch in range(1, params["epochs"] + 1):
            order = np.random.default_rng(params["seed"] + epoch).permutation(len(training))
            train_loss, train_mse, residual, orthogonal, turnover = self._epoch(
                torch, [training[i] for i in order], device, optimizer)
            _, valid_mse, _, _, _ = self._epoch(torch, validation, device)
            if not np.isfinite([train_loss, valid_mse]).all():
                raise ValueError(f"nonfinite ABCM1 metrics at epoch {epoch}")
            improved = valid_mse < best
            if improved:
                best, state, best_epoch, stale = valid_mse, _cpu_state_dict(self.model), epoch, 0
            else:
                stale += 1
            self.loss_history.append(dict(window_id=context.artifact_dir.name, epoch=epoch, train_loss=train_loss,
                train_mse=train_mse, train_unexplained_return=residual, train_orthogonal=orthogonal,
                train_turnover=turnover, valid_loss=valid_mse, best_loss=best, is_best=improved,
                stale_epochs=stale, elapsed_seconds=perf_counter() - started))
            if diagnostics["print_epoch"]:
                print(f"abcm1 epoch={epoch} train_loss={train_loss:.6g} train_mse={train_mse:.6g} "
                      f"unexplained_return={residual:.6g} orthogonal={orthogonal:.6g} turnover={turnover:.6g} "
                      f"valid_mse={valid_mse:.6g} stale={stale}/{params['patience']}", flush=True)
            if stale >= params["patience"]:
                break
        self.model.load_state_dict(state)
        self.model.to("cpu").eval()
        self.model_info = dict(params=params, feature_columns=self.feature_columns, labels=labels,
                               best_epoch=best_epoch, best_valid_mse=best, epochs_run=len(self.loss_history),
                               beta_outputs=params["beta_size"], train_dates=len(training), valid_dates=len(validation),
                               selection="datewise_alpha_validation_mse")

    def save(self, path):
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(kind="abcm1", version=1, params=self.params, feature_columns=self.feature_columns,
            state_dict=_cpu_state_dict(self.model), model_info=self.model_info, loss_history=self.loss_history), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("kind") != "abcm1" or saved.get("version") != 1:
            raise ValueError("unsupported ABCM1 checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = _network(torch, nn, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.model_info, model.loss_history = saved["model_info"], saved["loss_history"]
        return model
