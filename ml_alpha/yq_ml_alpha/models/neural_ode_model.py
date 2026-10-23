"""DFQ variational Neural ODE alpha on historical formal-factor sequences."""
from __future__ import annotations

from yq_ml_alpha.output import artifact_io

from collections import deque
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.bar_gru_model import _release_torch_memory
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _cpu_state_dict, _device, _set_seed, _torch_modules


def _params(raw):
    params = dict(sequence_length=30, hidden_size=128, num_layers=2, latent_size=64,
                  dynamics_size=128, ode_substeps=1, checkpoint_ode=True,
                  reconstruction_weight=1.0, kl_weight=0.001, epochs=200,
                  patience=20, average_epochs=5, dates_per_epoch=100, batch_size=3000,
                  lr=0.0001, weight_decay=0.0, min_delta=1e-8, seed=42, device="auto")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown NeuralODE parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("sequence_length", "hidden_size", "num_layers", "latent_size", "dynamics_size",
                "ode_substeps", "epochs", "patience", "average_epochs", "dates_per_epoch", "batch_size"):
        value = params[key]
        params[key] = int(value)
        if isinstance(value, bool) or params[key] <= 0 or params[key] != value:
            raise ValueError(f"{key} must be a positive integer")
    if params["sequence_length"] < 2:
        raise ValueError("sequence_length must contain at least two observations")
    if not isinstance(params["checkpoint_ode"], bool):
        raise ValueError("checkpoint_ode must be boolean")
    for key in ("lr", "weight_decay", "min_delta", "reconstruction_weight", "kl_weight"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if params["lr"] == 0:
        raise ValueError("lr must be positive")
    params["seed"] = int(params["seed"])
    return params


def build_neural_ode(nn, input_size, params):
    """Variational initial state, latent ODE, Gaussian observations and return MLP.

    The observation grid covers the historical window only. Integration uses
    fixed RK4 steps and discrete autograd, not an adaptive or adjoint solver.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    class NeuralODE(nn.Module):
        def __init__(self):
            super().__init__()
            hidden, latent = params["hidden_size"], params["latent_size"]
            width = params["dynamics_size"]
            self.encoder = nn.GRU(input_size, hidden, num_layers=params["num_layers"], batch_first=True)
            self.posterior_mean = nn.Linear(hidden, latent)
            self.posterior_logvar = nn.Linear(hidden, latent)
            self.dynamics = nn.Sequential(
                nn.Linear(latent + 1, width), nn.Tanh(),
                nn.Linear(width, width), nn.Tanh(), nn.Linear(width, latent),
            )
            self.observation_mean = nn.Linear(latent, input_size)
            self.observation_logvar = nn.Linear(latent, input_size)
            # Read the reconstructed terminal observation, as in the report.
            self.head = nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(), nn.Linear(hidden, 1))

        def _drift(self, state, time):
            clock = state.new_full((len(state), 1), time)
            return self.dynamics(torch.cat((state, clock), dim=1))

        def _advance(self, state, observation_index):
            substeps = params["ode_substeps"]
            intervals = params["sequence_length"] - 1
            step = 1.0 / (intervals * substeps)
            # Derive times from integer indices; no accumulating clock drift.
            for substep in range(substeps):
                time = ((observation_index - 1) * substeps + substep) * step
                k1 = self._drift(state, time)
                k2 = self._drift(state + 0.5 * step * k1, time + 0.5 * step)
                k3 = self._drift(state + 0.5 * step * k2, time + 0.5 * step)
                k4 = self._drift(state + step * k3, time + step)
                state = state + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return state

        def _observation_nll(self, state, observed):
            mean = self.observation_mean(state)
            logvar = self.observation_logvar(state).clamp(-8.0, 8.0)
            # Full Gaussian negative log likelihood; average over factor dimensions.
            return (0.5 * (np.log(2.0 * np.pi) + logvar
                           + (observed - mean).square() * torch.exp(-logvar))).mean(dim=1)

        def _step_loss(self, state, observed, observation_index):
            if observation_index:
                state = self._advance(state, observation_index)
            return state, self._observation_nll(state, observed)

        def forward(self, x, with_losses=False):
            if x.ndim != 3 or tuple(x.shape[1:]) != (params["sequence_length"], input_size):
                raise ValueError("NeuralODE expects [stocks, configured sessions, input factors]")
            _, hidden = self.encoder(x)
            mean = self.posterior_mean(hidden[-1])
            logvar = self.posterior_logvar(hidden[-1]).clamp(-10.0, 10.0)
            state = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.training else mean
            reconstruction = state.new_zeros(len(state))
            for index in range(params["sequence_length"]):
                if with_losses:
                    # Passing the index explicitly avoids late-bound loop closures
                    # when checkpoint recomputes RK4 and likelihood during backward.
                    if params["checkpoint_ode"] and torch.is_grad_enabled():
                        state, loss = checkpoint(self._step_loss, state, x[:, index], index)
                    else:
                        state, loss = self._step_loss(state, x[:, index], index)
                    reconstruction = reconstruction + loss
                elif index:
                    state = self._advance(state, index)
            score = self.head(self.observation_mean(state)).squeeze(-1)
            if not with_losses:
                return score
            # KL(q(z0|x) || N(0,I)): sum latent dimensions, one value per stock.
            kl = 0.5 * (mean.square() + torch.exp(logvar) - 1.0 - logvar).sum(dim=1)
            return score, reconstruction / params["sequence_length"], kl

    return NeuralODE()


def _target(rows, column):
    raw = pd.to_numeric(rows[column], errors="coerce")
    return _zscore_series(raw).to_numpy(dtype=np.float32)


class NeuralODEAlphaModel(MLPAlphaModel):
    def __init__(self):
        super().__init__()
        self.feature_columns = []

    def fit(self, train_data, valid_data, context):
        raise ValueError("NeuralODE requires features.type=factor_sequence and fit_bundle")

    def predict(self, data, context):
        raise ValueError("NeuralODE requires predict_bundle with factor histories")

    def _source(self, bundle):
        source = bundle.tensor_source
        if (source is None or list(bundle.feature_columns) != self.feature_columns
                or source.columns != self.feature_columns or source.length != self.params["sequence_length"]
                or getattr(source, "history_transform", None) != "zscore_clip3"):
            raise ValueError("NeuralODE requires matching factor histories with history_transform=zscore_clip3")
        return source

    def _scores(self, date, rows, source, device):
        torch, _ = _torch_modules()
        symbols = rows.ts_code.tolist()
        result = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), self.params["batch_size"]):
            selected = symbols[start:start + self.params["batch_size"]]
            x = torch.from_numpy(source.get(int(date), selected)).to(device)
            result[start:start + len(selected)] = self.model(x).cpu().numpy()
            del x
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite NeuralODE scores on {date}")
        return result

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
                score = self._scores(date, rows.loc[keep], source, device)
                metrics.append(float(np.corrcoef(score, target[keep])[0, 1]) if score.std() > 1e-8 else 0.0)
        if not metrics:
            raise ValueError("NeuralODE validation has no usable labeled cross-sections")
        return float(np.mean(metrics))

    def fit_bundle(self, train, valid, context):
        self.params = _params(context.model_params)
        if context.model_search:
            raise ValueError("NeuralODE does not use model.search")
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        source = self._source(train)
        self._source(valid)
        if not self.input_dim or train.frame.empty or valid.frame.empty:
            raise ValueError("NeuralODE requires factors and separate nonempty train/validation partitions")
        if train.label_column != context.label_column or valid.label_column != context.label_column:
            raise ValueError("NeuralODE label columns must match context")
        if train.frame.trade_date.max() >= valid.frame.trade_date.min():
            raise ValueError("NeuralODE train/validation dates must be chronological and disjoint")
        # Precompute date-local labels, not expanded feature sequence tensors.
        groups = []
        for date, rows in train.frame.groupby("trade_date", sort=True):
            y = _target(rows, context.label_column)
            keep = np.isfinite(y)
            if keep.sum() >= 2 and y[keep].std() > 1e-8:
                groups.append((int(date), rows.ts_code.to_numpy()[keep].tolist(), y[keep]))
        if not groups:
            raise ValueError("NeuralODE training has no usable labeled cross-sections")
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = build_neural_ode(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        rng = np.random.RandomState(self.params["seed"])
        recent = deque(maxlen=self.params["average_epochs"])
        best, best_state, stale, best_epoch = -np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                selected = np.sort(rng.choice(len(groups), min(len(groups), self.params["dates_per_epoch"]), replace=False))
                total = np.zeros(3, dtype=np.float64)
                for index in selected:
                    date, symbols, target = groups[index]
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(symbols), self.params["batch_size"]):
                        codes = symbols[start:start + self.params["batch_size"]]
                        x = torch.from_numpy(source.get(date, codes)).to(device)
                        y = torch.from_numpy(target[start:start + len(codes)]).to(device)
                        score, reconstruction, kl = self.model(x, with_losses=True)
                        # Normalize every term by the full date, not by the stock chunk.
                        mse_loss = (score - y).square().sum() / len(symbols)
                        reconstruction_loss = reconstruction.sum() / len(symbols)
                        kl_loss = kl.sum() / len(symbols)
                        loss = (mse_loss + self.params["reconstruction_weight"] * reconstruction_loss
                                + self.params["kl_weight"] * kl_loss)
                        if not bool(torch.isfinite(loss).item()):
                            raise ValueError(f"nonfinite NeuralODE loss on {date}")
                        loss.backward()
                        total += np.array([mse_loss.item(), reconstruction_loss.item(), kl_loss.item()])
                        del x, y, score, reconstruction, kl, loss, mse_loss, reconstruction_loss, kl_loss
                    finite_gradients = torch.stack([torch.isfinite(p.grad).all()
                                                    for p in self.model.parameters() if p.grad is not None])
                    if not bool(finite_gradients.all().item()):
                        raise ValueError(f"nonfinite NeuralODE gradient on {date}")
                    optimizer.step()
                live = _cpu_state_dict(self.model)
                recent.append(live)
                averaged = {name: sum(state[name] for state in recent) / len(recent) for name in live}
                self.model.load_state_dict(averaged)
                try:
                    metric = self._evaluate(valid, device)
                finally:
                    self.model.load_state_dict(live)
                if not np.isfinite(metric):
                    raise ValueError("nonfinite NeuralODE validation IC")
                improved = metric > best + self.params["min_delta"]
                if improved:
                    best, best_state, stale, best_epoch = metric, averaged, 0, epoch
                else:
                    stale += 1
                mse, reconstruction, kl = (total / len(selected)).tolist()
                objective = mse + self.params["reconstruction_weight"] * reconstruction + self.params["kl_weight"] * kl
                self.loss_history.append(dict(
                    window_id=context.artifact_dir.name, epoch=epoch, train_loss=objective,
                    train_mse=mse, train_reconstruction_nll=reconstruction, train_kl=kl,
                    valid_ic=metric, best_ic=best, is_best=improved, stale_epochs=stale,
                    sampled_dates=len(selected), averaged_epochs=len(recent),
                    elapsed_seconds=perf_counter() - started,
                ))
                if context.diagnostics.get("print_epoch", False):
                    print(f"neural_ode window={context.artifact_dir.name} epoch={epoch} "
                          f"loss={objective:.6g} mse={mse:.6g} reconstruction={reconstruction:.6g} "
                          f"kl={kl:.6g} valid_ic={metric:.6g} "
                          f"dates={len(selected)} average={len(recent)} stale={stale}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id, window_id=context.artifact_dir.name,
            feature_columns=self.feature_columns, params=self.params, device=str(device),
            best_epoch=best_epoch, best_valid_ic=best, epochs_run=len(self.loss_history),
            train_rows=len(train.frame), valid_rows=len(valid.frame), train_usable_dates=len(groups),
            loss="return_mse_plus_weighted_gaussian_nll_and_standard_normal_kl",
            input_transform="daily_zscore_clip3_fill0", label_transform="daily_return_zscore",
            prediction="posterior_mean_initial_state_terminal_reconstruction_mlp_no_labels",
            solver="fixed_rk4_normalized_history_grid_discrete_autograd",
            posterior_logvar_bounds=[-10.0, 10.0], observation_logvar_bounds=[-8.0, 8.0],
            reconstruction_reduction="mean_time_and_factors_then_mean_stocks",
            kl_reduction="sum_latent_then_mean_stocks",
            checkpoint="best_validation_trailing_epoch_parameter_mean",
        )

    def predict_bundle(self, bundle, context):
        if self.model is None:
            raise RuntimeError("NeuralODE is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("NeuralODE prediction feature order changed")
        torch, _ = _torch_modules()
        source = self._source(bundle)
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        frame = bundle.frame.reset_index(drop=True)
        result = np.empty(len(frame), dtype=np.float32)
        try:
            with torch.no_grad():
                for date, rows in frame.groupby("trade_date", sort=True):
                    result[rows.index.to_numpy()] = self._scores(date, rows, source, device)
            return pd.Series(result, index=bundle.frame.index)
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted NeuralODE")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="dfq_neural_ode", params=self.params,
                        feature_columns=self.feature_columns, state_dict=_cpu_state_dict(self.model),
                        loss_history=self.loss_history, model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(path, map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "dfq_neural_ode":
            raise ValueError("unsupported NeuralODE checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_neural_ode(nn, model.input_dim, model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
