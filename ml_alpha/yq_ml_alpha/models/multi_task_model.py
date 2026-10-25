"""Huatai AI 67: hard-sharing MLP with uncertainty or dynamic task weights."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from yq_ml_alpha.features.transforms import _zscore_series
from yq_ml_alpha.models.adaptive_gate_model import _rank_weights
from yq_ml_alpha.models.bar_gru_model import _cpu_state_dict, _device, _release_torch_memory, _set_seed, _torch_modules
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _diagnostics, _features
from yq_ml_alpha.output import artifact_io


def _params(raw):
    params = dict(hidden_size=256, weighting="dwa", negative_slope=0.01,
                  rank_weight_half_life=0.5, dwa_temperature=2.0,
                  lr=0.0001, weight_decay=0.0, epochs=200, patience=20,
                  batch_size=3000, seed=42, min_delta=1e-12, device="auto")
    unknown = set(raw).difference(params)
    if unknown:
        raise ValueError(f"unknown multi-task parameters: {sorted(unknown)}")
    params.update(raw)
    for key in ("hidden_size", "epochs", "patience", "batch_size", "seed"):
        value = params[key]
        params[key] = int(value)
        if isinstance(value, bool) or params[key] != value or (key != "seed" and params[key] <= 0):
            raise ValueError(f"invalid integer multi-task parameter {key}")
    for key in ("negative_slope", "rank_weight_half_life", "dwa_temperature", "lr", "weight_decay", "min_delta"):
        params[key] = float(params[key])
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if any(params[key] <= 0 for key in ("rank_weight_half_life", "dwa_temperature", "lr")):
        raise ValueError("rank_weight_half_life, dwa_temperature and lr must be positive")
    if params["weighting"] not in {"uw", "dwa"}:
        raise ValueError("multi-task weighting must be uw or dwa")
    return params


def _label_columns(context):
    columns = list(context.label_columns or [context.label_column])
    if (len(columns) < 2 or len(set(columns)) != len(columns)
            or columns[0] != context.label_column):
        raise ValueError("multi-task training requires distinct ordered primary and auxiliary labels")
    return columns


def build_multi_task(nn, input_size, task_count, params):
    import torch

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            width = params["hidden_size"]
            slope = params["negative_slope"]
            self.shared = nn.Sequential(
                nn.Linear(input_size, width), nn.LeakyReLU(slope),
                nn.Linear(width, width), nn.LeakyReLU(slope),
            )
            self.heads = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(slope), nn.Linear(width, 1))
                for _ in range(task_count)
            ])
            # s=log(sigma^2): UW is 0.5*exp(-s)*L + 0.5*s, summed over tasks.
            if params["weighting"] == "uw":
                self.task_log_variances = nn.Parameter(torch.zeros(task_count))
            else:
                self.register_buffer("task_log_variances", torch.zeros(task_count))

        def forward(self, x):
            shared = self.shared(x)
            return torch.cat([head(shared) for head in self.heads], dim=1)

    return Model()


def _prepare_data(frame, columns, label_columns, params):
    """Rank and weight only the complete-label cross-section, after feature scaling."""
    missing = set(label_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing multi-task training label columns: {sorted(missing)}")
    clean = frame.reset_index(drop=True)
    x = _features(clean, columns)
    raw = clean[label_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    target = np.zeros(raw.shape, dtype="float32")
    weights = np.zeros(raw.shape, dtype="float32")
    groups, skipped = [], 0
    weight_params = dict(loss="rank_exp_mse", rank_weight_half_life=params["rank_weight_half_life"])
    for date, indices in clean.groupby("trade_date", sort=True).indices.items():
        rows = np.asarray(indices, dtype=np.int64)
        rows = rows[np.isfinite(raw[rows]).all(axis=1)]
        if len(rows) < 2 or (np.ptp(raw[rows], axis=0) <= 0).any():
            skipped += 1
            continue
        for task in range(len(label_columns)):
            values = pd.Series(raw[rows, task])
            target[rows, task] = _zscore_series(values.rank(method="average")).to_numpy(dtype="float32")
            weights[rows, task] = _rank_weights(raw[rows, task], weight_params)
        groups.append((int(date), rows))
    return x, target, weights, groups, skipped


def _dwa_weights(history, task_count, temperature):
    # Initial two EPOCHS use equal weights; ratios use unweighted per-task losses.
    if len(history) < 2:
        return np.ones(task_count, dtype="float32")
    ratio = history[-1] / np.maximum(history[-2], 1e-12)
    logits = ratio / temperature
    scaled = np.exp(logits - logits.max())
    return (task_count * scaled / scaled.sum()).astype("float32")


def _evaluate(torch, model, data, params, device):
    model.eval()
    totals = []
    with torch.no_grad():
        for date, rows in data[3]:
            task_losses = np.zeros(data[1].shape[1], dtype="float64")
            for start in range(0, len(rows), params["batch_size"]):
                block = rows[start:start + params["batch_size"]]
                x = torch.from_numpy(data[0][block]).to(device)
                y = torch.from_numpy(data[1][block]).to(device)
                weight = torch.from_numpy(data[2][block]).to(device)
                prediction = model(x)
                task_losses += (weight * (prediction - y).square()).sum(dim=0).cpu().numpy() / len(rows)
                del x, y, weight, prediction
            if not np.isfinite(task_losses).all():
                raise ValueError(f"nonfinite multi-task validation loss on {date}")
            totals.append(task_losses)
    if not totals:
        raise ValueError("multi-task validation has no complete, nonconstant label cross-sections")
    return np.mean(totals, axis=0)


class MultiTaskAlphaModel(MLPAlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True

    def __init__(self):
        super().__init__()
        self.feature_columns = []
        self.label_columns = []

    def fit(self, train_data, valid_data, context):
        if context.model_search:
            raise ValueError("multi-task MLP does not use model.search")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.label_columns = _label_columns(context)
        self.input_dim = len(self.feature_columns)
        if not self.input_dim or train_data.empty or valid_data.empty:
            raise ValueError("multi-task MLP requires factors and nonempty train/validation partitions")
        if train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("multi-task train/validation dates must be chronological and disjoint")
        train = _prepare_data(train_data, self.feature_columns, self.label_columns, self.params)
        valid = _prepare_data(valid_data, self.feature_columns, self.label_columns, self.params)
        if not train[3] or not valid[3]:
            raise ValueError("multi-task train/validation need dates with all tasks usable")
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        task_count = len(self.label_columns)
        self.model = build_multi_task(nn, self.input_dim, task_count, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        diagnostics = _diagnostics(context)
        raw_history = []
        best_loss, best_state, best_epoch, stale = np.inf, None, 0, 0
        self.loss_history = []
        started = perf_counter()
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                self.model.train()
                dwa = _dwa_weights(raw_history, task_count, self.params["dwa_temperature"])
                coefficients = torch.from_numpy(dwa).to(device)
                total = np.zeros(task_count, dtype="float64")
                total_objective = 0.0
                order = np.random.default_rng(self.params["seed"] + epoch).permutation(len(train[3]))
                for index in order:
                    date, rows = train[3][index]
                    optimizer.zero_grad(set_to_none=True)
                    for start in range(0, len(rows), self.params["batch_size"]):
                        block = rows[start:start + self.params["batch_size"]]
                        x = torch.from_numpy(train[0][block]).to(device)
                        y = torch.from_numpy(train[1][block]).to(device)
                        weight = torch.from_numpy(train[2][block]).to(device)
                        prediction = self.model(x)
                        task_losses = (weight * (prediction - y).square()).sum(dim=0) / len(rows)
                        if self.params["weighting"] == "uw":
                            logvar = self.model.task_log_variances
                            # Allocate the regularizer by chunk fraction: exactly once per date.
                            objective = (0.5 * torch.exp(-logvar) * task_losses
                                         + 0.5 * logvar * (len(block) / len(rows))).sum()
                        else:
                            objective = (coefficients * task_losses).sum()
                        if not bool(torch.isfinite(objective).item()):
                            raise ValueError(f"nonfinite multi-task training loss on {date}")
                        objective.backward()
                        total += task_losses.detach().cpu().numpy()
                        total_objective += objective.item()
                        del x, y, weight, prediction, task_losses, objective
                    gradients = [torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]
                    if not gradients or not bool(torch.stack(gradients).all().item()):
                        raise ValueError(f"nonfinite or absent multi-task gradients on {date}")
                    optimizer.step()
                train_tasks = total / len(train[3])
                raw_history.append(train_tasks.copy())
                valid_tasks = _evaluate(torch, self.model, valid, self.params, device)
                # Selection cannot be gamed by learned uncertainty or changing DWA weights.
                score = float(valid_tasks.mean())
                improved = score < best_loss - self.params["min_delta"]
                if improved:
                    best_loss, best_state, best_epoch, stale = score, _cpu_state_dict(self.model), epoch, 0
                else:
                    stale += 1
                if self.params["weighting"] == "uw":
                    end_weights = (0.5 * torch.exp(-self.model.task_log_variances.detach())).cpu().numpy()
                else:
                    end_weights = dwa
                row = dict(window_id=context.artifact_dir.name, epoch=epoch,
                           train_loss=total_objective / len(train[3]), valid_loss=score,
                           best_loss=best_loss, is_best=improved, stale_epochs=stale,
                           elapsed_seconds=perf_counter() - started)
                for task, label in enumerate(self.label_columns):
                    row[f"train_wmse_{label}"] = float(train_tasks[task])
                    row[f"valid_wmse_{label}"] = float(valid_tasks[task])
                    row[f"end_task_weight_{label}"] = float(end_weights[task])
                self.loss_history.append(row)
                if diagnostics["print_epoch"]:
                    print(f"multi-task window={context.artifact_dir.name} epoch={epoch} "
                          f"method={self.params['weighting']} train_loss={row['train_loss']:.6g} "
                          f"valid_wmse={score:.6g} task_losses={train_tasks.tolist()} "
                          f"task_weights={end_weights.tolist()} stale={stale}/{self.params['patience']}", flush=True)
                if stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = dict(
            model_class=self.__class__.__name__, alpha_id=context.alpha_id, window_id=context.artifact_dir.name,
            params=self.params, feature_columns=self.feature_columns, label_columns=self.label_columns,
            train_rows=len(train_data), valid_rows=len(valid_data),
            train_used_rows=sum(len(rows) for _, rows in train[3]),
            valid_used_rows=sum(len(rows) for _, rows in valid[3]),
            train_dates=len(train[3]), valid_dates=len(valid[3]),
            train_skipped_dates=train[4], valid_skipped_dates=valid[4],
            best_epoch=best_epoch, best_valid_wmse=best_loss, epochs_run=len(self.loss_history), device=str(device),
            target="date_rank_zscore", sample_weights="return_rank_exponential_half_life",
            training_batch="complete_date_with_stock_chunk_gradient_accumulation",
            validation="date_mean_task_mean_wmse_without_task_weights",
            prediction="arithmetic_mean_of_task_heads_without_task_loss_weights",
        )

    def predict(self, data, context):
        if self.model is None:
            raise RuntimeError("multi-task MLP is not fitted")
        if list(context.feature_columns) != self.feature_columns or _label_columns(context) != self.label_columns:
            raise ValueError("multi-task prediction feature/label order changed")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        self.model.to(device).eval()
        result = np.empty(len(data), dtype="float32")
        try:
            with torch.no_grad():
                for start in range(0, len(data), self.params["batch_size"]):
                    block = data.iloc[start:start + self.params["batch_size"]]
                    x = torch.from_numpy(_features(block, self.feature_columns)).to(device)
                    result[start:start + len(block)] = self.model(x).mean(dim=1).cpu().numpy()
                    del x
            if not np.isfinite(result).all():
                raise ValueError("nonfinite multi-task predictions")
            return pd.Series(result, index=data.index, dtype="float32")
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("cannot save an unfitted multi-task MLP")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        artifact_io.torch_save(dict(version=1, model_kind="huatai_multi_task", params=self.params,
                                   feature_columns=self.feature_columns, label_columns=self.label_columns,
                                   state_dict=_cpu_state_dict(self.model), loss_history=self.loss_history,
                                   model_info=self.model_info), path)

    @classmethod
    def load(cls, path):
        torch, nn = _torch_modules()
        saved = artifact_io.torch_load(Path(path), map_location="cpu")
        if saved.get("version") != 1 or saved.get("model_kind") != "huatai_multi_task":
            raise ValueError("unsupported multi-task checkpoint")
        model = cls()
        model.params = _params(saved["params"])
        model.feature_columns = list(saved["feature_columns"])
        model.label_columns = list(saved["label_columns"])
        model.input_dim = len(model.feature_columns)
        model.model = build_multi_task(nn, model.input_dim, len(model.label_columns), model.params)
        model.model.load_state_dict(saved["state_dict"])
        model.model.eval()
        model.loss_history, model.model_info = saved["loss_history"], saved["model_info"]
        return model
