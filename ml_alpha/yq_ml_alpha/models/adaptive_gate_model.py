from __future__ import annotations

from yq_ml_alpha.output import artifact_io

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.bar_gru_model import _cpu_state_dict, _device, _release_torch_memory, _set_seed, _torch_modules
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _diagnostics, _features


class AdaptiveGateAlphaModel(MLPAlphaModel):
    """Single-gate adaptive-depth factor combination (Huatai AI 71)."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_columns: list[str] = []

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if not self.input_dim or train_data.empty:
            raise ValueError("AdaptiveGateAlphaModel requires training rows and factor columns")
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = _AdaptiveGateRegressor(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])
        train = _prepare_data(train_data, context, self.params)
        valid = _prepare_data(valid_data, context, self.params)
        if not train[3]:
            raise ValueError("training requires at least two finite labels per date")
        if not valid_data.empty and not valid[3]:
            raise ValueError("validation has no usable dates")
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        started = time.perf_counter()
        self.loss_history = []
        diagnostics = _diagnostics(context)
        for epoch in range(1, self.params["epochs"] + 1):
            order = list(train[3])
            np.random.default_rng(self.params["seed"] + epoch).shuffle(order)
            self.model.train()
            metrics = []
            step_dates = self.params["dates_per_step"]
            for start in range(0, len(order), step_dates):
                batch_dates = order[start:start + step_dates]
                optimizer.zero_grad()
                for rows in batch_dates:
                    x = torch.from_numpy(train[0][rows]).to(device)
                    y = torch.from_numpy(train[1][rows]).to(device)
                    weights = torch.from_numpy(train[2][rows]).to(device)
                    prediction, gate = self.model(x)
                    loss, mse, penalty = _objective(prediction, y, weights, gate, self.params)
                    (loss / len(batch_dates)).backward()
                    metrics.append([loss.item(), mse.item(), penalty.item(), gate.detach().mean().item()])
                    del x, y, weights, prediction, gate, loss, mse, penalty
                optimizer.step()
            train_metrics = np.mean(metrics, axis=0)
            valid_metrics = _evaluate(torch, self.model, valid, self.params, device)
            score = float(valid_metrics[0] if valid_metrics is not None else train_metrics[0])
            if not math.isfinite(score):
                raise ValueError("non-finite adaptive-gate loss")
            improved = score < best_loss - 1e-12
            if improved:
                best_loss, best_epoch, stale = score, epoch, 0
                best_state = _cpu_state_dict(self.model)
            else:
                stale += 1
            row = {"window_id": context.artifact_dir.name, "epoch": epoch,
                   "train_loss": float(train_metrics[0]), "valid_loss": None if valid_metrics is None else float(valid_metrics[0]),
                   "train_weighted_mse": float(train_metrics[1]), "train_gate_loss": float(train_metrics[2]),
                   "train_gate_pass_rate": float(train_metrics[3]),
                   "valid_weighted_mse": None if valid_metrics is None else float(valid_metrics[1]),
                   "valid_gate_loss": None if valid_metrics is None else float(valid_metrics[2]),
                   "valid_gate_pass_rate": None if valid_metrics is None else float(valid_metrics[3]),
                   "best_loss": best_loss, "is_best": improved, "stale_epochs": stale,
                   "elapsed_seconds": time.perf_counter() - started, "device": str(device)}
            self.loss_history.append(row)
            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                print(f"window={context.artifact_dir.name} epoch={epoch} adaptive_loss={train_metrics[0]:.6g} "
                      f"valid_loss={row['valid_loss']} gate_pass={train_metrics[3]:.4f} "
                      f"patience={stale}/{self.params['patience']}", flush=True)
            if valid_metrics is not None and self.params["patience"] > 0 and stale >= self.params["patience"]:
                break
        self.model.load_state_dict(best_state)
        self.model.cpu().eval()
        _release_torch_memory(torch)
        self.model_info = {"model_class": self.__class__.__name__, "alpha_id": context.alpha_id,
                           "window_id": context.artifact_dir.name, "feature_columns": self.feature_columns,
                           "train_rows": len(train_data), "valid_rows": len(valid_data),
                           "train_dates": len(train[3]), "valid_dates": len(valid[3]),
                           "best_epoch": best_epoch, "best_loss": best_loss, "epochs_run": len(self.loss_history),
                           "device": str(device), "params": self.params,
                           "inference_gate": "deterministic_argmax", "gate_count": 1}

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("factor columns/order differ from fitted adaptive model")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        scores = np.empty(len(data), dtype="float32")
        self.model.to(device).eval()
        try:
            with torch.no_grad():
                for start in range(0, len(data), self.params["predict_batch_size"]):
                    frame = data.iloc[start:start + self.params["predict_batch_size"]]
                    prediction, _ = self.model(torch.from_numpy(_features(frame, self.feature_columns)).to(device))
                    scores[start:start + len(frame)] = prediction.cpu().numpy()
        finally:
            self.model.cpu()
            if not getattr(self, "_prediction_active", False):
                _release_torch_memory(torch)
        return pd.Series(scores, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        self.model.cpu()
        artifact_io.torch_save({"params": self.params, "feature_columns": self.feature_columns,
                    "state_dict": self.model.state_dict(), "loss_history": self.loss_history,
                    "model_info": self.model_info}, path)

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveGateAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = artifact_io.torch_load(Path(path), map_location="cpu")
        wrapper = cls()
        wrapper.params = _params(checkpoint["params"])
        wrapper.feature_columns = list(checkpoint["feature_columns"])
        wrapper.input_dim = len(wrapper.feature_columns)
        wrapper.model = _AdaptiveGateRegressor(nn, wrapper.input_dim, wrapper.params)
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.eval()
        wrapper.loss_history = list(checkpoint.get("loss_history", []))
        wrapper.model_info = dict(checkpoint.get("model_info", {}))
        return wrapper


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    for key, default in {"hidden_size": 64, "gate_hidden_size": 64, "epochs": 100, "patience": 20,
                         "dates_per_step": 32, "predict_batch_size": 3000, "seed": 42}.items():
        params[key] = int(params.get(key, default))
        if key not in {"seed", "patience"} and params[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key, default in {"lr": 0.001, "weight_decay": 0.0, "negative_slope": 0.01,
                         "gate_temperature": 1.0, "target_pass_rate": 0.4, "gate_loss_weight": 10.0,
                         "rank_weight_half_life": 0.2}.items():
        params[key] = float(params.get(key, default))
        if not math.isfinite(params[key]):
            raise ValueError(f"{key} must be finite")
    if not 0.0 <= params["target_pass_rate"] <= 1.0:
        raise ValueError("target_pass_rate must be in [0, 1]")
    if min(params["lr"], params["gate_temperature"], params["rank_weight_half_life"]) <= 0:
        raise ValueError("lr, gate_temperature and rank_weight_half_life must be positive")
    if min(params["weight_decay"], params["negative_slope"], params["gate_loss_weight"], params["patience"]) < 0:
        raise ValueError("regularization, negative_slope and patience must be non-negative")
    params["loss"] = str(params.get("loss", "rank_exp_mse"))
    if params["loss"] not in {"mse", "rank_exp_mse"}:
        raise ValueError("loss must be mse or rank_exp_mse")
    params["device"] = str(params.get("device", "auto"))
    return params


class _AdaptiveGateRegressor:
    def __new__(cls, nn, input_dim: int, params: dict[str, Any]):
        import torch
        from torch.nn import functional as F

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = params["hidden_size"]
                activation = lambda: nn.LeakyReLU(params["negative_slope"])
                self.entry = nn.Sequential(nn.Linear(input_dim, width), activation())
                self.optional = nn.Sequential(nn.Linear(width, width), activation())
                self.exit = nn.Sequential(nn.Linear(width, width), activation(), nn.Linear(width, 1))
                self.gate = nn.Sequential(nn.Linear(input_dim, params["gate_hidden_size"]),
                                          nn.BatchNorm1d(params["gate_hidden_size"]), nn.ReLU(),
                                          nn.Linear(params["gate_hidden_size"], 2))

            def forward(self, x):
                logits = self.gate(x)
                hidden = self.entry(x)
                if self.training:
                    # Gumbel-softmax consumes logits (equivalent to log softmax probabilities).
                    passed = F.gumbel_softmax(logits, tau=params["gate_temperature"], hard=True, dim=-1)[:, 1]
                    transformed = self.optional(hidden)
                    hidden = passed[:, None] * transformed + (1.0 - passed[:, None]) * hidden
                else:
                    # No Gumbel noise in validation/inference: stable routing and batch-invariant scores.
                    selected = logits.argmax(dim=-1) == 1
                    passed = selected.to(hidden.dtype)
                    routed = hidden.clone()
                    if selected.any():
                        routed[selected] = self.optional(hidden[selected])
                    hidden = routed
                return self.exit(hidden).reshape(-1), passed

        return Model()


def _rank_weights(values: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if params["loss"] == "mse":
        return np.ones(len(values), dtype="float32")
    ranks = pd.Series(values).rank(method="average", ascending=False).to_numpy()
    percentile = (ranks - 1.0) / max(1, len(values) - 1)
    # Subtract the best rank before exponentiation to avoid all-zero underflow.
    weights = np.exp2(-(percentile - percentile.min()) / params["rank_weight_half_life"])
    return (weights / weights.mean()).astype("float32")


def _prepare_data(frame, context, params):
    if frame.empty:
        return (np.empty((0, len(context.feature_columns)), dtype="float32"),
                np.empty(0, dtype="float32"), np.empty(0, dtype="float32"), [])
    clean = frame.reset_index(drop=True)
    x = _features(clean, context.feature_columns)
    y = clean[context.label_column].to_numpy(dtype="float32") if len(clean) else np.empty(0, dtype="float32")
    weights = np.zeros(len(clean), dtype="float32")
    groups = []
    if len(clean):
        for indices in clean.groupby("trade_date", sort=True).indices.values():
            rows = np.asarray(indices, dtype=np.int64)
            rows = rows[np.isfinite(y[rows])]
            if len(rows) >= 2:
                weights[rows] = _rank_weights(y[rows], params)
                groups.append(rows)
    return x, y, weights, groups


def _objective(prediction, target, weights, gate, params):
    mse = (weights * (prediction - target).square()).sum() / weights.sum()
    penalty = (gate.mean() - params["target_pass_rate"]).square()
    return mse + params["gate_loss_weight"] * penalty, mse, penalty


def _evaluate(torch, model, data, params, device):
    if not data[3]:
        return None
    model.eval()
    metrics = []
    with torch.no_grad():
        for rows in data[3]:
            predictions, gates = [], []
            for start in range(0, len(rows), params["predict_batch_size"]):
                selected = rows[start:start + params["predict_batch_size"]]
                prediction, gate = model(torch.from_numpy(data[0][selected]).to(device))
                predictions.append(prediction.cpu())
                gates.append(gate.cpu())
            prediction, gate = torch.cat(predictions), torch.cat(gates)
            loss, mse, penalty = _objective(prediction, torch.from_numpy(data[1][rows]),
                                             torch.from_numpy(data[2][rows]), gate, params)
            metrics.append([loss.item(), mse.item(), penalty.item(), gate.mean().item()])
    return np.mean(metrics, axis=0)
