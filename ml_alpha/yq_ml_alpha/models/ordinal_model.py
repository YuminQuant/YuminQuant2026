from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.bar_gru_model import _cpu_state_dict, _device, _release_torch_memory, _set_seed, _torch_modules
from yq_ml_alpha.models.mlp_model import MLPAlphaModel, _diagnostics, _features


class OrdinalAlphaModel(MLPAlphaModel):
    """Huatai AI 60: one scalar NN with sample-weighted logistic ordinal loss."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_columns: list[str] = []

    def _build_model(self, nn):
        width = self.params["hidden_size"]
        return nn.Sequential(
            nn.Linear(self.input_dim, width), nn.Sigmoid(), nn.BatchNorm1d(width),
            nn.Linear(width, width), nn.Sigmoid(), nn.BatchNorm1d(width),
            nn.Linear(width, width), nn.LeakyReLU(self.params["negative_slope"]),
            nn.Linear(width, 1),
        )

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        if context.model_search:
            raise ValueError("OrdinalAlphaModel does not use model_search")
        self.params = _params(context.model_params)
        self.feature_columns = list(context.feature_columns)
        self.input_dim = len(self.feature_columns)
        if not self.input_dim or train_data.empty or valid_data.empty:
            raise ValueError("ordinal training requires features and nonempty train/validation partitions")
        if train_data.trade_date.max() >= valid_data.trade_date.min():
            raise ValueError("ordinal validation dates must follow all training dates")
        train = _prepare_data(train_data, context, self.params)
        valid = _prepare_data(valid_data, context, self.params)
        if not train[3] or not valid[3]:
            raise ValueError("ordinal train/validation each require dates with at least two distinct finite labels")
        torch, nn = _torch_modules()
        _set_seed(torch, self.params["seed"])
        device = _device(torch, self.params["device"])
        self.model = self._build_model(nn).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"],
                                     weight_decay=self.params["weight_decay"])
        diagnostics = _diagnostics(context)
        started = time.perf_counter()
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        self.loss_history = []
        try:
            for epoch in range(1, self.params["epochs"] + 1):
                order = np.random.default_rng(self.params["seed"] + epoch).permutation(len(train[3]))
                self.model.train()
                losses = []
                for index in order:
                    rows, thresholds = train[3][index]
                    # One complete date per optimizer step preserves cross-sectional BatchNorm.
                    optimizer.zero_grad(set_to_none=True)
                    prediction = self.model(torch.from_numpy(train[0][rows]).to(device)).reshape(-1)
                    loss = _objective(torch, prediction, torch.from_numpy(train[1][rows]).to(device),
                                      torch.from_numpy(train[2][rows]).to(device),
                                      torch.from_numpy(thresholds).to(device))
                    if not bool(torch.isfinite(loss).item()):
                        raise ValueError(f"non-finite ordinal train loss: window={context.artifact_dir.name}, epoch={epoch}")
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
                    del prediction, loss
                train_loss = float(np.mean(losses))
                valid_loss = _evaluate(torch, self.model, valid, self.params, device)
                if not math.isfinite(valid_loss):
                    raise ValueError("non-finite ordinal validation loss")
                improved = valid_loss < best_loss - 1e-12
                if improved:
                    best_loss, best_epoch, stale = valid_loss, epoch, 0
                    best_state = _cpu_state_dict(self.model)
                else:
                    stale += 1
                self.loss_history.append({
                    "window_id": context.artifact_dir.name, "epoch": epoch,
                    "train_loss": train_loss, "valid_loss": valid_loss, "best_loss": best_loss,
                    "is_best": improved, "stale_epochs": stale,
                    "elapsed_seconds": time.perf_counter() - started, "device": str(device),
                })
                if diagnostics["enabled"] and diagnostics["print_epoch"]:
                    print(f"window={context.artifact_dir.name} epoch={epoch} sw_ord_logi={train_loss:.6g} "
                          f"valid_loss={valid_loss:.6g} patience={stale}/{self.params['patience']}", flush=True)
                if self.params["patience"] > 0 and stale >= self.params["patience"]:
                    break
            self.model.load_state_dict(best_state)
        finally:
            self.model.cpu().eval()
            del optimizer
            _release_torch_memory(torch)
        self.model_info = {
            "model_class": self.__class__.__name__, "alpha_id": context.alpha_id,
            "window_id": context.artifact_dir.name, "feature_columns": self.feature_columns,
            "params": self.params, "objective": "sw_ord_logi", "ensemble": False,
            "train_rows": len(train_data), "valid_rows": len(valid_data),
            "train_dates": len(train[3]), "valid_dates": len(valid[3]),
            "train_used_rows": sum(len(rows) for rows, _ in train[3]),
            "valid_used_rows": sum(len(rows) for rows, _ in valid[3]),
            "train_skipped_dates": train[4], "valid_skipped_dates": valid[4],
            "best_epoch": best_epoch, "best_loss": best_loss,
            "epochs_run": len(self.loss_history), "device": str(device),
            "training_batch": "complete_date", "score": "scalar_ordinal_score",
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        if list(context.feature_columns) != self.feature_columns:
            raise ValueError("factor columns/order differ from fitted ordinal model")
        torch, _ = _torch_modules()
        device = _device(torch, self.params["device"])
        scores = np.empty(len(data), dtype="float32")
        self.model.to(device).eval()
        try:
            with torch.no_grad():
                for start in range(0, len(data), self.params["predict_batch_size"]):
                    frame = data.iloc[start:start + self.params["predict_batch_size"]]
                    x = torch.from_numpy(_features(frame, self.feature_columns)).to(device)
                    scores[start:start + len(frame)] = self.model(x).reshape(-1).cpu().numpy()
                    del x
        finally:
            if not getattr(self, "_prediction_active", False):
                self.model.cpu()
                _release_torch_memory(torch)
        return pd.Series(scores, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": self.params, "feature_columns": self.feature_columns,
                    "state_dict": _cpu_state_dict(self.model), "loss_history": self.loss_history,
                    "model_info": self.model_info}, path)

    @classmethod
    def load(cls, path: str | Path) -> "OrdinalAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        wrapper = cls()
        wrapper.params = _params(checkpoint["params"])
        wrapper.feature_columns = list(checkpoint["feature_columns"])
        wrapper.input_dim = len(wrapper.feature_columns)
        wrapper.model = wrapper._build_model(nn)
        wrapper.model.load_state_dict(checkpoint["state_dict"])
        wrapper.model.eval()
        wrapper.loss_history = list(checkpoint.get("loss_history", []))
        wrapper.model_info = dict(checkpoint.get("model_info", {}))
        return wrapper


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    for key, default in {"hidden_size": 64, "n_classes": 10, "epochs": 100,
                         "patience": 20, "predict_batch_size": 3000, "seed": 42}.items():
        params[key] = int(params.get(key, default))
        if key != "seed" and params[key] < (0 if key == "patience" else 1):
            raise ValueError(f"invalid ordinal parameter {key}")
    if params["n_classes"] < 2:
        raise ValueError("n_classes must be at least two")
    for key, default in {"lr": 0.0001, "weight_decay": 0.0, "negative_slope": 0.01,
                         "rank_weight_half_life": 0.5}.items():
        params[key] = float(params.get(key, default))
        if not math.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative")
    if params["lr"] == 0 or params["rank_weight_half_life"] == 0:
        raise ValueError("lr and rank_weight_half_life must be positive")
    params["device"] = str(params.get("device", "auto"))
    return params


def _prepare_data(frame, context, params):
    x = _features(frame, context.feature_columns)
    y = frame[context.label_column].to_numpy(dtype="float32")
    weights = np.zeros(len(frame), dtype="float32")
    groups, skipped = [], 0
    quantiles = np.arange(1, params["n_classes"]) / params["n_classes"]
    for indices in frame.groupby("trade_date", sort=True).indices.values():
        rows = np.asarray(indices, dtype=np.int64)
        rows = rows[np.isfinite(y[rows])]
        if len(rows) < 2 or np.ptp(y[rows]) == 0:
            skipped += 1
            continue
        ranks = pd.Series(y[rows]).rank(method="average", ascending=False).to_numpy()
        percentile = (ranks - 1.0) / (len(rows) - 1)
        # A common shift cancels in the weighted mean and prevents all-zero underflow.
        weights[rows] = np.exp2(-(percentile - percentile.min()) / params["rank_weight_half_life"])
        thresholds = np.quantile(y[rows], quantiles).astype("float32")
        groups.append((rows, thresholds))
    if skipped:
        print(f"ordinal preparation: skipped_dates={skipped} (fewer than two distinct finite labels)", flush=True)
    return x, y, weights, groups, skipped


def _objective(torch, prediction, target, weights, thresholds):
    # Each scalar prediction is compared with all K-1 observed cross-sectional cuts.
    signs = torch.where(target[:, None] >= thresholds[None, :], 1.0, -1.0)
    margins = signs * (prediction[:, None] - thresholds[None, :])
    per_row = torch.nn.functional.softplus(-margins).sum(dim=1)
    return (weights * per_row).sum() / weights.sum()


def _evaluate(torch, model, data, params, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for rows, thresholds in data[3]:
            # eval() uses fitted BatchNorm statistics, so stock chunks are equivalent.
            predictions = []
            for start in range(0, len(rows), params["predict_batch_size"]):
                selected = rows[start:start + params["predict_batch_size"]]
                predictions.append(model(torch.from_numpy(data[0][selected]).to(device)).reshape(-1).cpu())
            loss = _objective(torch, torch.cat(predictions), torch.from_numpy(data[1][rows]),
                              torch.from_numpy(data[2][rows]), torch.from_numpy(thresholds))
            losses.append(loss.item())
    return float(np.mean(losses))
