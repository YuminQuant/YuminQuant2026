from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class _SequenceAlphaModel(AlphaModel):
    rnn_type = "RNN"

    def __init__(self) -> None:
        self.model = None
        self.input_size: int | None = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params, self.rnn_type)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        x_train, self.input_size = _sequence_features(train_data, context.feature_columns, int(self.params["sequence_length"]))
        y_train = train_data[context.label_column].astype("float32").to_numpy().reshape(-1, 1)
        self.model = _SequenceRegressor(nn, self.rnn_type, self.input_size, self.params).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_name = str(self.params["loss"])
        loss_fn = nn.MSELoss() if loss_name == "mse" else None

        x_valid = y_valid = valid_x = valid_y = valid_groups = None
        if not valid_data.empty and context.label_column in valid_data.columns:
            valid_x, _ = _sequence_features(valid_data, context.feature_columns, int(self.params["sequence_length"]))
            valid_y = valid_data[context.label_column].astype("float32").to_numpy().reshape(-1, 1)
            if loss_name != "mse":
                valid_groups = _date_groups(valid_data)

        train_groups = None
        if loss_name != "mse":
            train_groups = _date_groups(train_data)
        batch_size = int(self.params["batch_size"])
        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        self.loss_history = []
        started_at = time.perf_counter()
        has_validation = valid_x is not None if loss_name == "mse" else bool(valid_x is not None and valid_groups)

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0
            if loss_name == "mse":
                order = torch.randperm(x_train.shape[0]).cpu().numpy()
                for start in range(0, x_train.shape[0], batch_size):
                    batch_idx = order[start : start + batch_size]
                    x_batch = y_batch = pred = loss = None
                    try:
                        x_batch = torch.from_numpy(x_train[batch_idx]).to(device)
                        y_batch = torch.from_numpy(y_train[batch_idx]).to(device)
                        optimizer.zero_grad()
                        pred = self.model(x_batch)
                        loss = loss_fn(pred, y_batch)
                        loss.backward()
                        optimizer.step()
                        batch_count = int(batch_idx.shape[0])
                        train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                        train_count += batch_count
                    finally:
                        del x_batch, y_batch, pred, loss
                if getattr(device, "type", None) == "cuda":
                    torch.cuda.empty_cache()
                train_loss = train_loss_sum / max(1, train_count)
                valid_loss = (
                    _eval_mse_loss(torch, loss_fn, self.model, valid_x, valid_y, batch_size, device)
                    if valid_x is not None
                    else None
                )
            else:
                date_order = list(train_groups)
                rng = np.random.default_rng(int(self.params["seed"]) + epoch)
                rng.shuffle(date_order)
                for trade_date in date_order:
                    rows = train_groups[trade_date]
                    if len(rows) < 2:
                        continue
                    for batch_rows in _date_batches(rows, batch_size, rng):
                        if len(batch_rows) < 2:
                            continue
                        x_batch = y_batch = pred = loss = None
                        try:
                            x_batch = torch.from_numpy(x_train[batch_rows]).to(device)
                            y_batch = torch.from_numpy(y_train[batch_rows]).to(device)
                            optimizer.zero_grad()
                            pred = self.model(x_batch)
                            loss = _negative_ic_loss(torch, pred, y_batch)
                            if loss is None:
                                continue
                            loss.backward()
                            optimizer.step()
                            batch_count = int(len(batch_rows))
                            train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                            train_count += batch_count
                        finally:
                            del x_batch, y_batch, pred, loss
                    if getattr(device, "type", None) == "cuda":
                        torch.cuda.empty_cache()
                train_loss = train_loss_sum / train_count if train_count else None
                valid_loss = (
                    _eval_date_ic_loss(torch, self.model, valid_x, valid_y, valid_groups, device)
                    if valid_x is not None and valid_groups
                    else None
                )
            score_loss = valid_loss if valid_loss is not None else train_loss
            if score_loss is None:
                raise ValueError(f"{self.__class__.__name__} could not compute any finite {loss_name} loss")
            is_best = score_loss + 1e-12 < best_loss
            if is_best:
                best_loss = score_loss
                best_epoch = epoch
                if has_validation:
                    best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            else:
                if has_validation:
                    stale_epochs += 1

            row = {
                "window_id": window_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "best_loss": best_loss,
                "is_best": is_best,
                "stale_epochs": stale_epochs,
                "elapsed_seconds": time.perf_counter() - started_at,
                "device": str(device),
                "rnn_type": self.rnn_type,
                "loss": loss_name,
            }
            self.loss_history.append(row)
            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                train_text = "nan" if train_loss is None else f"{train_loss:.6g}"
                valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
                print(
                    f"window={window_id} epoch={epoch} {self.rnn_type.lower()}_train_loss={train_text} "
                    f"valid_loss={valid_text} best={best_loss:.6g} patience={stale_epochs}/{patience}",
                    flush=True,
                )

            if has_validation and patience > 0 and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        self.model_info = {
            "window_id": window_id,
            "model_class": self.__class__.__name__,
            "rnn_type": self.rnn_type,
            "alpha_id": context.alpha_id,
            "input_size": self.input_size,
            "feature_count": len(context.feature_columns),
            "train_rows": int(len(train_data)),
            "valid_rows": int(len(valid_data)),
            "epochs_run": len(self.loss_history),
            "best_epoch": best_epoch,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "device": str(device),
            "sequence_length": int(self.params["sequence_length"]),
            "hidden_size": int(self.params["hidden_size"]),
            "num_layers": int(self.params["num_layers"]),
            "dropout": float(self.params["dropout"]),
            "lr": float(self.params["lr"]),
            "weight_decay": float(self.params["weight_decay"]),
            "batch_size": int(self.params["batch_size"]),
            "patience": int(self.params["patience"]),
            "loss": loss_name,
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        self.model.eval()
        x, _ = _sequence_features(data, context.feature_columns, int(self.params["sequence_length"]))
        with torch.no_grad():
            score = self.model(torch.from_numpy(x)).numpy().reshape(-1)
        return pd.Series(score, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_size is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {
                "rnn_type": self.rnn_type,
                "input_size": self.input_size,
                "params": self.params,
                "state_dict": self.model.state_dict(),
                "loss_history": self.loss_history,
                "model_info": self.model_info,
            },
            path,
        )

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        diagnostics = _diagnostics(context)
        if not diagnostics["enabled"]:
            return []
        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if diagnostics["write_loss_history"] and self.loss_history:
            path = context.artifact_dir / "loss_history.parquet"
            pd.DataFrame(self.loss_history).to_parquet(path, index=False)
            written.append(path)
        if diagnostics["write_model_info"] and self.model_info:
            path = context.artifact_dir / "model_info.json"
            with path.open("w", encoding="utf-8") as file:
                json.dump(self.model_info, file, ensure_ascii=False, indent=2)
            written.append(path)
        return written

    @classmethod
    def load(cls, path: str | Path):
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.params = _params(checkpoint["params"], str(checkpoint["rnn_type"]))
        model.input_size = int(checkpoint["input_size"])
        model.model = _SequenceRegressor(nn, str(checkpoint["rnn_type"]), model.input_size, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


class RNNAlphaModel(_SequenceAlphaModel):
    rnn_type = "RNN"


class LSTMAlphaModel(_SequenceAlphaModel):
    rnn_type = "LSTM"


class GRUAlphaModel(_SequenceAlphaModel):
    rnn_type = "GRU"


class _SequenceRegressor:
    def __new__(cls, nn, rnn_type: str, input_size: int, params: dict[str, Any]):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                module_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[rnn_type]
                self.rnn = module_cls(
                    input_size=input_size,
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["num_layers"]) > 1 else 0.0,
                )
                self.head = nn.Linear(int(params["hidden_size"]), 1)

            def forward(self, x):
                output, _ = self.rnn(x)
                return self.head(output[:, -1, :])

        return Model()


def _params(raw: dict[str, Any], rnn_type: str) -> dict[str, Any]:
    params = dict(raw)
    params["rnn_type"] = rnn_type
    params["sequence_length"] = int(params.get("sequence_length", 6))
    params["hidden_size"] = int(params.get("hidden_size", 64))
    params["num_layers"] = int(params.get("num_layers", 2))
    params["dropout"] = float(params.get("dropout", 0.10))
    params["epochs"] = int(params.get("epochs", 50))
    params["batch_size"] = int(params.get("batch_size", 8192))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 1e-5))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
    params["patience"] = int(params.get("patience", 10))
    params["loss"] = str(params.get("loss", "mse"))
    if params["loss"] not in {"mse", "pearson_ic"}:
        raise ValueError("sequence model loss must be one of: mse, pearson_ic")
    if params["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    return params


def _diagnostics(context: ModelContext) -> dict[str, bool]:
    raw = dict(context.diagnostics or {})
    enabled = bool(raw.get("enabled", False))
    return {
        "enabled": enabled,
        "print_epoch": enabled and bool(raw.get("print_epoch", False)),
        "write_loss_history": enabled and bool(raw.get("write_loss_history", False)),
        "write_model_info": enabled and bool(raw.get("write_model_info", False)),
    }


def _sequence_features(frame: pd.DataFrame, columns: list[str], sequence_length: int) -> tuple[np.ndarray, int]:
    flat = (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    remainder = flat.shape[1] % sequence_length
    if remainder:
        pad_width = sequence_length - remainder
        flat = np.pad(flat, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    input_size = flat.shape[1] // sequence_length
    return flat.reshape(flat.shape[0], sequence_length, input_size).astype("float32", copy=False), input_size


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("sequence alpha models require installing the optional torch package") from exc
    return torch, nn


def _device(torch, value: str):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_loss(torch, loss_fn, model, x_valid, y_valid) -> float:
    model.eval()
    with torch.no_grad():
        loss = loss_fn(model(x_valid), y_valid)
    return float(loss.detach().cpu().item())


def _eval_mse_loss(torch, loss_fn, model, x_valid: np.ndarray, y_valid: np.ndarray, batch_size: int, device) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, x_valid.shape[0], batch_size):
            rows = slice(start, start + batch_size)
            x_batch = y_batch = pred = loss = None
            try:
                x_batch = torch.from_numpy(x_valid[rows]).to(device)
                y_batch = torch.from_numpy(y_valid[rows]).to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                batch_count = int(x_batch.shape[0])
                loss_sum += float(loss.detach().cpu().item()) * batch_count
                count += batch_count
            finally:
                del x_batch, y_batch, pred, loss
    model.train()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return loss_sum / max(1, count)


def _date_groups(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    if frame.empty:
        return {}
    groups: dict[int, np.ndarray] = {}
    for trade_date, rows in frame.groupby("trade_date", sort=True).indices.items():
        groups[int(trade_date)] = np.asarray(rows, dtype=np.int64)
    return groups


def _date_batches(rows: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(rows) <= batch_size:
        return [rows]
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    return [shuffled[idx : idx + batch_size] for idx in range(0, len(shuffled), batch_size)]


def _negative_ic_loss(torch, pred, target):
    if pred.numel() < 2:
        return None
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if int(valid.sum().detach().cpu().item()) < 2:
        return None
    pred = pred[valid]
    target = target[valid]
    pred = pred - pred.mean()
    target = target - target.mean()
    pred_norm = torch.sqrt(torch.sum(pred * pred))
    target_norm = torch.sqrt(torch.sum(target * target))
    eps = 1e-8
    if float(pred_norm.detach().cpu().item()) <= eps or float(target_norm.detach().cpu().item()) <= eps:
        return None
    ic = torch.sum(pred * target) / (pred_norm * target_norm + eps)
    return -ic


def _eval_date_ic_loss(torch, model, x_valid: np.ndarray, y_valid: np.ndarray, groups, device) -> float | None:
    if x_valid is None or y_valid is None or not groups:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < 2:
                continue
            pred = model(torch.from_numpy(x_valid[rows]).to(device))
            target = torch.from_numpy(y_valid[rows]).to(device)
            loss = _negative_ic_loss(torch, pred, target)
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return None
    return float(np.mean(losses))


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
