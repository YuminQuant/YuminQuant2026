from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class MLPAlphaModel(AlphaModel):
    """PyTorch MLP regressor for factor-frame alpha combination."""

    def __init__(self) -> None:
        self.model = None
        self.input_dim: int | None = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        x_train = _features(train_data, context.feature_columns)
        y_train = train_data[context.label_column].astype("float32").to_numpy().reshape(-1, 1)
        self.input_dim = x_train.shape[1]
        self.model = _build_network(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        loss_fn = nn.MSELoss()

        x_valid = y_valid = None
        if not valid_data.empty and context.label_column in valid_data.columns:
            x_valid = torch.from_numpy(_features(valid_data, context.feature_columns)).to(device)
            y_valid = torch.from_numpy(
                valid_data[context.label_column].astype("float32").to_numpy().reshape(-1, 1)
            ).to(device)

        x_tensor = torch.from_numpy(x_train).to(device)
        y_tensor = torch.from_numpy(y_train).to(device)
        batch_size = int(self.params["batch_size"])
        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        self.loss_history = []
        started_at = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self.model.train()
            order = torch.randperm(x_tensor.shape[0], device=device)
            train_loss_sum = 0.0
            train_count = 0
            for start in range(0, x_tensor.shape[0], batch_size):
                batch_idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model(x_tensor[batch_idx]), y_tensor[batch_idx])
                loss.backward()
                optimizer.step()
                batch_count = int(batch_idx.shape[0])
                train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                train_count += batch_count

            train_loss = train_loss_sum / max(1, train_count)
            valid_loss = _eval_loss(torch, loss_fn, self.model, x_valid, y_valid) if x_valid is not None else None
            score_loss = valid_loss if valid_loss is not None else train_loss
            is_best = score_loss + 1e-12 < best_loss
            if is_best:
                best_loss = score_loss
                best_epoch = epoch
                if x_valid is not None:
                    best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            else:
                if x_valid is not None:
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
            }
            self.loss_history.append(row)
            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
                print(
                    f"window={window_id} epoch={epoch} train_loss={train_loss:.6g} "
                    f"valid_loss={valid_text} best={best_loss:.6g} "
                    f"patience={stale_epochs}/{patience}",
                    flush=True,
                )

            if x_valid is not None and patience > 0 and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        self.model_info = {
            "window_id": window_id,
            "model_class": self.__class__.__name__,
            "alpha_id": context.alpha_id,
            "input_dim": self.input_dim,
            "feature_count": len(context.feature_columns),
            "train_rows": int(len(train_data)),
            "valid_rows": int(len(valid_data)),
            "epochs_run": len(self.loss_history),
            "best_epoch": best_epoch,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "device": str(device),
            "hidden_layers": list(self.params["hidden_layers"]),
            "dropout": float(self.params["dropout"]),
            "lr": float(self.params["lr"]),
            "weight_decay": float(self.params["weight_decay"]),
            "batch_size": int(self.params["batch_size"]),
            "patience": int(self.params["patience"]),
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        self.model.eval()
        with torch.no_grad():
            score = self.model(torch.from_numpy(_features(data, context.feature_columns))).numpy().reshape(-1)
        return pd.Series(score, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_dim is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {
                "input_dim": self.input_dim,
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
    def load(cls, path: str | Path) -> "MLPAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.input_dim = int(checkpoint["input_dim"])
        model.params = _params(checkpoint["params"])
        model.model = _build_network(nn, model.input_dim, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    hidden_layers = params.get("hidden_layers", [128, 64])
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    params["hidden_layers"] = [int(value) for value in hidden_layers]
    params["dropout"] = float(params.get("dropout", 0.10))
    params["epochs"] = int(params.get("epochs", 50))
    params["batch_size"] = int(params.get("batch_size", 8192))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 1e-5))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
    params["patience"] = int(params.get("patience", 10))
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


def _build_network(nn, input_dim: int, params: dict[str, Any]):
    layers = []
    current_dim = input_dim
    for hidden_dim in params["hidden_layers"]:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        dropout = float(params["dropout"])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, 1))
    return nn.Sequential(*layers)


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return (
        frame[columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("MLPAlphaModel requires installing the optional torch package") from exc
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


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
