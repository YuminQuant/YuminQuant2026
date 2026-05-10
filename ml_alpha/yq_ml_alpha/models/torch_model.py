from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class TorchMLPAlphaModel(AlphaModel):
    """PyTorch MLP regressor for factor-frame alpha combination."""

    def __init__(self) -> None:
        self.model = None
        self.input_dim: int | None = None
        self.params: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))

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
        stale_epochs = 0

        for _ in range(epochs):
            self.model.train()
            order = torch.randperm(x_tensor.shape[0], device=device)
            for start in range(0, x_tensor.shape[0], batch_size):
                batch_idx = order[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.model(x_tensor[batch_idx]), y_tensor[batch_idx])
                loss.backward()
                optimizer.step()

            if x_valid is None:
                continue
            current_loss = _eval_loss(torch, loss_fn, self.model, x_valid, y_valid)
            if current_loss + 1e-12 < best_loss:
                best_loss = current_loss
                best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if patience > 0 and stale_epochs >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")

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
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TorchMLPAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.input_dim = int(checkpoint["input_dim"])
        model.params = _params(checkpoint["params"])
        model.model = _build_network(nn, model.input_dim, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
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
        raise ImportError("TorchMLPAlphaModel requires installing the optional torch package") from exc
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
