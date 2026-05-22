from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class BarGRUAlphaModel(AlphaModel):
    """End-to-end bar panel GRU trained with date-wise negative IC loss."""

    def __init__(self) -> None:
        self.model = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        self._fit(train_data, valid_data, context, train_tensor=None, valid_tensor=None)

    def fit_bundle(self, train_bundle, valid_bundle, context: ModelContext) -> None:
        self._fit(
            train_bundle.frame,
            valid_bundle.frame,
            context,
            train_tensor=_require_bar_bundle_tensor(train_bundle),
            valid_tensor=_optional_bar_bundle_tensor(valid_bundle),
        )

    def _fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        context: ModelContext,
        *,
        train_tensor: np.ndarray | None,
        valid_tensor: np.ndarray | None,
    ) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        if train_data.empty:
            raise ValueError("BarGRUAlphaModel requires non-empty train_data")

        self.model = _BarGRURegressor(nn, self.params).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        train_groups = _date_groups(train_data)
        valid_groups = _date_groups(valid_data) if not valid_data.empty else {}
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
            date_order = list(train_groups)
            rng = np.random.default_rng(int(self.params["seed"]) + epoch)
            rng.shuffle(date_order)
            train_loss_sum = 0.0
            train_count = 0

            for trade_date in date_order:
                rows = train_groups[trade_date]
                if len(rows) < 2:
                    continue
                for batch_rows in _date_batches(rows, int(self.params["batch_size"]), rng):
                    if len(batch_rows) < 2:
                        continue
                    x_np = _bar_tensor_batch(train_data, context.feature_columns, batch_rows, self.params, train_tensor)
                    y_np = train_data.loc[batch_rows, context.label_column].astype("float32").to_numpy()
                    x_tensor = torch.from_numpy(x_np).to(device)
                    y_tensor = torch.from_numpy(y_np).to(device)
                    optimizer.zero_grad()
                    pred = self.model(x_tensor)
                    loss = _negative_ic_loss(torch, pred, y_tensor)
                    if loss is None:
                        continue
                    loss.backward()
                    optimizer.step()
                    batch_count = int(len(batch_rows))
                    train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                    train_count += batch_count

            train_loss = train_loss_sum / train_count if train_count else None
            valid_loss = _eval_date_loss(
                torch,
                self.model,
                valid_data,
                valid_groups,
                context,
                self.params,
                device,
                tensor=valid_tensor,
            )
            score_loss = valid_loss if valid_loss is not None else train_loss
            if score_loss is None:
                raise ValueError("BarGRUAlphaModel could not compute any finite train or valid IC loss")

            is_best = score_loss + 1e-12 < best_loss
            if is_best:
                best_loss = float(score_loss)
                best_epoch = epoch
                best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            else:
                if valid_groups:
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
                "model_class": self.__class__.__name__,
            }
            self.loss_history.append(row)

            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                train_text = "nan" if train_loss is None else f"{train_loss:.6g}"
                valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
                print(
                    f"window={window_id} epoch={epoch} bar_gru_train_loss={train_text} "
                    f"valid_loss={valid_text} best={best_loss:.6g} patience={stale_epochs}/{patience}",
                    flush=True,
                )

            if valid_groups and patience > 0 and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        self.model_info = {
            "window_id": window_id,
            "model_class": self.__class__.__name__,
            "alpha_id": context.alpha_id,
            "feature_count": len(context.feature_columns),
            "train_rows": int(len(train_data)),
            "valid_rows": int(len(valid_data)),
            "train_dates": int(len(train_groups)),
            "valid_dates": int(len(valid_groups)),
            "epochs_run": len(self.loss_history),
            "best_epoch": best_epoch,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "device": str(device),
            "sequence_length": int(self.params["sequence_length"]),
            "input_size": int(self.params["input_size"]),
            "hidden_size": int(self.params["hidden_size"]),
            "num_layers": int(self.params["num_layers"]),
            "batch_size": int(self.params["batch_size"]),
            "epochs": int(self.params["epochs"]),
            "patience": int(self.params["patience"]),
            "lr": float(self.params["lr"]),
            "weight_decay": float(self.params["weight_decay"]),
            "loss": "negative_datewise_pearson_ic",
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        device = _device(torch, str(self.params.get("device", "auto")))
        self.model.to(device)
        self.model.eval()

        batch_size = int(self.params["batch_size"])
        scores = np.empty(len(data), dtype="float32")
        row_index = data.index.to_numpy()
        with torch.no_grad():
            for start in range(0, len(row_index), batch_size):
                batch_rows = row_index[start : start + batch_size]
                x_np = _bar_tensor(data, context.feature_columns, batch_rows, self.params)
                pred = self.model(torch.from_numpy(x_np).to(device))
                scores[start : start + len(batch_rows)] = pred.detach().cpu().numpy().astype("float32")
        self.model.to("cpu")
        return pd.Series(scores, index=data.index, dtype="float32")

    def predict_bundle(self, bundle, context: ModelContext) -> pd.Series:
        tensor = _require_bar_bundle_tensor(bundle)
        return self._predict_tensor(bundle.frame, tensor, context)

    def _predict_tensor(self, data: pd.DataFrame, tensor: np.ndarray, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        device = _device(torch, str(self.params.get("device", "auto")))
        self.model.to(device)
        self.model.eval()

        batch_size = int(self.params["batch_size"])
        scores = np.empty(len(data), dtype="float32")
        row_index = data.index.to_numpy()
        with torch.no_grad():
            for start in range(0, len(row_index), batch_size):
                batch_rows = row_index[start : start + batch_size]
                x_np = _bar_tensor_from_array(tensor, batch_rows, self.params)
                pred = self.model(torch.from_numpy(x_np).to(device))
                scores[start : start + len(batch_rows)] = pred.detach().cpu().numpy().astype("float32")
        self.model.to("cpu")
        return pd.Series(scores, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {
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
    def load(cls, path: str | Path) -> "BarGRUAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.params = _params(checkpoint["params"])
        model.model = _BarGRURegressor(nn, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


class _BarGRURegressor:
    def __new__(cls, nn, params: dict[str, Any]):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=int(params["hidden_size"]),
                    num_layers=int(params["num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["num_layers"]) > 1 else 0.0,
                )
                self.batch_norm = nn.BatchNorm1d(int(params["hidden_size"]))
                self.head = nn.Linear(int(params["hidden_size"]), 1)

            def forward(self, x):
                _, hidden = self.gru(x)
                last = hidden[-1]
                normalized = self.batch_norm(last)
                return self.head(normalized).reshape(-1)

        return Model()


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    params["sequence_length"] = int(params.get("sequence_length", 320))
    params["input_size"] = int(params.get("input_size", 6))
    params["hidden_size"] = int(params.get("hidden_size", 30))
    params["num_layers"] = int(params.get("num_layers", 1))
    params["dropout"] = float(params.get("dropout", 0.0))
    params["epochs"] = int(params.get("epochs", 100))
    params["batch_size"] = int(params.get("batch_size", 5000))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 0.0))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
    params["patience"] = int(params.get("patience", 10))
    if params["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    if params["input_size"] <= 0:
        raise ValueError("input_size must be positive")
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


def _date_groups(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    if frame.empty:
        return {}
    groups: dict[int, np.ndarray] = {}
    for trade_date, rows in frame.groupby("trade_date", sort=True).groups.items():
        groups[int(trade_date)] = np.asarray(list(rows))
    return groups


def _date_batches(rows: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(rows) <= batch_size:
        return [rows]
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    return [shuffled[idx : idx + batch_size] for idx in range(0, len(shuffled), batch_size)]


def _bar_tensor(
    frame: pd.DataFrame,
    columns: list[str],
    rows: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    flat = (
        frame.loc[rows, columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
        .to_numpy()
    )
    expected = int(params["sequence_length"]) * int(params["input_size"])
    if flat.shape[1] != expected:
        raise ValueError(f"bar GRU expects {expected} feature columns, got {flat.shape[1]}")
    return flat.reshape(flat.shape[0], int(params["sequence_length"]), int(params["input_size"])).astype(
        "float32",
        copy=False,
    )


def _bar_tensor_batch(
    frame: pd.DataFrame,
    columns: list[str],
    rows: np.ndarray,
    params: dict[str, Any],
    tensor: np.ndarray | None,
) -> np.ndarray:
    if tensor is None:
        return _bar_tensor(frame, columns, rows, params)
    return _bar_tensor_from_array(tensor, rows, params)


def _bar_tensor_from_array(tensor: np.ndarray, rows: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    expected_shape = (int(params["sequence_length"]), int(params["input_size"]))
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected_shape:
        raise ValueError(f"bar GRU expects tensor shape [N,{expected_shape[0]},{expected_shape[1]}], got {tensor.shape}")
    output = np.ascontiguousarray(tensor[np.asarray(rows, dtype=np.int64)], dtype="float32")
    if not np.isfinite(output).all():
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    return output


def _require_bar_bundle_tensor(bundle) -> np.ndarray:
    tensors = getattr(bundle, "tensors", None)
    if not tensors or "bar" not in tensors:
        raise ValueError("BarGRUAlphaModel requires DatasetBundle.tensors['bar']")
    return tensors["bar"]


def _optional_bar_bundle_tensor(bundle) -> np.ndarray | None:
    if bundle is None:
        return None
    tensors = getattr(bundle, "tensors", None)
    if not tensors:
        return None
    return tensors.get("bar")


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


def _eval_date_loss(torch, model, frame, groups, context, params, device, tensor: np.ndarray | None = None) -> float | None:
    if frame.empty or not groups:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < 2:
                continue
            x_np = _bar_tensor_batch(frame, context.feature_columns, rows, params, tensor)
            y_np = frame.loc[rows, context.label_column].astype("float32").to_numpy()
            pred = model(torch.from_numpy(x_np).to(device))
            target = torch.from_numpy(y_np).to(device)
            loss = _negative_ic_loss(torch, pred, target)
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return None
    return float(np.mean(losses))


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("BarGRUAlphaModel requires installing the optional torch package") from exc
    return torch, nn


def _device(torch, value: str):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _set_seed(torch, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
