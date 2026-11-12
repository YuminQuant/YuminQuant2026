from __future__ import annotations

from yq_ml_alpha.output import artifact_io

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class BarPanelAlphaModel(AlphaModel):
    """Shared tensor input and checkpoint handling for daily bar-panel models."""

    prediction_replay_safe = True
    prediction_inputs_readonly = True

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
        if not getattr(self, "_prediction_active", False):
            self.model.to("cpu")
            _release_torch_memory(torch)
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
        if not getattr(self, "_prediction_active", False):
            self.model.to("cpu")
            _release_torch_memory(torch)
        return pd.Series(scores, index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        path = Path(path)
        artifact_io.mkdir(path.parent, parents=True, exist_ok=True)
        self.model.to("cpu")
        _release_torch_memory(torch)
        artifact_io.torch_save(
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
        artifact_io.mkdir(context.artifact_dir, parents=True, exist_ok=True)
        written: list[Path] = []
        if diagnostics["write_loss_history"] and self.loss_history:
            path = context.artifact_dir / "loss_history.parquet"
            artifact_io.write_parquet(pd.DataFrame(self.loss_history), path, index=False)
            written.append(path)
        if diagnostics["write_model_info"] and self.model_info:
            path = context.artifact_dir / "model_info.json"
            with artifact_io.open_file(path, "w", encoding="utf-8") as file:
                json.dump(self.model_info, file, ensure_ascii=False, indent=2)
            written.append(path)
        return written

    @classmethod
    def load(cls, path: str | Path) -> "BarPanelAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = artifact_io.torch_load(Path(path), map_location="cpu")
        model = cls()
        model.params = model._normalize_params(checkpoint["params"])
        model.model = model._build_regressor(nn)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


def normalize_bar_params(raw: dict[str, Any]) -> dict[str, Any]:
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
        raise ValueError(f"bar panel expects {expected} feature columns, got {flat.shape[1]}")
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
        raise ValueError(f"bar panel expects tensor shape [N,{expected_shape[0]},{expected_shape[1]}], got {tensor.shape}")
    output = np.ascontiguousarray(tensor[np.asarray(rows, dtype=np.int64)], dtype="float32")
    if not np.isfinite(output).all():
        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    return output


def _require_bar_bundle_tensor(bundle) -> np.ndarray:
    tensors = getattr(bundle, "tensors", None)
    if not tensors or "bar" not in tensors:
        raise ValueError("Bar panel model requires DatasetBundle.tensors['bar']")
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


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("BarGRUAlphaModel requires installing the optional torch package") from exc
    return torch, nn


def _release_torch_memory(torch) -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        return


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
