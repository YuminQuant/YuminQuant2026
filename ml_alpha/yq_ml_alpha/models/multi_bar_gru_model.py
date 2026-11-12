from __future__ import annotations

from yq_ml_alpha.output import artifact_io

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.torch_support import (
    _cpu_state_dict,
    _device,
    _release_torch_memory,
    _set_seed,
    _torch_modules,
)
from yq_ml_alpha.models.bar_panel_model import _date_batches, _date_groups, _negative_ic_loss
from yq_ml_alpha.models.bar_training import evaluate_datewise_ic
from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.multi_bar_tensors import (
    _multi_bar_batch, _multi_bar_tensors, _optional_multi_bundle_tensors, _require_multi_bundle_tensors,
)


class MultiBarGRUAlphaModel(AlphaModel):
    prediction_replay_safe = True
    prediction_inputs_readonly = True
    """Two-branch daily + intraday bar GRU trained with date-wise negative IC loss."""

    def __init__(self) -> None:
        self.model = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        self._fit(train_data, valid_data, context, train_tensors=None, valid_tensors=None)

    def fit_bundle(self, train_bundle, valid_bundle, context: ModelContext) -> None:
        self._fit(
            train_bundle.frame,
            valid_bundle.frame,
            context,
            train_tensors=_require_multi_bundle_tensors(train_bundle),
            valid_tensors=_optional_multi_bundle_tensors(valid_bundle),
        )

    def _fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        context: ModelContext,
        *,
        train_tensors: dict[str, np.ndarray] | None,
        valid_tensors: dict[str, np.ndarray] | None,
    ) -> None:
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        if train_data.empty:
            raise ValueError("MultiBarGRUAlphaModel requires non-empty train_data")

        self.model = _MultiBarGRURegressor(nn, self.params).to(device)
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
                    daily_tensor = minute_tensor = y_tensor = pred = loss = None
                    try:
                        daily_np, minute_np = _multi_bar_batch(
                            train_data,
                            context.feature_columns,
                            batch_rows,
                            self.params,
                            train_tensors,
                        )
                        y_np = train_data.loc[batch_rows, context.label_column].astype("float32").to_numpy()
                        daily_tensor = torch.from_numpy(daily_np).to(device)
                        minute_tensor = torch.from_numpy(minute_np).to(device)
                        y_tensor = torch.from_numpy(y_np).to(device)
                        optimizer.zero_grad()
                        pred = self.model(daily_tensor, minute_tensor)
                        loss = _negative_ic_loss(torch, pred, y_tensor)
                        if loss is None:
                            continue
                        loss.backward()
                        optimizer.step()
                        batch_count = int(len(batch_rows))
                        train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                        train_count += batch_count
                    finally:
                        del daily_tensor, minute_tensor, y_tensor, pred, loss

            train_loss = train_loss_sum / train_count if train_count else None
            valid_loss = _eval_multi_date_loss(
                torch,
                self.model,
                valid_data,
                valid_groups,
                context,
                self.params,
                device,
                tensors=valid_tensors,
            )
            score_loss = valid_loss if valid_loss is not None else train_loss
            if score_loss is None:
                raise ValueError("MultiBarGRUAlphaModel could not compute any finite train or valid IC loss")

            is_best = score_loss + 1e-12 < best_loss
            if is_best:
                best_loss = float(score_loss)
                best_epoch = epoch
                best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            elif valid_groups:
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
                    f"window={window_id} epoch={epoch} multi_bar_gru_train_loss={train_text} "
                    f"valid_loss={valid_text} best={best_loss:.6g} patience={stale_epochs}/{patience}",
                    flush=True,
                )

            if valid_groups and patience > 0 and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        _release_torch_memory(torch)
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
            "daily_sequence_length": int(self.params["daily_sequence_length"]),
            "minute_sequence_length": int(self.params["minute_sequence_length"]),
            "input_size": int(self.params["input_size"]),
            "daily_hidden_size": int(self.params["daily_hidden_size"]),
            "minute_hidden_size": int(self.params["minute_hidden_size"]),
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
                daily_np, minute_np = _multi_bar_tensors(data, context.feature_columns, batch_rows, self.params)
                pred = self.model(torch.from_numpy(daily_np).to(device), torch.from_numpy(minute_np).to(device))
                scores[start : start + len(batch_rows)] = pred.detach().cpu().numpy().astype("float32")
        if not getattr(self, "_prediction_active", False):
            self.model.to("cpu")
            _release_torch_memory(torch)
        return pd.Series(scores, index=data.index, dtype="float32")

    def predict_bundle(self, bundle, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        device = _device(torch, str(self.params.get("device", "auto")))
        self.model.to(device)
        self.model.eval()

        tensors = _require_multi_bundle_tensors(bundle)
        batch_size = int(self.params["batch_size"])
        scores = np.empty(len(bundle.frame), dtype="float32")
        row_index = bundle.frame.index.to_numpy()
        with torch.no_grad():
            for start in range(0, len(row_index), batch_size):
                batch_rows = row_index[start : start + batch_size]
                daily_np, minute_np = _multi_bar_batch(bundle.frame, context.feature_columns, batch_rows, self.params, tensors)
                pred = self.model(torch.from_numpy(daily_np).to(device), torch.from_numpy(minute_np).to(device))
                scores[start : start + len(batch_rows)] = pred.detach().cpu().numpy().astype("float32")
        if not getattr(self, "_prediction_active", False):
            self.model.to("cpu")
            _release_torch_memory(torch)
        return pd.Series(scores, index=bundle.frame.index, dtype="float32")

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
    def load(cls, path: str | Path) -> "MultiBarGRUAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = artifact_io.torch_load(Path(path), map_location="cpu")
        model = cls()
        model.params = _params(checkpoint["params"])
        model.model = _MultiBarGRURegressor(nn, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


class _MultiBarGRURegressor:
    def __new__(cls, nn, params: dict[str, Any]):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.daily_gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=int(params["daily_hidden_size"]),
                    num_layers=int(params["daily_num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["daily_num_layers"]) > 1 else 0.0,
                )
                self.minute_gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=int(params["minute_hidden_size"]),
                    num_layers=int(params["minute_num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["minute_num_layers"]) > 1 else 0.0,
                )
                self.daily_batch_norm = nn.BatchNorm1d(int(params["daily_hidden_size"]))
                self.minute_batch_norm = nn.BatchNorm1d(int(params["minute_hidden_size"]))
                self.head = nn.Linear(int(params["daily_hidden_size"]) + int(params["minute_hidden_size"]), 1)

            def forward(self, daily_x, minute_x):
                _, daily_hidden = self.daily_gru(daily_x)
                _, minute_hidden = self.minute_gru(minute_x)
                daily_last = self.daily_batch_norm(daily_hidden[-1])
                minute_last = self.minute_batch_norm(minute_hidden[-1])
                combined = torch.cat([daily_last, minute_last], dim=1)
                return self.head(combined).reshape(-1)

        import torch

        return Model()


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    params["daily_sequence_length"] = int(params.get("daily_sequence_length", 40))
    params["minute_sequence_length"] = int(params.get("minute_sequence_length", 320))
    params["input_size"] = int(params.get("input_size", 6))
    params["daily_hidden_size"] = int(params.get("daily_hidden_size", params.get("hidden_size", 30)))
    params["minute_hidden_size"] = int(params.get("minute_hidden_size", params.get("hidden_size", 30)))
    params["daily_num_layers"] = int(params.get("daily_num_layers", params.get("num_layers", 1)))
    params["minute_num_layers"] = int(params.get("minute_num_layers", params.get("num_layers", 1)))
    params["dropout"] = float(params.get("dropout", 0.0))
    params["epochs"] = int(params.get("epochs", 100))
    params["batch_size"] = int(params.get("batch_size", 5000))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 0.0))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
    params["patience"] = int(params.get("patience", 10))
    if params["daily_sequence_length"] <= 0 or params["minute_sequence_length"] <= 0:
        raise ValueError("daily_sequence_length and minute_sequence_length must be positive")
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


def _eval_multi_date_loss(
    torch,
    model,
    frame,
    groups,
    context,
    params,
    device,
    tensors: dict[str, np.ndarray] | None = None,
) -> float | None:
    def predict_rows(rows):
        daily_np, minute_np = _multi_bar_batch(frame, context.feature_columns, rows, params, tensors)
        return model(torch.from_numpy(daily_np).to(device), torch.from_numpy(minute_np).to(device))

    return evaluate_datewise_ic(torch, model, frame, groups, context, params, predict_rows)
