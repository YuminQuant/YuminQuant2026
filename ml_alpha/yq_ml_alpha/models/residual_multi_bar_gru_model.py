from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from yq_ml_alpha.models.bar_gru_model import (
    _cpu_state_dict,
    _date_batches,
    _date_groups,
    _device,
    _negative_ic_loss,
    _set_seed,
    _torch_modules,
)
from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.models.multi_bar_gru_model import _multi_bar_tensors


class ResidualMultiBarGRUAlphaModel(AlphaModel):
    """Two-stage daily-frozen residual GRU model for daily + intraday bar panels."""

    def __init__(self) -> None:
        self.model = None
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

        if train_data.empty:
            raise ValueError("ResidualMultiBarGRUAlphaModel requires non-empty train_data")

        self.model = _ResidualMultiBarGRURegressor(nn, self.params).to(device)
        train_groups = _date_groups(train_data)
        valid_groups = _date_groups(valid_data) if not valid_data.empty else {}
        self.loss_history = []
        started_at = time.perf_counter()

        stage1 = _run_stage(
            torch,
            self.model,
            train_data,
            valid_data,
            train_groups,
            valid_groups,
            context,
            self.params,
            device,
            stage="stage1_daily",
            epochs=int(self.params["stage1_epochs"]),
            patience=int(self.params["stage1_patience"]),
            trainable_parameters=self.model.daily_parameters(),
            forward_name="forward_daily",
            diagnostics=diagnostics,
            window_id=window_id,
            started_at=started_at,
        )
        if stage1["best_state"] is not None:
            self.model.load_state_dict(stage1["best_state"])

        self.model.freeze_daily_branch()
        stage2 = _run_stage(
            torch,
            self.model,
            train_data,
            valid_data,
            train_groups,
            valid_groups,
            context,
            self.params,
            device,
            stage="stage2_residual",
            epochs=int(self.params["stage2_epochs"]),
            patience=int(self.params["stage2_patience"]),
            trainable_parameters=self.model.minute_parameters(),
            forward_name="forward_residual",
            diagnostics=diagnostics,
            window_id=window_id,
            started_at=started_at,
        )
        if stage2["best_state"] is not None:
            self.model.load_state_dict(stage2["best_state"])
            self.model.freeze_daily_branch()

        self.loss_history = [*stage1["history"], *stage2["history"]]
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
            "stage1_best_epoch": stage1["best_epoch"],
            "stage1_best_loss": stage1["best_loss"],
            "stage2_best_epoch": stage2["best_epoch"],
            "stage2_best_loss": stage2["best_loss"],
            "daily_branch_frozen_after_stage1": True,
            "device": str(device),
            "daily_sequence_length": int(self.params["daily_sequence_length"]),
            "minute_sequence_length": int(self.params["minute_sequence_length"]),
            "input_size": int(self.params["input_size"]),
            "daily_hidden_size": int(self.params["daily_hidden_size"]),
            "minute_hidden_size": int(self.params["minute_hidden_size"]),
            "batch_size": int(self.params["batch_size"]),
            "stage1_epochs": int(self.params["stage1_epochs"]),
            "stage2_epochs": int(self.params["stage2_epochs"]),
            "stage1_patience": int(self.params["stage1_patience"]),
            "stage2_patience": int(self.params["stage2_patience"]),
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
                pred = self.model.forward_residual(
                    torch.from_numpy(daily_np).to(device),
                    torch.from_numpy(minute_np).to(device),
                )
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
    def load(cls, path: str | Path) -> "ResidualMultiBarGRUAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.params = _params(checkpoint["params"])
        model.model = _ResidualMultiBarGRURegressor(nn, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.freeze_daily_branch()
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


class _ResidualMultiBarGRURegressor:
    def __new__(cls, nn, params: dict[str, Any]):
        import torch

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
                self.daily_batch_norm = nn.BatchNorm1d(int(params["daily_hidden_size"]))
                self.daily_head = nn.Linear(int(params["daily_hidden_size"]), 1)

                self.minute_gru = nn.GRU(
                    input_size=int(params["input_size"]),
                    hidden_size=int(params["minute_hidden_size"]),
                    num_layers=int(params["minute_num_layers"]),
                    batch_first=True,
                    dropout=float(params["dropout"]) if int(params["minute_num_layers"]) > 1 else 0.0,
                )
                self.minute_batch_norm = nn.BatchNorm1d(int(params["minute_hidden_size"]))
                self.minute_head = nn.Linear(int(params["minute_hidden_size"]), 1)

            def daily_parameters(self):
                yield from self.daily_gru.parameters()
                yield from self.daily_batch_norm.parameters()
                yield from self.daily_head.parameters()

            def minute_parameters(self):
                yield from self.minute_gru.parameters()
                yield from self.minute_batch_norm.parameters()
                yield from self.minute_head.parameters()

            def freeze_daily_branch(self) -> None:
                for parameter in self.daily_parameters():
                    parameter.requires_grad = False
                self.daily_gru.eval()
                self.daily_batch_norm.eval()
                self.daily_head.eval()

            def train_minute_only(self) -> None:
                self.train()
                self.freeze_daily_branch()

            def forward_daily(self, daily_x):
                _, hidden = self.daily_gru(daily_x)
                normalized = self.daily_batch_norm(hidden[-1])
                return self.daily_head(normalized).reshape(-1)

            def forward_minute(self, minute_x):
                _, hidden = self.minute_gru(minute_x)
                normalized = self.minute_batch_norm(hidden[-1])
                return self.minute_head(normalized).reshape(-1)

            def forward_residual(self, daily_x, minute_x):
                with torch.no_grad():
                    daily_score = self.forward_daily(daily_x)
                return daily_score + self.forward_minute(minute_x)

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
    params["stage1_epochs"] = int(params.get("stage1_epochs", params.get("epochs", 100)))
    params["stage2_epochs"] = int(params.get("stage2_epochs", params.get("epochs", 100)))
    params["stage1_patience"] = int(params.get("stage1_patience", params.get("patience", 10)))
    params["stage2_patience"] = int(params.get("stage2_patience", params.get("patience", 10)))
    params["batch_size"] = int(params.get("batch_size", 5000))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 0.0))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
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


def _run_stage(
    torch,
    model,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    train_groups: dict[int, np.ndarray],
    valid_groups: dict[int, np.ndarray],
    context: ModelContext,
    params: dict[str, Any],
    device,
    *,
    stage: str,
    epochs: int,
    patience: int,
    trainable_parameters: Iterable,
    forward_name: str,
    diagnostics: dict[str, bool],
    window_id: str,
    started_at: float,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        list(trainable_parameters),
        lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        if stage == "stage2_residual":
            model.train_minute_only()
        else:
            model.train()
        date_order = list(train_groups)
        rng = np.random.default_rng(int(params["seed"]) + epoch + (10_000 if stage == "stage2_residual" else 0))
        rng.shuffle(date_order)
        train_loss_sum = 0.0
        train_count = 0

        for trade_date in date_order:
            rows = train_groups[trade_date]
            if len(rows) < 2:
                continue
            for batch_rows in _date_batches(rows, int(params["batch_size"]), rng):
                if len(batch_rows) < 2:
                    continue
                daily_np, minute_np = _multi_bar_tensors(train_data, context.feature_columns, batch_rows, params)
                y_np = train_data.loc[batch_rows, context.label_column].astype("float32").to_numpy()
                daily_tensor = torch.from_numpy(daily_np).to(device)
                minute_tensor = torch.from_numpy(minute_np).to(device)
                y_tensor = torch.from_numpy(y_np).to(device)
                optimizer.zero_grad()
                pred = _forward(model, forward_name, daily_tensor, minute_tensor)
                loss = _negative_ic_loss(torch, pred, y_tensor)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                batch_count = int(len(batch_rows))
                train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                train_count += batch_count

        train_loss = train_loss_sum / train_count if train_count else None
        valid_loss = _eval_stage_loss(torch, model, valid_data, valid_groups, context, params, device, forward_name)
        score_loss = valid_loss if valid_loss is not None else train_loss
        if score_loss is None:
            raise ValueError(f"ResidualMultiBarGRUAlphaModel could not compute finite {stage} IC loss")

        is_best = score_loss + 1e-12 < best_loss
        if is_best:
            best_loss = float(score_loss)
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            stale_epochs = 0
        elif valid_groups:
            stale_epochs += 1

        row = {
            "window_id": window_id,
            "stage": stage,
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_loss": best_loss,
            "is_best": is_best,
            "stale_epochs": stale_epochs,
            "elapsed_seconds": time.perf_counter() - started_at,
            "device": str(device),
            "model_class": "ResidualMultiBarGRUAlphaModel",
        }
        history.append(row)

        if diagnostics["enabled"] and diagnostics["print_epoch"]:
            train_text = "nan" if train_loss is None else f"{train_loss:.6g}"
            valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
            print(
                f"window={window_id} stage={stage} epoch={epoch} train_loss={train_text} "
                f"valid_loss={valid_text} best={best_loss:.6g} patience={stale_epochs}/{patience}",
                flush=True,
            )

        if valid_groups and patience > 0 and stale_epochs >= patience:
            break

    return {
        "best_state": best_state,
        "best_loss": None if best_loss == float("inf") else best_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def _forward(model, forward_name: str, daily_tensor, minute_tensor):
    if forward_name == "forward_daily":
        return model.forward_daily(daily_tensor)
    if forward_name == "forward_residual":
        return model.forward_residual(daily_tensor, minute_tensor)
    raise ValueError(f"unsupported forward_name: {forward_name}")


def _eval_stage_loss(torch, model, frame, groups, context, params, device, forward_name: str) -> float | None:
    if frame.empty or not groups:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < 2:
                continue
            daily_np, minute_np = _multi_bar_tensors(frame, context.feature_columns, rows, params)
            y_np = frame.loc[rows, context.label_column].astype("float32").to_numpy()
            pred = _forward(
                model,
                forward_name,
                torch.from_numpy(daily_np).to(device),
                torch.from_numpy(minute_np).to(device),
            )
            target = torch.from_numpy(y_np).to(device)
            loss = _negative_ic_loss(torch, pred, target)
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return None
    return float(np.mean(losses))
