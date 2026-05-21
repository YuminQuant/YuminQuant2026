from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class LogsigOrthogonalMLPAlphaModel(AlphaModel):
    """MLP that turns signature features into decorrelated base factors."""

    def __init__(self) -> None:
        self.model = None
        self.input_dim: int | None = None
        self.params: dict[str, Any] = {}
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        if train_data.empty:
            raise ValueError("LogsigOrthogonalMLPAlphaModel requires non-empty train_data")
        torch, nn = _torch_modules()
        self.params = _params(context.model_params)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        x_train_raw = _features(train_data, context.feature_columns)
        self.feature_mean, self.feature_std = _fit_standardizer(x_train_raw)
        x_train = _apply_standardizer(x_train_raw, self.feature_mean, self.feature_std)
        self.input_dim = x_train.shape[1]
        self.model = _build_network(nn, self.input_dim, self.params).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        train_groups = _date_groups(train_data)
        valid_groups = _date_groups(valid_data) if not valid_data.empty else {}
        x_valid = None
        if valid_groups:
            x_valid = _apply_standardizer(
                _features(valid_data, context.feature_columns),
                self.feature_mean,
                self.feature_std,
            )

        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        started_at = time.perf_counter()
        self.loss_history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            rng = np.random.default_rng(int(self.params["seed"]) + epoch)
            dates = list(train_groups)
            rng.shuffle(dates)
            loss_sum = 0.0
            loss_count = 0
            for trade_date in dates:
                rows = train_groups[trade_date]
                if len(rows) < 2:
                    continue
                for batch_rows in _date_batches(rows, int(self.params["batch_size"]), rng):
                    if len(batch_rows) < 2:
                        continue
                    y_np = train_data.iloc[batch_rows][context.label_column].astype("float32").to_numpy()
                    x_tensor = torch.from_numpy(x_train[batch_rows]).to(device)
                    y_tensor = torch.from_numpy(y_np).to(device)
                    optimizer.zero_grad()
                    base = self.model(x_tensor)
                    loss = _orthogonal_ic_loss(torch, base, y_tensor, float(self.params["orthogonal_lambda"]))
                    if loss is None:
                        continue
                    loss.backward()
                    optimizer.step()
                    count = int(len(batch_rows))
                    loss_sum += float(loss.detach().cpu().item()) * count
                    loss_count += count

            train_loss = loss_sum / loss_count if loss_count else None
            valid_loss = _eval_date_loss(
                torch,
                self.model,
                x_valid,
                valid_data,
                valid_groups,
                context,
                device,
                float(self.params["orthogonal_lambda"]),
            )
            score_loss = valid_loss if valid_loss is not None else train_loss
            if score_loss is None:
                raise ValueError("could not compute a finite Logsig orthogonal MLP loss")
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
                    f"window={window_id} epoch={epoch} logsig_train_loss={train_text} "
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
            "input_dim": self.input_dim,
            "feature_count": len(context.feature_columns),
            "base_factors": int(self.params["base_factors"]),
            "orthogonal_lambda": float(self.params["orthogonal_lambda"]),
            "train_rows": int(len(train_data)),
            "valid_rows": int(len(valid_data)),
            "train_dates": int(len(train_groups)),
            "valid_dates": int(len(valid_groups)),
            "epochs_run": len(self.loss_history),
            "best_epoch": best_epoch,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "device": str(device),
            "hidden_layers": list(self.params["hidden_layers"]),
            "batch_size": int(self.params["batch_size"]),
            "loss": "negative_datewise_ic_plus_base_factor_correlation_penalty",
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("model is not fitted")
        torch, _ = _torch_modules()
        device = _device(torch, str(self.params.get("device", "auto")))
        self.model.to(device)
        self.model.eval()
        x = _apply_standardizer(_features(data, context.feature_columns), self.feature_mean, self.feature_std)
        scores = np.empty(len(data), dtype="float32")
        row_index = data.index.to_numpy()
        with torch.no_grad():
            for start in range(0, len(row_index), int(self.params["batch_size"])):
                end = start + int(self.params["batch_size"])
                base = self.model(torch.from_numpy(x[start:end]).to(device))
                composite = _composite_factor(torch, base).detach().cpu().numpy().astype("float32")
                scores[start : start + len(composite)] = composite
        self.model.to("cpu")
        series = pd.Series(scores, index=data.index, dtype="float32")
        return series.groupby(data["trade_date"], group_keys=False).transform(_zscore_series).astype("float32")

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
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
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
    def load(cls, path: str | Path) -> "LogsigOrthogonalMLPAlphaModel":
        torch, nn = _torch_modules()
        checkpoint = _torch_load(torch, Path(path))
        model = cls()
        model.input_dim = int(checkpoint["input_dim"])
        model.params = _params(checkpoint["params"])
        model.feature_mean = checkpoint["feature_mean"]
        model.feature_std = checkpoint["feature_std"]
        model.model = _build_network(nn, model.input_dim, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
    hidden_layers = params.get("hidden_layers", [256, 128])
    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    params["hidden_layers"] = [int(value) for value in hidden_layers]
    params["base_factors"] = int(params.get("base_factors", 8))
    params["orthogonal_lambda"] = float(params.get("orthogonal_lambda", 0.05))
    params["dropout"] = float(params.get("dropout", 0.10))
    params["epochs"] = int(params.get("epochs", 50))
    params["batch_size"] = int(params.get("batch_size", 8192))
    params["lr"] = float(params.get("lr", 1e-3))
    params["weight_decay"] = float(params.get("weight_decay", 1e-5))
    params["seed"] = int(params.get("seed", 42))
    params["device"] = str(params.get("device", "auto"))
    params["patience"] = int(params.get("patience", 10))
    if params["base_factors"] <= 0:
        raise ValueError("base_factors must be positive")
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
    current = input_dim
    for hidden in params["hidden_layers"]:
        layers.append(nn.Linear(current, hidden))
        layers.append(nn.ReLU())
        if float(params["dropout"]) > 0.0:
            layers.append(nn.Dropout(float(params["dropout"])))
        current = hidden
    layers.append(nn.Linear(current, int(params["base_factors"])))
    return nn.Sequential(*layers)


def _features(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame[columns].replace([np.inf, -np.inf], np.nan).astype("float32").to_numpy()


def _fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype("float32")
    mean = np.where(np.isfinite(mean), mean, 0.0).astype("float32")
    centered = values - mean
    std = np.nanstd(centered, axis=0).astype("float32")
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype("float32")
    return mean, std


def _apply_standardizer(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    scaled = (values - mean) / std
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _date_groups(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    if frame.empty:
        return {}
    positions = pd.Series(np.arange(len(frame), dtype="int64"), index=frame.index)
    return {
        int(date): rows.to_numpy(dtype="int64")
        for date, rows in positions.groupby(frame["trade_date"], sort=True)
    }


def _date_batches(rows: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    if len(rows) <= batch_size:
        return [rows]
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    return [shuffled[idx : idx + batch_size] for idx in range(0, len(shuffled), batch_size)]


def _orthogonal_ic_loss(torch, base, target, penalty_weight: float):
    if base.shape[0] < 2:
        return None
    valid = torch.isfinite(target.reshape(-1)) & torch.isfinite(base).all(dim=1)
    if int(valid.sum().detach().cpu().item()) < 2:
        return None
    base = base[valid]
    target = target.reshape(-1)[valid]
    composite = _composite_factor(torch, base)
    ic = _pearson_corr(torch, composite, target)
    if ic is None:
        return None
    penalty = _offdiag_corr_l2(torch, _zscore_tensor(torch, base))
    return -ic + penalty_weight * penalty


def _composite_factor(torch, base):
    standardized = _zscore_tensor(torch, base)
    return standardized.mean(dim=1)


def _zscore_tensor(torch, values):
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (values - mean) / std


def _pearson_corr(torch, left, right):
    left = left.reshape(-1)
    right = right.reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.sqrt(torch.sum(left * left)) * torch.sqrt(torch.sum(right * right))
    if float(denom.detach().cpu().item()) <= 1e-8:
        return None
    return torch.sum(left * right) / (denom + 1e-8)


def _offdiag_corr_l2(torch, standardized):
    if standardized.shape[1] <= 1:
        return standardized.sum() * 0.0
    corr = standardized.T @ standardized / max(1, standardized.shape[0])
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    offdiag = corr * (1.0 - eye)
    return torch.sqrt(torch.mean(offdiag * offdiag) + 1e-12)


def _eval_date_loss(torch, model, x_valid, frame, groups, context, device, penalty_weight: float) -> float | None:
    if x_valid is None or frame.empty or not groups:
        return None
    model.eval()
    losses = []
    with torch.no_grad():
        for rows in groups.values():
            if len(rows) < 2:
                continue
            base = model(torch.from_numpy(x_valid[rows]).to(device))
            y = torch.from_numpy(frame.iloc[rows][context.label_column].astype("float32").to_numpy()).to(device)
            loss = _orthogonal_ic_loss(torch, base, y, penalty_weight)
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    model.train()
    if not losses:
        return None
    return float(np.mean(losses))


def _zscore_series(values: pd.Series) -> pd.Series:
    finite = values.replace([np.inf, -np.inf], np.nan)
    std = finite.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.zeros(len(values), dtype="float32"), index=values.index)
    return ((finite - finite.mean()) / std).fillna(0.0).astype("float32")


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LogsigOrthogonalMLPAlphaModel requires installing the optional torch package") from exc
    return torch, nn


def _torch_load(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
