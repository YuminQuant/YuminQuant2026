from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext


class eLSTMCell:
    def __new__(cls, nn, input_size: int, hidden_size: int, eps: float = 1e-8):
        class Cell(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                self.eps = eps
                self.x_i = nn.Linear(input_size, hidden_size)
                self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_f = nn.Linear(input_size, hidden_size)
                self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_o = nn.Linear(input_size, hidden_size)
                self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)
                self.x_g = nn.Linear(input_size, hidden_size)
                self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)

            def forward(self, x_t, state):
                h_prev, c_prev, n_prev, m_prev = state
                i_pre = self.x_i(x_t) + self.h_i(h_prev)
                f_pre = self.x_f(x_t) + self.h_f(h_prev)
                o_pre = self.x_o(x_t) + self.h_o(h_prev)
                g_pre = self.x_g(x_t) + self.h_g(h_prev)

                m_t = torch.maximum(f_pre + m_prev, i_pre).detach()
                i_t = torch.exp(i_pre - m_t)
                f_t = torch.exp(f_pre + m_prev - m_t)
                o_t = torch.sigmoid(o_pre)
                g_t = torch.tanh(g_pre)

                c_t = f_t * c_prev + i_t * g_t
                n_t = f_t * n_prev + i_t
                h_t = o_t * (c_t / (n_t + self.eps))
                return h_t, (h_t, c_t, n_t, m_t)

        import torch

        return Cell()


class eLSTMLayer:
    def __new__(cls, nn, input_size: int, hidden_size: int, batch_first: bool = True, eps: float = 1e-8):
        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not batch_first:
                    raise ValueError("eLSTMLayer only supports batch_first=True")
                self.hidden_size = hidden_size
                self.cell = eLSTMCell(nn, input_size, hidden_size, eps)

            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                h = x.new_zeros(batch_size, self.hidden_size)
                c = x.new_zeros(batch_size, self.hidden_size)
                n = x.new_zeros(batch_size, self.hidden_size)
                m = x.new_zeros(batch_size, self.hidden_size)
                outputs = []
                state = (h, c, n, m)
                for step in range(seq_len):
                    h, state = self.cell(x[:, step, :], state)
                    outputs.append(h)
                return torch.stack(outputs, dim=1), h

        import torch

        return Layer()


class eLSTM:
    def __new__(
        cls,
        nn,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
        eps: float = 1e-8,
    ):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if not batch_first:
                    raise ValueError("eLSTM only supports batch_first=True")
                self.layers = nn.ModuleList(
                    [
                        eLSTMLayer(
                            nn,
                            input_size if idx == 0 else hidden_size,
                            hidden_size,
                            batch_first=True,
                            eps=eps,
                        )
                        for idx in range(num_layers)
                    ]
                )
                self.dropout = nn.Dropout(dropout) if dropout > 0.0 and num_layers > 1 else None

            def forward(self, x):
                output = x
                h_last = None
                for idx, layer in enumerate(self.layers):
                    output, h_last = layer(output)
                    if self.dropout is not None and idx + 1 < len(self.layers):
                        output = self.dropout(output)
                return output, h_last

        return Model()


class eLSTMRankNetAlphaModel(AlphaModel):
    def __init__(self) -> None:
        self.model = None
        self.input_size: int | None = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        torch, nn, _ = _torch_modules()
        self.params = _params(context.model_params)
        diagnostics = _diagnostics(context)
        _set_seed(torch, int(self.params["seed"]))
        rng = np.random.default_rng(int(self.params["seed"]))
        device = _device(torch, str(self.params["device"]))
        window_id = context.artifact_dir.name

        x_train, self.input_size = _sequence_features(
            train_data, context.feature_columns, int(self.params["sequence_length"])
        )
        y_train = train_data[context.label_column].astype("float32").to_numpy()
        train_dates = train_data["trade_date"].astype("int64").to_numpy()
        train_groups = _date_groups(train_dates)
        self.model = _eLSTMRankNetRegressor(nn, self.input_size, self.params).to(device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train)
        x_valid_tensor = y_valid_tensor = None
        valid_groups: list[np.ndarray] = []
        if not valid_data.empty and context.label_column in valid_data.columns:
            x_valid, _ = _sequence_features(valid_data, context.feature_columns, int(self.params["sequence_length"]))
            y_valid = valid_data[context.label_column].astype("float32").to_numpy()
            valid_dates = valid_data["trade_date"].astype("int64").to_numpy()
            x_valid_tensor = torch.from_numpy(x_valid)
            y_valid_tensor = torch.from_numpy(y_valid)
            valid_groups = _date_groups(valid_dates)

        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])
        sigma = float(self.params["sigma"])
        max_pairs = int(self.params["max_pairs_per_date"])
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        self.loss_history = []
        started_at = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for group_idx in rng.permutation(len(train_groups)):
                idx = torch.as_tensor(train_groups[int(group_idx)], dtype=torch.long)
                x_batch = x_train_tensor.index_select(0, idx).to(device)
                y_batch = y_train_tensor.index_select(0, idx).to(device)
                optimizer.zero_grad()
                pred = self.model(x_batch)
                loss = ranknet_loss_one_date(pred, y_batch, sigma=sigma, max_pairs=max_pairs)
                if loss is None:
                    continue
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
            if not train_losses:
                raise RuntimeError("eLSTM RankNet training found no valid same-date target pairs")

            train_loss = float(np.mean(train_losses))
            valid_loss = None
            if x_valid_tensor is not None and y_valid_tensor is not None and valid_groups:
                valid_loss = _eval_ranknet_loss(
                    torch,
                    self.model,
                    x_valid_tensor,
                    y_valid_tensor,
                    valid_groups,
                    device,
                    sigma,
                    max_pairs,
                )
            is_best = valid_loss is not None and valid_loss + 1e-12 < best_loss
            if is_best:
                best_loss = valid_loss
                best_epoch = epoch
                best_state = _cpu_state_dict(self.model)
                stale_epochs = 0
            elif valid_loss is not None:
                stale_epochs += 1

            row = {
                "window_id": window_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "best_loss": None if best_loss == float("inf") else best_loss,
                "is_best": is_best,
                "stale_epochs": stale_epochs,
                "elapsed_seconds": time.perf_counter() - started_at,
                "device": str(device),
            }
            self.loss_history.append(row)
            if diagnostics["enabled"] and diagnostics["print_epoch"]:
                valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
                best_text = "nan" if best_loss == float("inf") else f"{best_loss:.6g}"
                print(
                    f"window={window_id} epoch={epoch} train_ranknet={train_loss:.6g} "
                    f"valid_ranknet={valid_text} best={best_text} patience={stale_epochs}/{patience}",
                    flush=True,
                )
            if valid_loss is not None and patience > 0 and stale_epochs >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.to("cpu")
        self.model_info = {
            "window_id": window_id,
            "model_class": self.__class__.__name__,
            "alpha_id": context.alpha_id,
            "input_size": self.input_size,
            "feature_count": len(context.feature_columns),
            "train_rows": int(len(train_data)),
            "valid_rows": int(len(valid_data)),
            "train_dates": int(len(train_groups)),
            "valid_dates": int(len(valid_groups)),
            "epochs_run": len(self.loss_history),
            "best_epoch": best_epoch,
            "best_loss": None if best_loss == float("inf") else best_loss,
            "device": str(device),
            "hidden_size": int(self.params["hidden_size"]),
            "num_layers": int(self.params["num_layers"]),
            "dropout": float(self.params["dropout"]),
            "lr": float(self.params["lr"]),
            "weight_decay": float(self.params["weight_decay"]),
            "max_pairs_per_date": int(self.params["max_pairs_per_date"]),
            "sigma": float(self.params["sigma"]),
        }

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        torch, _, _ = _torch_modules()
        self.model.eval()
        x, _ = _sequence_features(data, context.feature_columns, int(self.params["sequence_length"]))
        batch_size = int(self.params["batch_size"])
        scores = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.from_numpy(x[start : start + batch_size])
                scores.append(self.model(batch).detach().cpu().numpy())
        score = np.concatenate(scores).reshape(-1) if scores else np.array([], dtype="float32")
        return pd.Series(score.astype("float32", copy=False), index=data.index, dtype="float32")

    def save(self, path: str | Path) -> None:
        if self.model is None or self.input_size is None:
            raise RuntimeError("model is not fitted")
        torch, _, _ = _torch_modules()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu")
        torch.save(
            {
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
    def load(cls, path: str | Path) -> "eLSTMRankNetAlphaModel":
        torch, nn, _ = _torch_modules()
        checkpoint = torch.load(Path(path), map_location="cpu")
        model = cls()
        model.input_size = int(checkpoint["input_size"])
        model.params = _params(checkpoint["params"])
        model.model = _eLSTMRankNetRegressor(nn, model.input_size, model.params)
        model.model.load_state_dict(checkpoint["state_dict"])
        model.model.eval()
        model.loss_history = list(checkpoint.get("loss_history", []))
        model.model_info = dict(checkpoint.get("model_info", {}))
        return model


def ranknet_loss(pred, target, date_id, sigma: float = 1.0, max_pairs_per_date: int = 20000):
    import torch

    losses = []
    for date in torch.unique(date_id):
        mask = date_id == date
        loss = ranknet_loss_one_date(pred[mask], target[mask], sigma=sigma, max_pairs=max_pairs_per_date)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def ranknet_loss_one_date(pred, target, sigma: float = 1.0, max_pairs: int = 20000):
    import torch
    import torch.nn.functional as F

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    valid = torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    n = int(target.shape[0])
    if n < 2:
        return None
    total_pairs = n * (n - 1) // 2
    if max_pairs <= 0 or total_pairs <= max_pairs:
        left, right = torch.triu_indices(n, n, offset=1, device=target.device)
        target_diff = target[left] - target[right]
        valid_pair = target_diff != 0
        if not bool(valid_pair.any()):
            return None
        left = left[valid_pair]
        right = right[valid_pair]
        target_diff = target_diff[valid_pair]
        winner = torch.where(target_diff > 0, left, right)
        loser = torch.where(target_diff > 0, right, left)
    else:
        winner, loser = _sample_rank_pairs(target, max_pairs)
        if winner is None or loser is None:
            return None
    diff = pred[winner] - pred[loser]
    return F.softplus(-float(sigma) * diff).mean()


def _sample_rank_pairs(target, max_pairs: int):
    import torch

    winners = []
    losers = []
    remaining = int(max_pairs)
    n = int(target.shape[0])
    for _ in range(12):
        draw = max(remaining * 3, max_pairs)
        left = torch.randint(0, n, (draw,), device=target.device)
        right = torch.randint(0, n, (draw,), device=target.device)
        target_diff = target[left] - target[right]
        valid_pair = target_diff != 0
        if not bool(valid_pair.any()):
            continue
        left = left[valid_pair]
        right = right[valid_pair]
        target_diff = target_diff[valid_pair]
        winner = torch.where(target_diff > 0, left, right)
        loser = torch.where(target_diff > 0, right, left)
        winners.append(winner[:remaining])
        losers.append(loser[:remaining])
        remaining -= int(winners[-1].shape[0])
        if remaining <= 0:
            break
    if not winners:
        return None, None
    return torch.cat(winners)[:max_pairs], torch.cat(losers)[:max_pairs]


def _eLSTMRankNetRegressor(nn, input_size: int, params: dict[str, Any]):
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.elstm = eLSTM(
                nn,
                input_size=input_size,
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                batch_first=True,
                eps=float(params["eps"]),
            )
            self.head = nn.Linear(int(params["hidden_size"]), 1)

        def forward(self, x):
            _, h_last = self.elstm(x)
            return self.head(h_last).reshape(-1)

    return Model()


def _params(raw: dict[str, Any]) -> dict[str, Any]:
    params = dict(raw)
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
    params["max_pairs_per_date"] = int(params.get("max_pairs_per_date", 20000))
    params["sigma"] = float(params.get("sigma", 1.0))
    params["eps"] = float(params.get("eps", 1e-8))
    if params["sequence_length"] <= 0:
        raise ValueError("sequence_length must be positive")
    if params["hidden_size"] <= 0:
        raise ValueError("hidden_size must be positive")
    if params["num_layers"] <= 0:
        raise ValueError("num_layers must be positive")
    return params


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


def _date_groups(dates: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(dates == date) for date in pd.unique(dates)]


def _eval_ranknet_loss(torch, model, x_tensor, y_tensor, groups, device, sigma: float, max_pairs: int) -> float | None:
    model.eval()
    losses = []
    with torch.no_grad():
        for group in groups:
            idx = torch.as_tensor(group, dtype=torch.long)
            x_batch = x_tensor.index_select(0, idx).to(device)
            y_batch = y_tensor.index_select(0, idx).to(device)
            loss = ranknet_loss_one_date(model(x_batch), y_batch, sigma=sigma, max_pairs=max_pairs)
            if loss is not None:
                losses.append(float(loss.detach().cpu().item()))
    if not losses:
        return None
    return float(np.mean(losses))


def _diagnostics(context: ModelContext) -> dict[str, bool]:
    raw = dict(context.diagnostics or {})
    enabled = bool(raw.get("enabled", False))
    return {
        "enabled": enabled,
        "print_epoch": enabled and bool(raw.get("print_epoch", False)),
        "write_loss_history": enabled and bool(raw.get("write_loss_history", False)),
        "write_model_info": enabled and bool(raw.get("write_model_info", False)),
    }


def _torch_modules():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("eLSTMRankNetAlphaModel requires installing the optional torch package") from exc
    return torch, nn, F


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
