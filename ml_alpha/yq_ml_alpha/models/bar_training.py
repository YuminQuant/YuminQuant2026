"""Date-batched training and validation for bar-panel networks."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.bar_panel_model import (
    BarPanelAlphaModel, _bar_tensor_batch, _date_batches, _date_groups,
    _negative_ic_loss, normalize_bar_params,
)
from yq_ml_alpha.models.base import ModelContext
from yq_ml_alpha.models.torch_support import (
    _cpu_state_dict, _device, _diagnostics, _release_torch_memory, _set_seed, _torch_modules,
)


class DatewiseICBarAlphaModel(BarPanelAlphaModel):
    """Bar panel estimator trained and selected by negative date-wise Pearson IC."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return normalize_bar_params(raw)

    def _training_loss(self, torch, inputs, target):
        return _negative_ic_loss(torch, self.model(inputs), target)

    def _fit(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        context: ModelContext,
        *,
        train_tensor: np.ndarray | None,
        valid_tensor: np.ndarray | None,
    ) -> None:
        fit_bar_panel(
            self, train_data, valid_data, context,
            train_tensor=train_tensor, valid_tensor=valid_tensor,
            validation_loss=_negative_ic_loss, loss_name="negative_datewise_pearson_ic",
            train_loss_log_key="bar_gru_train_loss",
            missing_loss_error="Bar panel model could not compute any finite train or validation IC loss",
        )


def fit_bar_panel(
    alpha, train_data, valid_data, context, *,
    train_tensor, valid_tensor, validation_loss, loss_name, train_loss_log_key,
    missing_loss_error,
):
    torch, nn = _torch_modules()
    alpha.params = alpha._normalize_params(context.model_params)
    diagnostics = _diagnostics(context)
    _set_seed(torch, int(alpha.params["seed"]))
    device = _device(torch, str(alpha.params["device"]))
    window_id = context.artifact_dir.name

    if train_data.empty:
        raise ValueError(f"{alpha.__class__.__name__} requires non-empty train_data")

    alpha.model = alpha._build_regressor(nn).to(device)
    optimizer = torch.optim.Adam(
        alpha.model.parameters(),
        lr=float(alpha.params["lr"]),
        weight_decay=float(alpha.params["weight_decay"]),
    )

    train_groups = _date_groups(train_data)
    valid_groups = _date_groups(valid_data) if not valid_data.empty else {}
    epochs = int(alpha.params["epochs"])
    patience = int(alpha.params["patience"])
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    alpha.loss_history = []
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        alpha.model.train()
        date_order = list(train_groups)
        rng = np.random.default_rng(int(alpha.params["seed"]) + epoch)
        rng.shuffle(date_order)
        train_loss_sum = 0.0
        train_count = 0

        for trade_date in date_order:
            rows = train_groups[trade_date]
            if len(rows) < 2:
                continue
            for batch_rows in _date_batches(rows, int(alpha.params["batch_size"]), rng):
                if len(batch_rows) < 2:
                    continue
                x_tensor = y_tensor = loss = None
                try:
                    x_np = _bar_tensor_batch(train_data, context.feature_columns, batch_rows, alpha.params, train_tensor)
                    y_np = train_data.loc[batch_rows, context.label_column].astype("float32").to_numpy()
                    x_tensor = torch.from_numpy(x_np).to(device)
                    y_tensor = torch.from_numpy(y_np).to(device)
                    optimizer.zero_grad()
                    loss = alpha._training_loss(torch, x_tensor, y_tensor)
                    if loss is None:
                        continue
                    loss.backward()
                    optimizer.step()
                    batch_count = int(len(batch_rows))
                    train_loss_sum += float(loss.detach().cpu().item()) * batch_count
                    train_count += batch_count
                finally:
                    del x_tensor, y_tensor, loss

        train_loss = train_loss_sum / train_count if train_count else None
        valid_loss = evaluate_bar_panel(
            torch,
            alpha.model,
            valid_data,
            valid_groups,
            context,
            alpha.params,
            device,
            tensor=valid_tensor,
            loss_fn=validation_loss,
        )
        score_loss = valid_loss if valid_loss is not None else train_loss
        if score_loss is None:
            raise ValueError(missing_loss_error)

        is_best = score_loss + 1e-12 < best_loss
        if is_best:
            best_loss = float(score_loss)
            best_epoch = epoch
            best_state = _cpu_state_dict(alpha.model)
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
            "model_class": alpha.__class__.__name__,
        }
        alpha.loss_history.append(row)

        if diagnostics["enabled"] and diagnostics["print_epoch"]:
            train_text = "nan" if train_loss is None else f"{train_loss:.6g}"
            valid_text = "nan" if valid_loss is None else f"{valid_loss:.6g}"
            print(
                f"window={window_id} epoch={epoch} {train_loss_log_key}={train_text} "
                f"valid_loss={valid_text} best={best_loss:.6g} patience={stale_epochs}/{patience}",
                flush=True,
            )

        if valid_groups and patience > 0 and stale_epochs >= patience:
            break

    if best_state is not None:
        alpha.model.load_state_dict(best_state)
    alpha.model.to("cpu")
    _release_torch_memory(torch)
    alpha.model_info = {
        "window_id": window_id,
        "model_class": alpha.__class__.__name__,
        "alpha_id": context.alpha_id,
        "feature_count": len(context.feature_columns),
        "train_rows": int(len(train_data)),
        "valid_rows": int(len(valid_data)),
        "train_dates": int(len(train_groups)),
        "valid_dates": int(len(valid_groups)),
        "epochs_run": len(alpha.loss_history),
        "best_epoch": best_epoch,
        "best_loss": None if best_loss == float("inf") else best_loss,
        "device": str(device),
        "sequence_length": int(alpha.params["sequence_length"]),
        "input_size": int(alpha.params["input_size"]),
        "hidden_size": int(alpha.params["hidden_size"]),
        "num_layers": int(alpha.params["num_layers"]),
        "batch_size": int(alpha.params["batch_size"]),
        "epochs": int(alpha.params["epochs"]),
        "patience": int(alpha.params["patience"]),
        "lr": float(alpha.params["lr"]),
        "weight_decay": float(alpha.params["weight_decay"]),
        "loss": loss_name,
    }


def evaluate_bar_panel(
    torch, model, frame, groups, context, params, device, tensor: np.ndarray | None = None, *, loss_fn,
) -> float | None:
    def predict_rows(rows):
        x_np = _bar_tensor_batch(frame, context.feature_columns, rows, params, tensor)
        return model(torch.from_numpy(x_np).to(device))

    return evaluate_date_batches(torch, model, frame, groups, context, params, predict_rows, loss_fn=loss_fn)


def evaluate_date_batches(torch, model, frame, groups, context, params, predict_rows, *, loss_fn) -> float | None:
    """Bound forward-pass memory while evaluating each full date."""
    if frame.empty or not groups:
        return None
    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.no_grad():
            for rows in groups.values():
                if len(rows) < 2:
                    continue
                predictions = []
                for start in range(0, len(rows), int(params["batch_size"])):
                    batch_rows = rows[start : start + int(params["batch_size"])]
                    predictions.append(predict_rows(batch_rows).detach().cpu())
                pred = torch.cat(predictions)
                y_np = frame.loc[rows, context.label_column].astype("float32").to_numpy()
                loss = loss_fn(torch, pred, torch.from_numpy(y_np))
                if loss is not None:
                    losses.append(float(loss.item()))
    finally:
        model.train(was_training)
    if not losses:
        return None
    return float(np.mean(losses))


def evaluate_datewise_ic(torch, model, frame, groups, context, params, predict_rows) -> float | None:
    return evaluate_date_batches(torch, model, frame, groups, context, params, predict_rows,
                                 loss_fn=_negative_ic_loss)
