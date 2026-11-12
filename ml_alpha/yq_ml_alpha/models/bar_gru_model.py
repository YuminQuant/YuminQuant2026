from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from yq_ml_alpha.models.bar_panel_model import (
    BarPanelAlphaModel,
    _bar_tensor as _bar_tensor,
    _bar_tensor_batch as _bar_tensor_batch,
    _bar_tensor_from_array as _bar_tensor_from_array,
    _cpu_state_dict as _cpu_state_dict,
    _date_batches as _date_batches,
    _date_groups as _date_groups,
    _device as _device,
    _diagnostics as _diagnostics,
    _negative_ic_loss,
    _optional_bar_bundle_tensor as _optional_bar_bundle_tensor,
    _release_torch_memory as _release_torch_memory,
    _require_bar_bundle_tensor as _require_bar_bundle_tensor,
    _set_seed as _set_seed,
    _torch_modules as _torch_modules,
    normalize_bar_params as _params,
)
from yq_ml_alpha.models.bar_training import evaluate_bar_panel, evaluate_date_batches, fit_bar_panel
from yq_ml_alpha.models.base import ModelContext


class BarGRUAlphaModel(BarPanelAlphaModel):
    """End-to-end bar panel GRU trained with date-wise negative IC loss."""

    def _normalize_params(self, raw: dict[str, Any]) -> dict[str, Any]:
        return _params(raw)

    def _build_regressor(self, nn):
        return _BarGRURegressor(nn, self.params)

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
            missing_loss_error="BarGRUAlphaModel could not compute any finite train or valid IC loss",
        )

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


def _eval_date_loss(torch, model, frame, groups, context, params, device, tensor: np.ndarray | None = None) -> float | None:
    return evaluate_bar_panel(torch, model, frame, groups, context, params, device,
                              tensor=tensor, loss_fn=_negative_ic_loss)


def _eval_batched_date_loss(torch, model, frame, groups, context, params, predict_rows) -> float | None:
    return evaluate_date_batches(torch, model, frame, groups, context, params, predict_rows,
                                 loss_fn=_negative_ic_loss)
