"""Torch runtime helpers, diagnostics and shared model state."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from yq_ml_alpha.models.base import AlphaModel, ModelContext
from yq_ml_alpha.output import artifact_io


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
    except ImportError as exc:  # pragma: no cover - depends on optional local package
        raise ImportError("PyTorch alpha models require installing the optional torch package") from exc
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


class TorchAlphaModel(AlphaModel):
    """Model state and diagnostics shared by PyTorch estimators."""

    def __init__(self) -> None:
        self.model = None
        self.input_dim: int | None = None
        self.params: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []
        self.model_info: dict[str, Any] = {}

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
