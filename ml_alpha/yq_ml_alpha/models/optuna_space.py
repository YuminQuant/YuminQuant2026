from __future__ import annotations

from typing import Any


def suggest_params(trial, configured_space: dict[str, Any] | None, default_space: dict[str, Any]) -> dict[str, Any]:
    space = configured_space if isinstance(configured_space, dict) and configured_space else default_space
    return {name: _suggest_one(trial, name, spec) for name, spec in space.items()}


def _suggest_one(trial, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if not isinstance(spec, dict):
        raise ValueError(f"Optuna search space for {name} must be a table or list")
    if "choices" in spec:
        return trial.suggest_categorical(name, list(spec["choices"]))
    kind = str(spec.get("type", "float")).lower()
    if kind == "int":
        kwargs = {}
        if "step" in spec:
            kwargs["step"] = int(spec["step"])
        if "log" in spec:
            kwargs["log"] = bool(spec["log"])
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs)
    if kind == "float":
        kwargs = {}
        if "step" in spec:
            kwargs["step"] = float(spec["step"])
        if "log" in spec:
            kwargs["log"] = bool(spec["log"])
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), **kwargs)
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"unsupported Optuna search space type for {name}: {kind}")
