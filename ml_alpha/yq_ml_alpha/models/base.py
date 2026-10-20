from __future__ import annotations

import pickle
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelContext:
    run_id: str
    alpha_id: str
    feature_columns: list[str]
    label_column: str
    artifact_dir: Path
    model_params: dict[str, Any]
    model_search: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class AlphaModel:
    @contextmanager
    def prediction_session(self):
        """Keep device resources alive across date batches of one model window."""
        previous = getattr(self, "_prediction_active", False)
        self._prediction_active = True
        try:
            yield self
        finally:
            self._prediction_active = previous
            if not previous:
                for name in ("model", "_prediction_network"):
                    module = getattr(self, name, None)
                    if callable(getattr(module, "to", None)):
                        module.to("cpu")
                for module in getattr(self, "encoders", {}).values():
                    if callable(getattr(module, "to", None)):
                        module.to("cpu")
                self.__dict__.pop("_prediction_network", None)
                self.__dict__.pop("_prediction_state_index", None)
                # Do not import/initialize Torch for sklearn-only workflows.
                import sys
                torch = sys.modules.get("torch")
                if torch is not None and torch.cuda.is_initialized():
                    torch.cuda.empty_cache()

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        raise NotImplementedError

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    def write_diagnostics(self, context: ModelContext) -> list[Path]:
        return []

    @classmethod
    def load(cls, path: str | Path) -> "AlphaModel":
        with Path(path).open("rb") as file:
            return pickle.load(file)

class MeanFeatureAlphaModel(AlphaModel):
    """Dependency-free smoke-test model that scores by row-wise feature mean."""

    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        return None

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        values = data[context.feature_columns].replace([np.inf, -np.inf], np.nan)
        return values.mean(axis=1).fillna(0.0).astype("float32")
