from __future__ import annotations

import pickle
from dataclasses import dataclass
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


class AlphaModel:
    def fit(self, train_data: pd.DataFrame, valid_data: pd.DataFrame, context: ModelContext) -> None:
        raise NotImplementedError

    def predict(self, data: pd.DataFrame, context: ModelContext) -> pd.Series:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

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
