from __future__ import annotations

from pathlib import Path


def window_artifact_path(artifact_dir: str | Path, window_id: str) -> Path:
    path = Path(artifact_dir) / window_id / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
