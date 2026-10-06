"""Bounded, append-only tensor assembly; the bundle owns the mapped files."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


class TensorSpool:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="tensor-", dir=root)
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.rows: dict[str, int] = {}
        self.maps: dict[str, np.memmap] = {}

    def append(self, tensors: dict[str, np.ndarray]) -> None:
        if self.shapes and set(tensors) != set(self.shapes):
            raise ValueError("tensor branches changed within a bundle")
        for key, values in tensors.items():
            if not key.isidentifier():
                raise ValueError(f"invalid tensor branch: {key}")
            shape = values.shape[1:]
            if key in self.shapes and self.shapes[key] != shape:
                raise ValueError(f"tensor shape changed for {key}")
            self.shapes[key] = shape
            self.rows[key] = self.rows.get(key, 0) + len(values)
            with (Path(self.directory.name) / key).open("ab") as file:
                np.asarray(values, dtype="float32").tofile(file)

    def finish(self) -> dict[str, np.ndarray]:
        for key, shape in self.shapes.items():
            full_shape = (self.rows[key], *shape)
            self.maps[key] = (
                np.memmap(Path(self.directory.name) / key, dtype="float32", mode="r", shape=full_shape)
                if self.rows[key] else np.empty(full_shape, dtype="float32")
            )
        return self.maps

    def close(self) -> None:
        for array in self.maps.values():
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.maps.clear()
        self.directory.cleanup()

    def __del__(self):
        if hasattr(self, "maps"):
            self.close()
