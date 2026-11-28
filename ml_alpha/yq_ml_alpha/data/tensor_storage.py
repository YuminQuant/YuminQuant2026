"""Append-only tensor assembly; the bundle owns the mapped files."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


class TensorSpool:
    def __init__(self, root: Path) -> None:
        root = root.resolve()
        if root.drive.lower() == "c:":
            raise ValueError("tensor scratch must not be placed on C:")
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


class ChunkedTensor:
    """Read an arbitrary batch from per-date arrays without a full-window copy."""

    def __init__(self, chunks):
        self.chunks = tuple(chunks)
        self.ends = np.cumsum([len(chunk) for chunk in chunks], dtype=np.int64)
        self.starts = np.concatenate(([0], self.ends[:-1]))
        self.shape = (int(self.ends[-1]), *chunks[0].shape[1:])
        self.ndim = len(self.shape)
        self.dtype = chunks[0].dtype

    def __len__(self):
        return self.shape[0]

    @property
    def nbytes(self):
        return sum(chunk.nbytes for chunk in self.chunks)

    def __getitem__(self, rows):
        scalar = np.isscalar(rows)
        if isinstance(rows, slice):
            rows = np.arange(*rows.indices(len(self)))
        rows = np.atleast_1d(rows)
        if rows.dtype == bool:
            if len(rows) != len(self):
                raise IndexError("invalid tensor mask length")
            rows = np.flatnonzero(rows)
        rows = np.where(rows < 0, rows + len(self), rows).astype(np.int64, copy=False)
        if not len(rows):
            return np.empty((0, *self.shape[1:]), dtype=self.dtype)
        if rows.min() < 0 or rows.max() >= len(self):
            raise IndexError("tensor row index out of bounds")
        day_ids = np.searchsorted(self.ends, rows, side="right")
        if day_ids.min() == day_ids.max():
            index = day_ids[0]
            result = self.chunks[index][rows - self.starts[index]]
            return result[0] if scalar else result
        result = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        for index in np.unique(day_ids):
            selected = np.flatnonzero(day_ids == index)
            result[selected] = self.chunks[index][rows[selected] - self.starts[index]]
        return result


class MemoryTensorSpool:
    """Retain prepared date tensors in RAM for all epochs in the window."""

    def __init__(self):
        self.chunks = {}
        self.maps = {}

    def append(self, tensors):
        if self.chunks and set(tensors) != set(self.chunks):
            raise ValueError("tensor branches changed within a bundle")
        for key, values in tensors.items():
            array = np.asarray(values, dtype="float32")
            chunks = self.chunks.setdefault(key, [])
            if chunks and chunks[0].shape[1:] != array.shape[1:]:
                raise ValueError(f"tensor shape changed for {key}")
            chunks.append(array)

    def finish(self):
        for key, chunks in self.chunks.items():
            self.maps[key] = chunks[0] if len(chunks) == 1 else ChunkedTensor(chunks)
        self.chunks.clear()
        return self.maps

    def close(self):
        self.chunks.clear()
        self.maps.clear()
