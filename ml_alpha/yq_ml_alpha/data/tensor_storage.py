"""Bounded, append-only tensor assembly; the bundle owns the mapped files."""
from __future__ import annotations

import tempfile
import os
import threading
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


class MemoryTensorSpool:
    """Remote training retains one window in RAM and never spills to disk."""
    _guard = threading.Lock()
    _reserved = 0
    def __init__(self):
        self.chunks = {}
        self.maps = {}
        self.bytes = 0
        self.limit = int(os.environ.get('YQ_TENSOR_MEMORY_BYTES', str(2 * 1024**3)))

    def _reserve(self, cost):
        with self._guard:
            if type(self)._reserved + cost > self.limit:
                raise MemoryError('Concurrent training tensors exceed the configured RAM budget; reduce the window or increase memory')
            type(self)._reserved += cost

    def _release(self, cost):
        with self._guard:
            type(self)._reserved -= cost

    def append(self, tensors):
        if self.chunks and set(tensors) != set(self.chunks):
            raise ValueError('tensor branches changed within a bundle')
        cost = sum(np.asarray(v).size * 4 for v in tensors.values())
        for key, values in tensors.items():
            previous=self.chunks.get(key,[])
            if previous and previous[0].shape[1:] != values.shape[1:]:
                raise ValueError('tensor shape changed within a bundle')
        self._reserve(cost)
        self.bytes += cost
        for key, values in tensors.items():
            array = np.asarray(values, dtype='float32')
            previous = self.chunks.setdefault(key, [])
            previous.append(array)

    def finish(self):
        for key in list(self.chunks):
            chunks = self.chunks[key]
            cost = sum(v.nbytes for v in chunks)
            extra=cost if len(chunks)>1 else 0
            self._reserve(extra)
            try:
                self.maps[key] = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
                chunks.clear()
            finally:
                self._release(extra)
            del self.chunks[key]
        return self.maps

    def close(self):
        self.chunks.clear()
        self.maps.clear()
        self._release(self.bytes)
        self.bytes = 0

    def __del__(self):
        if hasattr(self, 'bytes'):
            self.close()
