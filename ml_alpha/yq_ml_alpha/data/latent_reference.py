"""Versioned, detached latent reference snapshots retained only in process RAM."""
from __future__ import annotations

import numpy as np


def reference_refresh_due(epoch, warmup_epochs, refill_epochs):
    age = epoch - warmup_epochs - 1
    return age >= 0 and age % refill_epochs == 0


class LatentReferenceBank:
    def __init__(self, capacity, dimension):
        if type(capacity) is not int or capacity < 2 or type(dimension) is not int or dimension < 1:
            raise ValueError("latent reference capacity/dimension must be positive; capacity >= 2")
        self.capacity, self.dimension = capacity, dimension
        self.values = np.empty((0, dimension), dtype="float32")
        self.version = None

    def replace(self, values, *, version):
        values = np.array(values, dtype="float32", copy=True)
        if (values.ndim != 2 or values.shape[1] != self.dimension or len(values) > self.capacity
                or len(values) < 2 or not np.isfinite(values).all()):
            raise ValueError("invalid latent reference snapshot")
        values.setflags(write=False)
        self.values, self.version = values, int(version)

    def sample(self, count, rng):
        if type(count) is not int or count < 2 or count > len(self.values):
            raise ValueError("latent references must be sampled without replacement within snapshot capacity")
        return self.values[rng.choice(len(self.values), size=count, replace=False)].copy()
