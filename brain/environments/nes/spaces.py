"""Minimal space stubs when gymnasium is not installed (NES-only paths)."""

from __future__ import annotations

import numpy as np


class Box:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = np.float32


class Discrete:
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(self.n))
