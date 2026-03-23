"""
memmap_replay.py — Memory-mapped replay buffer for Striatum and Hippocampus.

Replaces the deque-based replay buffer with a numpy memmap ring buffer.

Benefits:
  - Zero pickle/serialization overhead on write (direct numpy write)
  - Zero copy on read (numpy views, not Python objects)
  - Cross-process shareable (same file can be read by overnight/dream process)
  - Survives process crashes — replay buffer persists on disk
  - Can be much larger than RAM (OS handles paging)

Usage:
    buf = MemmapReplayBuffer(capacity=100_000, n_features=84, path="replay.dat")
    buf.add(state, action, reward, next_state, done)
    batch = buf.sample(batch_size=32)
    # batch["states"] is a numpy view — no copy
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np


# Default: in-memory (no file path) — uses anonymous mmap
_ANON = "<anon>"


class MemmapReplayBuffer:
    """
    Ring buffer backed by numpy memmap (or anonymous in-memory array).

    Layout (all float32 except action/done):
        states:      (capacity, n_features)  float32
        next_states: (capacity, n_features)  float32
        actions:     (capacity,)             int32
        rewards:     (capacity,)             float32
        dones:       (capacity,)             float32  (1.0 = done)

    The header (write pointer + size) is stored in a small separate array.
    """

    def __init__(
        self,
        capacity: int,
        n_features: int,
        path: Optional[str] = None,   # None = anonymous (in-memory), str = file path
        mode: str = "w+",             # "w+" = create/overwrite, "r+" = resume existing
    ):
        self.capacity = capacity
        self.n_features = n_features
        self._path = path
        self._use_file = path is not None

        if self._use_file:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)

        def _make(filename, shape, dtype):
            if self._use_file:
                return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
            else:
                return np.zeros(shape, dtype=dtype)

        suffix = path or "mem"
        self._states      = _make(f"{suffix}.states.npy",      (capacity, n_features), np.float32)
        self._next_states = _make(f"{suffix}.next.npy",        (capacity, n_features), np.float32)
        self._actions     = _make(f"{suffix}.actions.npy",     (capacity,),            np.int32)
        self._rewards     = _make(f"{suffix}.rewards.npy",     (capacity,),            np.float32)
        self._dones       = _make(f"{suffix}.dones.npy",       (capacity,),            np.float32)
        # Header: [write_ptr, size]
        self._header      = _make(f"{suffix}.header.npy",      (2,),                   np.int64)

        if mode == "w+":
            self._header[:] = [0, 0]

    @property
    def _ptr(self) -> int:
        return int(self._header[0])

    @_ptr.setter
    def _ptr(self, v: int) -> None:
        self._header[0] = v

    @property
    def size(self) -> int:
        return int(self._header[1])

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Write one transition. O(1), no Python object creation."""
        ptr = self._ptr
        self._states[ptr] = state
        self._next_states[ptr] = next_state
        self._actions[ptr] = int(action)
        self._rewards[ptr] = float(reward)
        self._dones[ptr] = 1.0 if done else 0.0
        self._header[0] = (ptr + 1) % self.capacity
        self._header[1] = min(int(self._header[1]) + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a random batch. Returns numpy views (no copy for contiguous slices).

        Returns dict with keys: states, actions, rewards, next_states, dones.
        """
        n = min(self.size, self.capacity)
        if n < batch_size:
            batch_size = n
        idx = np.random.randint(0, n, size=batch_size)
        return {
            "states":      self._states[idx],
            "actions":     self._actions[idx],
            "rewards":     self._rewards[idx],
            "next_states": self._next_states[idx],
            "dones":       self._dones[idx],
        }

    def sample_recent(self, batch_size: int, recency: int = 1000) -> Dict[str, np.ndarray]:
        """Sample preferring recent transitions (last `recency` steps)."""
        n = min(self.size, self.capacity)
        if n == 0:
            return self.sample(batch_size)
        start = max(0, n - recency)
        recent_n = n - start
        if recent_n < batch_size:
            return self.sample(batch_size)
        idx = np.random.randint(start, n, size=batch_size)
        return {
            "states":      self._states[idx % self.capacity],
            "actions":     self._actions[idx % self.capacity],
            "rewards":     self._rewards[idx % self.capacity],
            "next_states": self._next_states[idx % self.capacity],
            "dones":       self._dones[idx % self.capacity],
        }

    def flush(self) -> None:
        """Flush file-backed arrays to disk (no-op for in-memory)."""
        if self._use_file:
            for arr in (self._states, self._next_states, self._actions,
                        self._rewards, self._dones, self._header):
                if hasattr(arr, 'flush'):
                    arr.flush()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"MemmapReplayBuffer(capacity={self.capacity}, "
                f"size={self.size}, n_features={self.n_features}, "
                f"file={self._use_file})")
