"""
sensor_fusion.py — Merge CNN visual features with RAM semantic features.

CNN provides: spatial understanding, texture recognition, pixel patterns
RAM provides: precise entity positions, exact game state, inventory

Fused representation is strictly better than either alone.

Architecture:
    CNN features (128-dim) ─┐
                             ├─ concat ─► FC(256) ─► ReLU ─► FC(128) ─► fused
    RAM features (32-dim)  ─┘

Usage:
    fusion = SensorFusion(cnn_dim=128, ram_dim=32, output_dim=128)
    fused = fusion.fuse(cnn_features, ram_features)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class SensorFusion:
    """
    Merges CNN visual features with RAM semantic features.

    Simple numpy-based fusion (no PyTorch dependency) using learned
    linear projection with ReLU. The weights are initialized randomly
    and can be trained alongside the main DQN.
    """

    def __init__(
        self,
        cnn_dim: int = 128,
        ram_dim: int = 32,
        output_dim: int = 128,
        hidden_dim: int = 256,
    ):
        self._cnn_dim = cnn_dim
        self._ram_dim = ram_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim

        # Initialize weights (Xavier init)
        input_dim = cnn_dim + ram_dim
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self._w1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self._b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._w2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * scale2
        self._b2 = np.zeros(output_dim, dtype=np.float32)

        # Running normalization for RAM features
        self._ram_mean = np.zeros(ram_dim, dtype=np.float64)
        self._ram_var = np.ones(ram_dim, dtype=np.float64)
        self._n_seen: int = 0

        # Stats
        self._total_fusions: int = 0

    def _normalize_ram(self, ram_features: np.ndarray) -> np.ndarray:
        """Running normalization of RAM features."""
        self._n_seen += 1
        alpha = min(0.01, 1.0 / self._n_seen)
        f = ram_features.astype(np.float64)
        self._ram_mean += alpha * (f[:len(self._ram_mean)] - self._ram_mean[:len(f)])
        diff = f[:len(self._ram_var)] - self._ram_mean[:len(f)]
        self._ram_var[:len(f)] += alpha * (diff ** 2 - self._ram_var[:len(f)])

        std = np.sqrt(self._ram_var[:len(f)] + 1e-8)
        return ((f[:len(self._ram_mean)] - self._ram_mean[:len(f)]) / std).astype(np.float32)

    def extract_ram_features(
        self,
        ram: np.ndarray,
        mapper=None,
    ) -> np.ndarray:
        """
        Extract a compact feature vector from raw RAM + semantic mapper.

        Extracts the most meaningful bytes (positions, state flags,
        counters) into a fixed-size vector.
        """
        ram = np.asarray(ram, dtype=np.uint8).flatten()
        features = np.zeros(self._ram_dim, dtype=np.float32)
        idx = 0

        if mapper is not None:
            # Position bytes
            for entry in mapper.get_registry().get("position", []):
                if idx >= self._ram_dim:
                    break
                addr = entry["addr"]
                if addr < len(ram):
                    features[idx] = float(ram[addr]) / 255.0
                    idx += 1

            # State flags
            for entry in mapper.get_registry().get("state_flag", []):
                if idx >= self._ram_dim:
                    break
                addr = entry["addr"]
                if addr < len(ram):
                    features[idx] = float(ram[addr]) / 255.0
                    idx += 1

            # Subgoal bytes
            for item in mapper.get_subgoal_bytes():
                if idx >= self._ram_dim:
                    break
                addr = item["addr"]
                if addr < len(ram):
                    features[idx] = float(ram[addr]) / 255.0
                    idx += 1
        else:
            # No mapper: use first N RAM bytes
            n = min(self._ram_dim, len(ram))
            features[:n] = ram[:n].astype(np.float32) / 255.0

        return features

    def fuse(
        self,
        cnn_features: np.ndarray,
        ram_features: np.ndarray,
    ) -> np.ndarray:
        """
        Fuse CNN and RAM features into a single representation.

        Args:
            cnn_features: (cnn_dim,) visual features from CNN
            ram_features: (ram_dim,) semantic features from RAM

        Returns:
            (output_dim,) fused representation
        """
        self._total_fusions += 1

        cnn = np.asarray(cnn_features, dtype=np.float32).flatten()
        ram = self._normalize_ram(np.asarray(ram_features, dtype=np.float32).flatten())

        # Pad/truncate to expected dimensions
        if len(cnn) < self._cnn_dim:
            cnn = np.pad(cnn, (0, self._cnn_dim - len(cnn)))
        cnn = cnn[:self._cnn_dim]

        if len(ram) < self._ram_dim:
            ram = np.pad(ram, (0, self._ram_dim - len(ram)))
        ram = ram[:self._ram_dim]

        # Concat → FC → ReLU → FC
        concat = np.concatenate([cnn, ram])
        hidden = concat @ self._w1 + self._b1
        hidden = np.maximum(hidden, 0)  # ReLU
        output = hidden @ self._w2 + self._b2

        return output

    def fuse_from_raw(
        self,
        cnn_features: np.ndarray,
        ram: np.ndarray,
        mapper=None,
    ) -> np.ndarray:
        """Convenience: extract RAM features and fuse in one call."""
        ram_features = self.extract_ram_features(ram, mapper)
        return self.fuse(cnn_features, ram_features)

    def report(self) -> Dict[str, Any]:
        return {
            "cnn_dim": self._cnn_dim,
            "ram_dim": self._ram_dim,
            "output_dim": self._output_dim,
            "total_fusions": self._total_fusions,
            "n_ram_observations": self._n_seen,
        }
