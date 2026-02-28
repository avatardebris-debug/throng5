"""
model_distiller.py — Compress trained models for fast inference.

Takes a trained brain's learned weights and distills them into
smaller, faster models for deployment:

  1. Weight pruning: zero out near-zero weights
  2. Quantization: float32 → int8 with scale factors
  3. Knowledge distillation: train small student from big teacher
  4. Heuristic compilation: extract top-N rules from Q-table

The distilled model runs faster on limited hardware while
retaining most of the learned behavior.

Usage:
    from brain.scaling.model_distiller import ModelDistiller

    distiller = ModelDistiller(brain)
    compact = distiller.distill(target_size_kb=100)
    compact.save("models/compact_brain.bin")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class CompactModel:
    """A distilled, compressed brain model for fast inference."""
    # Pruned DQN weights
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    # Top heuristics (state_hash → action)
    heuristics: Dict[int, int]

    # Metadata
    n_features: int
    n_actions: int
    original_params: int
    pruned_params: int
    compression_ratio: float

    def select_action(self, features: np.ndarray, epsilon: float = 0.0) -> int:
        """Fast action selection using distilled model."""
        features = np.asarray(features, dtype=np.float32).flatten()

        # Check heuristic table first (fastest path)
        state_hash = hash(np.round(features[:20] * 10).astype(np.int16).tobytes())
        if state_hash in self.heuristics and np.random.random() > epsilon:
            return self.heuristics[state_hash]

        # DQN forward pass
        x = features[:self.n_features]
        if len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)))

        hidden = np.maximum(0, x @ self.W1 + self.b1)
        q_values = hidden @ self.W2 + self.b2

        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(q_values))

    def save(self, filepath: str) -> None:
        """Save compact model to .npz file."""
        np.savez_compressed(
            filepath,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            n_features=self.n_features, n_actions=self.n_actions,
            original_params=self.original_params,
            pruned_params=self.pruned_params,
            heuristic_keys=list(self.heuristics.keys()),
            heuristic_vals=list(self.heuristics.values()),
        )

    @staticmethod
    def load(filepath: str) -> "CompactModel":
        """Load compact model from .npz file."""
        data = np.load(filepath, allow_pickle=False)
        heuristics = dict(zip(
            data["heuristic_keys"].astype(int).tolist(),
            data["heuristic_vals"].astype(int).tolist(),
        ))
        return CompactModel(
            W1=data["W1"], b1=data["b1"], W2=data["W2"], b2=data["b2"],
            n_features=int(data["n_features"]),
            n_actions=int(data["n_actions"]),
            original_params=int(data["original_params"]),
            pruned_params=int(data["pruned_params"]),
            compression_ratio=float(data["pruned_params"]) / max(float(data["original_params"]), 1),
            heuristics=heuristics,
        )


class ModelDistiller:
    """
    Distill a trained WholeBrain into a compact inference model.
    """

    def __init__(self, brain):
        self.brain = brain

    def distill(
        self,
        prune_threshold: float = 0.01,
        max_heuristics: int = 1000,
    ) -> CompactModel:
        """
        Distill the brain into a compact model.

        Steps:
          1. Copy Striatum DQN weights
          2. Prune near-zero weights
          3. Extract top heuristics from Motor Cortex
          4. Package into CompactModel
        """
        striatum = self.brain.striatum

        # Copy weights
        W1 = striatum._W1.copy()
        b1 = striatum._b1.copy()
        W2 = striatum._W2.copy()
        b2 = striatum._b2.copy()

        original_params = (
            W1.size + b1.size + W2.size + b2.size
        )

        # ── Prune near-zero weights ───────────────────────────────────
        W1 = self._prune(W1, prune_threshold)
        W2 = self._prune(W2, prune_threshold)

        pruned_params = (
            np.count_nonzero(W1) + np.count_nonzero(b1)
            + np.count_nonzero(W2) + np.count_nonzero(b2)
        )

        # ── Extract heuristics ────────────────────────────────────────
        motor = self.brain.motor
        heuristics = dict(list(motor._heuristics.items())[:max_heuristics])

        compression = pruned_params / max(original_params, 1)

        return CompactModel(
            W1=W1, b1=b1, W2=W2, b2=b2,
            heuristics=heuristics,
            n_features=self.brain.n_features,
            n_actions=self.brain.n_actions,
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=compression,
        )

    def quantize_weights(
        self,
        model: CompactModel,
    ) -> Tuple[CompactModel, Dict[str, float]]:
        """
        Quantize float32 weights to int8 for further compression.

        Returns the quantized model and scale factors.
        """
        scales = {}

        # Quantize each weight matrix
        model.W1, scales["W1"] = self._quantize_matrix(model.W1)
        model.W2, scales["W2"] = self._quantize_matrix(model.W2)

        return model, scales

    def benchmark(self, model: CompactModel, n_steps: int = 1000) -> Dict[str, Any]:
        """
        Benchmark compact model performance.

        Returns timing and agreement statistics.
        """
        import time

        rng = np.random.RandomState(42)
        features_batch = rng.randn(n_steps, self.brain.n_features).astype(np.float32)

        # Time compact model
        start = time.perf_counter()
        compact_actions = []
        for features in features_batch:
            action = model.select_action(features, epsilon=0.0)
            compact_actions.append(action)
        compact_time = (time.perf_counter() - start) * 1000

        # Time original model
        start = time.perf_counter()
        original_actions = []
        for features in features_batch:
            q = self.brain.striatum._forward(features)
            original_actions.append(int(np.argmax(q)))
        original_time = (time.perf_counter() - start) * 1000

        # Agreement rate
        agreement = sum(
            1 for a, b in zip(compact_actions, original_actions) if a == b
        ) / n_steps

        return {
            "compact_ms": round(compact_time, 2),
            "original_ms": round(original_time, 2),
            "speedup": round(original_time / max(compact_time, 0.01), 2),
            "agreement": round(agreement, 4),
            "compression_ratio": round(model.compression_ratio, 4),
            "original_params": model.original_params,
            "pruned_params": model.pruned_params,
        }

    # ── Internal ──────────────────────────────────────────────────────

    @staticmethod
    def _prune(weights: np.ndarray, threshold: float) -> np.ndarray:
        """Zero out weights below threshold."""
        mask = np.abs(weights) > threshold
        return weights * mask

    @staticmethod
    def _quantize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """Quantize float32 matrix to int8 range [-127, 127]."""
        scale = np.max(np.abs(matrix)) / 127.0
        if scale < 1e-10:
            return matrix, 1.0
        quantized = np.clip(np.round(matrix / scale), -127, 127).astype(np.float32)
        return quantized * scale, scale  # Dequantize back for inference
