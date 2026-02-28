"""
curiosity.py — Environment-agnostic intrinsic motivation & reward shaping.

Provides curiosity-driven exploration signals that work with ANY environment,
replacing game-specific subgoal systems. Three complementary signals:

  1. Prediction Error (neural): surprise from world model prediction errors
  2. Visit Counts (hash-based): novelty from visiting rare state regions
  3. State Coverage: bonus for expanding the set of distinct states visited

When a WorldModel is connected, prediction error uses the neural model
for richer surprise signals. Otherwise falls back to a simple linear model.

Usage:
    from brain.environments.curiosity import CuriosityModule

    curiosity = CuriosityModule(n_features=84)

    for step in range(total_steps):
        intrinsic_reward = curiosity.compute(features, action, next_features)
        total_reward = extrinsic_reward + intrinsic_reward
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np


class CuriosityModule:
    """
    Environment-agnostic intrinsic motivation.

    Combines three novelty signals into a single intrinsic reward.
    No game-specific knowledge required.

    When connected to the WorldModel, uses neural prediction errors
    instead of a single linear layer — much richer surprise signal.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        prediction_weight: float = 0.4,
        visit_weight: float = 0.3,
        coverage_weight: float = 0.3,
        decay_rate: float = 0.999,
        n_bins: int = 20,  # Resolution for state hashing
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.prediction_weight = prediction_weight
        self.visit_weight = visit_weight
        self.coverage_weight = coverage_weight
        self.decay_rate = decay_rate
        self.n_bins = n_bins

        # ── Neural prediction (via WorldModel) ────────────────────────
        self._world_model = None  # Set via connect_world_model()

        # ── Fallback: 2-layer MLP for prediction ─────────────────────
        # Replaces the old single linear layer with a small MLP
        input_dim = n_features + n_actions
        hidden_dim = 128
        rng = np.random.RandomState(42)
        self._W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self._b1 = np.zeros(hidden_dim, dtype=np.float32)
        self._W2 = rng.randn(hidden_dim, n_features).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self._b2 = np.zeros(n_features, dtype=np.float32)
        self._pred_lr = 0.001
        self._total_pred_error = 0.0
        self._pred_steps = 0

        # ── Visit counts (hash-based) ────────────────────────────────
        self._visit_counts: Dict[int, int] = defaultdict(int)
        self._total_visits = 0

        # ── State coverage ────────────────────────────────────────────
        self._unique_states: set = set()
        self._coverage_bonus_scale = 1.0

        # ── Running statistics for normalization ──────────────────────
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def connect_world_model(self, world_model) -> None:
        """
        Connect a trained WorldModel for richer prediction errors.

        When connected, prediction error uses the neural world model
        instead of the simple MLP fallback. The world model shares
        the same feature space as the CNN encoder and DQN.
        """
        self._world_model = world_model

    def compute(
        self,
        features: np.ndarray,
        action: int,
        next_features: np.ndarray,
    ) -> float:
        """
        Compute intrinsic reward from a state transition.

        Returns a non-negative scalar intrinsic reward.
        """
        features = np.asarray(features, dtype=np.float32).flatten()
        next_features = np.asarray(next_features, dtype=np.float32).flatten()

        # 1. Prediction error (neural or MLP)
        pred_reward = self._prediction_error(features, action, next_features)

        # 2. Visit count novelty
        visit_reward = self._visit_novelty(next_features)

        # 3. Coverage bonus
        coverage_reward = self._coverage_bonus(next_features)

        # Combine
        raw_reward = (
            self.prediction_weight * pred_reward
            + self.visit_weight * visit_reward
            + self.coverage_weight * coverage_reward
        )

        # Normalize to prevent intrinsic reward from dominating
        normalized = self._normalize_reward(raw_reward)

        return max(0.0, normalized)

    # ── Prediction Error ──────────────────────────────────────────────

    def _prediction_error(
        self,
        features: np.ndarray,
        action: int,
        next_features: np.ndarray,
    ) -> float:
        """
        Predict next state and return prediction error as surprise signal.

        Uses WorldModel if connected (neural, multi-layer, shared with dreamer),
        otherwise uses 2-layer MLP fallback.
        """
        # ── WorldModel path (preferred) ──────────────────────────────
        if self._world_model is not None and self._world_model.is_ready:
            predicted_next, _ = self._world_model.predict(features, action)
            target = next_features[:self.n_features]
            if len(target) < self.n_features:
                target = np.pad(target, (0, self.n_features - len(target)))
            error = float(np.mean((predicted_next[:self.n_features] - target) ** 2))
            self._total_pred_error += error
            self._pred_steps += 1
            return error

        # ── 2-layer MLP fallback ─────────────────────────────────────
        return self._mlp_prediction_error(features, action, next_features)

    def _mlp_prediction_error(
        self,
        features: np.ndarray,
        action: int,
        next_features: np.ndarray,
    ) -> float:
        """2-layer MLP prediction with online gradient descent."""
        # Build input: [features; one_hot(action)]
        action_vec = np.zeros(self.n_actions, dtype=np.float32)
        if action < self.n_actions:
            action_vec[action] = 1.0

        x = np.concatenate([features[:self.n_features], action_vec])
        expected_dim = self._W1.shape[0]
        if len(x) < expected_dim:
            x = np.pad(x, (0, expected_dim - len(x)))
        elif len(x) > expected_dim:
            x = x[:expected_dim]

        # Forward: 2-layer MLP with ReLU
        h = x @ self._W1 + self._b1
        h_relu = np.maximum(0, h)
        predicted = h_relu @ self._W2 + self._b2

        target = next_features[:self.n_features]
        if len(target) < self.n_features:
            target = np.pad(target, (0, self.n_features - len(target)))

        # Error
        error_vec = predicted - target
        error = float(np.mean(error_vec ** 2))

        # Backward: gradient descent through both layers
        d_output = 2 * error_vec / self.n_features  # (n_features,)
        grad_W2 = np.outer(h_relu, d_output)         # (hidden, n_features)
        grad_b2 = d_output

        d_hidden = d_output @ self._W2.T              # (hidden,)
        d_hidden = d_hidden * (h > 0).astype(np.float32)  # ReLU gradient
        grad_W1 = np.outer(x, d_hidden)               # (input, hidden)
        grad_b1 = d_hidden

        self._W1 -= self._pred_lr * grad_W1
        self._b1 -= self._pred_lr * grad_b1
        self._W2 -= self._pred_lr * grad_W2
        self._b2 -= self._pred_lr * grad_b2

        self._total_pred_error += error
        self._pred_steps += 1

        return error

    # ── Visit Count Novelty ───────────────────────────────────────────

    def _visit_novelty(self, features: np.ndarray) -> float:
        """Return novelty bonus inversely proportional to visit count."""
        state_hash = self._hash_state(features)
        self._visit_counts[state_hash] += 1
        self._total_visits += 1
        count = self._visit_counts[state_hash]
        return 1.0 / np.sqrt(count)

    # ── Coverage Bonus ────────────────────────────────────────────────

    def _coverage_bonus(self, features: np.ndarray) -> float:
        """Bonus for visiting a never-before-seen state region."""
        state_hash = self._hash_state(features)
        if state_hash not in self._unique_states:
            self._unique_states.add(state_hash)
            # Decaying bonus: less exciting as coverage grows
            return self._coverage_bonus_scale / (1.0 + len(self._unique_states) * 0.001)
        return 0.0

    # ── Reward Normalization ──────────────────────────────────────────

    def _normalize_reward(self, raw: float) -> float:
        """Normalize using running stats to keep scale ~[0, 1]."""
        self._reward_count += 1
        alpha = max(0.01, 1.0 / self._reward_count)
        self._reward_mean = (1 - alpha) * self._reward_mean + alpha * raw
        self._reward_var = (1 - alpha) * self._reward_var + alpha * (raw - self._reward_mean) ** 2
        return (raw - self._reward_mean) / (np.sqrt(self._reward_var) + 1e-8)

    # ── State Hashing ─────────────────────────────────────────────────

    def _hash_state(self, features: np.ndarray) -> int:
        """Quantize features into a discrete hash for counting."""
        quantized = np.clip(features[:min(len(features), 20)], -5, 5)
        bins = np.round(quantized * self.n_bins / 10.0).astype(np.int16)
        return hash(bins.tobytes())

    # ── Reporting ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "unique_states": len(self._unique_states),
            "total_visits": self._total_visits,
            "avg_pred_error": self._total_pred_error / max(1, self._pred_steps),
            "pred_steps": self._pred_steps,
            "visit_buckets": len(self._visit_counts),
            "prediction_source": "world_model" if (
                self._world_model is not None and self._world_model.is_ready
            ) else "mlp_fallback",
        }

    def reset_episode(self) -> None:
        """Reset per-episode state (keep learned prediction model)."""
        pass  # Intentionally keep all learned state across episodes
