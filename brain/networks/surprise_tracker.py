"""
surprise_tracker.py — Track prediction-vs-reality delta for the World Model.

Wraps the WorldModel to:
  1. Record prediction errors (surprise) for every transition
  2. Maintain a prioritized buffer of most surprising transitions
  3. Extra-train the WM on biggest surprises (prioritized replay)
  4. Track surprise trends to auto-tune Dyna-Q weighting
  5. Detect plateau (WM not improving) → signal for stochastic mode

Usage:
    tracker = SurpriseTracker(world_model)
    surprise = tracker.predict_and_compare(state, action, actual_next, actual_reward)
    trend = tracker.surprise_trend()     # "improving", "plateau", "degrading"
    tracker.extra_train(n_batches=5)     # Train on most surprising transitions
    weight = tracker.dyna_weight         # Auto Dyna-Q weight [0.0, 0.5]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SurpriseRecord:
    """A single prediction-vs-reality comparison."""
    state: np.ndarray
    action: int
    actual_next: np.ndarray
    actual_reward: float
    surprise: float                # |predicted - actual| magnitude
    state_delta: float             # Per-feature state prediction error
    reward_delta: float            # Reward prediction error
    step: int = 0


class PrioritizedSurpriseBuffer:
    """
    Ring buffer that maintains access to highest-surprise transitions.

    Uses a simple sorted insertion approach — not heap-optimal but
    sufficient for buffer sizes ≤ 50k.
    """

    def __init__(self, max_size: int = 10000):
        self._buffer: List[SurpriseRecord] = []
        self._max_size = max_size
        self._total_added = 0

    def add(self, record: SurpriseRecord) -> None:
        self._buffer.append(record)
        self._total_added += 1
        if len(self._buffer) > self._max_size:
            # Remove least surprising record
            min_idx = min(range(len(self._buffer)),
                         key=lambda i: self._buffer[i].surprise)
            self._buffer.pop(min_idx)

    def sample_highest(self, k: int = 32) -> List[SurpriseRecord]:
        """Sample the k most surprising transitions."""
        if not self._buffer:
            return []
        sorted_buf = sorted(self._buffer, key=lambda r: -r.surprise)
        return sorted_buf[:min(k, len(sorted_buf))]

    def sample_random(self, k: int = 32) -> List[SurpriseRecord]:
        """Random sample from buffer."""
        if not self._buffer:
            return []
        k = min(k, len(self._buffer))
        indices = np.random.choice(len(self._buffer), size=k, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class SurpriseTracker:
    """
    Track prediction error over time and learn from biggest surprises.

    Core feedback loop:
      predict → compare to reality → record delta → learn from worst surprises
      → track trend → adjust confidence → gate Dyna-Q weight

    Surprise = mean(|predicted_state - actual_state|) + |predicted_reward - actual_reward|
    """

    def __init__(
        self,
        world_model,
        buffer_size: int = 10000,
        trend_window: int = 500,
        extra_train_interval: int = 50,
    ):
        self.wm = world_model
        self.buffer = PrioritizedSurpriseBuffer(buffer_size)
        self._surprise_history: deque = deque(maxlen=trend_window)
        self._trend_window = trend_window
        self._extra_train_interval = extra_train_interval

        # Running stats
        self._total_comparisons: int = 0
        self._total_extra_trains: int = 0
        self._ema_surprise: float = 1.0  # Exponential moving average
        self._ema_alpha: float = 0.01    # EMA decay rate

        # Per-feature surprise tracking (which features are hardest to predict)
        self._feature_surprise_sum: Optional[np.ndarray] = None
        self._feature_surprise_count: int = 0

    def predict_and_compare(
        self,
        state: np.ndarray,
        action: int,
        actual_next: np.ndarray,
        actual_reward: float,
        step: int = 0,
    ) -> float:
        """
        Predict next state/reward, compare to reality, record delta.

        Returns:
            surprise: scalar surprise magnitude
        """
        if self.wm is None:
            return 0.0

        try:
            pred_state, pred_reward = self.wm.predict(state, action)
        except Exception:
            return 0.0

        # Compute deltas
        pred_flat = np.asarray(pred_state, dtype=np.float32).flatten()
        actual_flat = np.asarray(actual_next, dtype=np.float32).flatten()

        min_len = min(len(pred_flat), len(actual_flat))
        feature_delta = np.abs(pred_flat[:min_len] - actual_flat[:min_len])
        state_delta = float(np.mean(feature_delta))
        reward_delta = abs(float(pred_reward) - actual_reward)

        surprise = state_delta + reward_delta

        # Record
        record = SurpriseRecord(
            state=state.copy(),
            action=action,
            actual_next=actual_next.copy(),
            actual_reward=actual_reward,
            surprise=surprise,
            state_delta=state_delta,
            reward_delta=reward_delta,
            step=step,
        )
        self.buffer.add(record)
        self._surprise_history.append(surprise)
        self._total_comparisons += 1

        # Update EMA
        self._ema_surprise = (
            (1 - self._ema_alpha) * self._ema_surprise
            + self._ema_alpha * surprise
        )

        # Per-feature tracking
        if self._feature_surprise_sum is None:
            self._feature_surprise_sum = np.zeros(min_len, dtype=np.float32)
        if len(feature_delta) == len(self._feature_surprise_sum):
            self._feature_surprise_sum += feature_delta
            self._feature_surprise_count += 1

        return surprise

    def surprise_trend(self) -> str:
        """
        Is surprise decreasing (WM improving), flat (plateau), or rising?

        Returns:
            "warmup"    — not enough data yet
            "improving" — surprise decreasing (WM learning)
            "plateau"   — surprise stable (WM stagnant)
            "degrading" — surprise increasing (WM getting worse)
        """
        history = self._surprise_history
        if len(history) < 100:
            return "warmup"

        recent = list(history)
        half = len(recent) // 2
        older_avg = float(np.mean(recent[:half]))
        newer_avg = float(np.mean(recent[half:]))

        if older_avg < 1e-8:
            return "plateau"

        ratio = newer_avg / older_avg
        if ratio < 0.85:
            return "improving"
        elif ratio > 1.15:
            return "degrading"
        return "plateau"

    def extra_train(self, n_batches: int = 5, batch_size: int = 32) -> int:
        """
        Train WM on most surprising transitions (prioritized replay).

        Returns number of training updates performed.
        """
        if self.wm is None:
            return 0

        updates = 0
        for _ in range(n_batches):
            # Mix: 75% highest surprise, 25% random (diversity)
            high_k = int(batch_size * 0.75)
            rand_k = batch_size - high_k

            batch = self.buffer.sample_highest(high_k)
            batch.extend(self.buffer.sample_random(rand_k))

            for record in batch:
                try:
                    self.wm.update(
                        record.state,
                        record.action,
                        record.actual_next,
                        record.actual_reward,
                    )
                    updates += 1
                except Exception:
                    pass

        self._total_extra_trains += 1
        return updates

    @property
    def avg_surprise(self) -> float:
        """Average surprise over recent window."""
        if not self._surprise_history:
            return float('inf')
        return float(np.mean(list(self._surprise_history)))

    @property
    def dyna_weight(self) -> float:
        """
        Auto-tune Dyna-Q synthetic step weight based on WM accuracy.

        Returns weight in [0.0, 0.5]:
          - 0.0: WM too inaccurate, don't use synthetic transitions
          - 0.1: Low confidence, light synthetic weighting
          - 0.3: Moderate confidence
          - 0.5: High confidence, trust synthetic steps equally
        """
        avg = self.avg_surprise
        if avg == float('inf') or avg > 2.0:
            return 0.0
        if avg > 1.0:
            return 0.05
        if avg > 0.5:
            return 0.1
        if avg > 0.2:
            return 0.2
        if avg > 0.1:
            return 0.3
        return 0.5

    @property
    def hardest_features(self) -> Optional[np.ndarray]:
        """
        Which features are hardest to predict? Indices sorted by difficulty.

        Useful for diagnosing what the WM struggles with.
        """
        if self._feature_surprise_sum is None or self._feature_surprise_count == 0:
            return None
        avg = self._feature_surprise_sum / self._feature_surprise_count
        return np.argsort(-avg)  # Descending order of difficulty

    def report(self) -> Dict[str, Any]:
        return {
            "total_comparisons": self._total_comparisons,
            "total_extra_trains": self._total_extra_trains,
            "avg_surprise": round(self.avg_surprise, 4) if self.avg_surprise != float('inf') else None,
            "ema_surprise": round(self._ema_surprise, 4),
            "surprise_trend": self.surprise_trend(),
            "dyna_weight": self.dyna_weight,
            "buffer_size": len(self.buffer),
            "hardest_features_top5": (
                self.hardest_features[:5].tolist()
                if self.hardest_features is not None else None
            ),
        }
