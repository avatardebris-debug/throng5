"""
entropy_monitor.py — Detect and prevent entropy collapse in policy and world model.

Monitors:
  - Policy entropy (action distribution diversity)
  - Q-value entropy (how spread are the Q-values?)
  - Surprise trend (from SurpriseTracker)

When collapse is detected (one action dominates or WM overfits):
  - Override epsilon to force exploration
  - Optionally inject noise into World Model weights
  - Signal stochastic mode to other brain modules

Usage:
    monitor = EntropyMonitor(n_actions=18)
    monitor.record_action(action=3, q_values=[0.1, 0.3, 0.8, 0.2])

    if monitor.is_collapsed:
        epsilon = monitor.get_epsilon_override()

    if monitor.should_inject_noise:
        monitor.inject_noise(world_model)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)


def _shannon_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    probs = probs + 1e-8  # Avoid log(0)
    return float(-np.sum(probs * np.log(probs)))


class EntropyMonitor:
    """
    Monitor policy entropy and prevent collapse.

    Entropy collapse = agent picks the same action repeatedly.
    This kills exploration and prevents learning from new states.

    The monitor tracks:
      - Action distribution entropy over a rolling window
      - Q-value spread (how decisive is the policy?)
      - Surprise trend correlation

    When collapse is detected, it overrides epsilon-greedy parameters
    to force exploration, and optionally perturbs the world model
    weights to prevent overfitting.
    """

    def __init__(
        self,
        n_actions: int = 18,
        window: int = 200,
        collapse_threshold: float = 0.5,
        noise_magnitude: float = 0.01,
        check_interval: int = 100,
    ):
        self.n_actions = n_actions
        self.window = window
        self.collapse_threshold = collapse_threshold
        self.noise_magnitude = noise_magnitude
        self.check_interval = check_interval

        # Rolling action history
        self._action_history: deque = deque(maxlen=window)
        self._q_entropy_history: deque = deque(maxlen=window)

        # State
        self._total_actions: int = 0
        self._collapse_count: int = 0       # Times collapse was detected
        self._noise_injections: int = 0     # Times noise was injected
        self._stochastic_mode: bool = False  # Currently in forced exploration
        self._stochastic_steps_left: int = 0

        # External signals
        self._surprise_trend: str = "warmup"

    def record_action(
        self,
        action: int,
        q_values: Optional[np.ndarray] = None,
    ) -> None:
        """Record an action taken and optionally its Q-values."""
        self._action_history.append(action)
        self._total_actions += 1

        if q_values is not None:
            q_arr = np.asarray(q_values, dtype=np.float32)
            probs = _softmax(q_arr)
            self._q_entropy_history.append(_shannon_entropy(probs))

        # Decay stochastic mode
        if self._stochastic_steps_left > 0:
            self._stochastic_steps_left -= 1
            if self._stochastic_steps_left == 0:
                self._stochastic_mode = False

    def set_surprise_trend(self, trend: str) -> None:
        """Update from SurpriseTracker: 'improving', 'plateau', 'degrading'."""
        self._surprise_trend = trend

    @property
    def policy_entropy(self) -> float:
        """Shannon entropy of recent action distribution."""
        if len(self._action_history) < 10:
            return float('inf')  # Not enough data
        counts = np.bincount(
            list(self._action_history),
            minlength=self.n_actions,
        )
        probs = counts / counts.sum()
        return _shannon_entropy(probs)

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy (uniform distribution)."""
        return float(np.log(self.n_actions))

    @property
    def entropy_ratio(self) -> float:
        """Policy entropy / max entropy. 1.0 = uniform, 0.0 = deterministic."""
        pe = self.policy_entropy
        if pe == float('inf'):
            return 1.0
        return min(pe / self.max_entropy, 1.0)

    @property
    def dominant_action(self) -> Optional[int]:
        """Which action dominates, if any."""
        if len(self._action_history) < 10:
            return None
        counts = np.bincount(
            list(self._action_history),
            minlength=self.n_actions,
        )
        max_pct = counts.max() / counts.sum()
        if max_pct > 0.5:  # One action > 50% of choices
            return int(counts.argmax())
        return None

    @property
    def is_collapsed(self) -> bool:
        """True if policy entropy is dangerously low."""
        return self.policy_entropy < self.collapse_threshold

    @property
    def should_inject_noise(self) -> bool:
        """True if plateau + collapse → inject WM noise."""
        return (
            self.is_collapsed
            and self._surprise_trend in ("plateau", "degrading")
        )

    def get_epsilon_override(self) -> Optional[float]:
        """
        Dynamic epsilon override based on entropy state.

        Returns None if no override needed (use default schedule).
        """
        if self._stochastic_mode:
            return 0.5  # Heavy exploration in stochastic mode

        if self.is_collapsed:
            self._collapse_count += 1
            if self._surprise_trend == "plateau":
                # Collapsed + plateau = trigger stochastic mode
                self._stochastic_mode = True
                self._stochastic_steps_left = 200  # 200 steps of heavy exploration
                return 0.5
            return 0.3  # Force moderate exploration

        if self._surprise_trend == "plateau" and self.entropy_ratio < 0.3:
            return 0.15  # Nudge exploration

        return None  # No override

    def inject_noise(self, world_model, magnitude: Optional[float] = None) -> int:
        """
        Slightly corrupt WM weights to escape local minima.

        Returns number of weight matrices perturbed.
        """
        mag = magnitude or self.noise_magnitude
        perturbed = 0

        for attr in ['w1', 'w2', 'wr1', 'wr2', 'b1', 'b2', 'br1', 'br2']:
            w = getattr(world_model, attr, None)
            if w is not None and isinstance(w, np.ndarray):
                w += np.random.randn(*w.shape).astype(w.dtype) * mag
                perturbed += 1

        self._noise_injections += 1
        return perturbed

    def report(self) -> Dict[str, Any]:
        return {
            "total_actions": self._total_actions,
            "policy_entropy": round(self.policy_entropy, 4) if self.policy_entropy != float('inf') else None,
            "entropy_ratio": round(self.entropy_ratio, 4),
            "is_collapsed": self.is_collapsed,
            "stochastic_mode": self._stochastic_mode,
            "stochastic_steps_left": self._stochastic_steps_left,
            "dominant_action": self.dominant_action,
            "collapse_count": self._collapse_count,
            "noise_injections": self._noise_injections,
            "surprise_trend": self._surprise_trend,
            "avg_q_entropy": (
                round(float(np.mean(list(self._q_entropy_history))), 4)
                if self._q_entropy_history else None
            ),
        }
