"""
bottleneck_tracker.py — Detect and prioritize failure hotspots.

Tracks where deaths/failures cluster in state space, identifies the
"frontier" (farthest progress), and flags stuck points for rehearsal
or LLM review.

Usage:
    tracker = BottleneckTracker()
    tracker.record_death(features, context={"reward": -1})
    tracker.record_success(features)
    worst = tracker.get_worst_bottleneck()
    frontier = tracker.get_frontier()
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BottleneckState:
    """Tracks failure/success statistics at a particular state cluster."""
    state_hash: int
    representative_features: Optional[np.ndarray] = None
    deaths: int = 0
    visits: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    last_success_step: int = 0
    last_death_step: int = 0
    total_reward_at_death: float = 0.0
    flagged_for_review: bool = False
    contexts: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def failure_rate(self) -> float:
        return self.deaths / max(self.visits, 1)

    @property
    def is_stuck(self) -> bool:
        return self.consecutive_failures >= 10

    def severity_score(self, global_step: int) -> float:
        """Higher = more urgent to rehearse."""
        score = self.failure_rate * 10.0
        # Recency bonus: recent deaths are more urgent
        recency = 1.0 / (1.0 + (global_step - self.last_death_step) / 1000)
        score *= (1.0 + recency)
        # Stuck multiplier
        if self.is_stuck:
            score *= 2.0
        return score


class BottleneckTracker:
    """
    Identifies where in the game the agent consistently fails.

    Hashes states into coarse clusters and tracks death/success rates
    per cluster. Maintains a "frontier" (farthest unique progress).
    """

    def __init__(
        self,
        n_buckets: int = 256,
        stuck_threshold: int = 10,
    ):
        self._n_buckets = n_buckets
        self._stuck_threshold = stuck_threshold
        self._states: Dict[int, BottleneckState] = {}
        self._global_step: int = 0

        # Frontier tracking: ordered list of state hashes encountered
        self._frontier_sequence: List[int] = []
        self._farthest_step: int = 0
        self._farthest_hash: int = 0

        # Stuck points for LLM/review
        self._stuck_points: List[Dict[str, Any]] = []

        # Running normalization
        self._running_mean = np.zeros(8, dtype=np.float64)
        self._running_var = np.ones(8, dtype=np.float64)
        self._n_seen: int = 0

    def _hash_state(self, features: np.ndarray) -> int:
        """Coarse state hash for clustering."""
        features = np.asarray(features, dtype=np.float32).flatten()
        k = min(8, len(features))
        top_k = features[:k]

        # Update running stats
        self._n_seen += 1
        alpha = min(0.01, 1.0 / self._n_seen)
        self._running_mean[:k] += alpha * (top_k.astype(np.float64) - self._running_mean[:k])
        diff = top_k.astype(np.float64) - self._running_mean[:k]
        self._running_var[:k] += alpha * (diff ** 2 - self._running_var[:k])

        std = np.sqrt(self._running_var[:k] + 1e-8)
        normalized = (top_k - self._running_mean[:k].astype(np.float32)) / std.astype(np.float32)
        bins = np.clip(((normalized + 3) / 6 * 8).astype(int), 0, 7)
        return hash(bins.tobytes()) % self._n_buckets

    def record_death(
        self,
        features: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record a death at this state. Returns the state hash.
        """
        self._global_step += 1
        h = self._hash_state(features)

        if h not in self._states:
            self._states[h] = BottleneckState(
                state_hash=h,
                representative_features=np.asarray(features, dtype=np.float32).copy(),
            )

        state = self._states[h]
        state.deaths += 1
        state.visits += 1
        state.consecutive_failures += 1
        state.max_consecutive_failures = max(
            state.max_consecutive_failures, state.consecutive_failures,
        )
        state.last_death_step = self._global_step
        if context:
            state.contexts.append(context)
            state.total_reward_at_death += context.get("episode_reward", 0)

        # Check if newly stuck
        if state.consecutive_failures >= self._stuck_threshold and not state.flagged_for_review:
            state.flagged_for_review = True
            self._stuck_points.append({
                "state_hash": h,
                "deaths": state.deaths,
                "failure_rate": round(state.failure_rate, 3),
                "step": self._global_step,
                "contexts": list(state.contexts),
            })

        return h

    def record_success(self, features: np.ndarray) -> int:
        """Record a successful passage through this state."""
        self._global_step += 1
        h = self._hash_state(features)

        if h not in self._states:
            self._states[h] = BottleneckState(
                state_hash=h,
                representative_features=np.asarray(features, dtype=np.float32).copy(),
            )

        state = self._states[h]
        state.visits += 1
        state.consecutive_failures = 0
        state.last_success_step = self._global_step

        # Track frontier progression
        if self._global_step > self._farthest_step:
            self._farthest_step = self._global_step
            self._farthest_hash = h
            if h not in self._frontier_sequence:
                self._frontier_sequence.append(h)

        return h

    def get_worst_bottleneck(self) -> Optional[BottleneckState]:
        """Get the highest-severity bottleneck state."""
        if not self._states:
            return None
        return max(
            self._states.values(),
            key=lambda s: s.severity_score(self._global_step),
        )

    def get_frontier(self) -> Optional[BottleneckState]:
        """Get the farthest state reached."""
        if self._farthest_hash in self._states:
            return self._states[self._farthest_hash]
        return None

    def get_flanking_states(self, state_hash: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Get state hashes just before and just after the given state
        in the frontier sequence.
        """
        if state_hash not in self._frontier_sequence:
            return None, None
        idx = self._frontier_sequence.index(state_hash)
        before = self._frontier_sequence[idx - 1] if idx > 0 else None
        after = self._frontier_sequence[idx + 1] if idx < len(self._frontier_sequence) - 1 else None
        return before, after

    def is_stuck(self, state_hash: int) -> bool:
        """Check if we're stuck at this state (10+ consecutive failures)."""
        if state_hash in self._states:
            return self._states[state_hash].is_stuck
        return False

    def get_stuck_points(self) -> List[Dict[str, Any]]:
        """Get all stuck points for LLM review."""
        return self._stuck_points.copy()

    def get_state_features(self, state_hash: int) -> Optional[np.ndarray]:
        """Get representative features for a state hash."""
        if state_hash in self._states:
            return self._states[state_hash].representative_features
        return None

    def report(self) -> Dict[str, Any]:
        active = [s for s in self._states.values() if s.deaths > 0]
        return {
            "total_states_tracked": len(self._states),
            "active_bottlenecks": len(active),
            "stuck_points": len(self._stuck_points),
            "farthest_frontier": self._farthest_step,
            "frontier_length": len(self._frontier_sequence),
            "worst": self.get_worst_bottleneck().state_hash if active else None,
            "top_5_severity": [
                {"hash": s.state_hash, "deaths": s.deaths, "rate": round(s.failure_rate, 3)}
                for s in sorted(active, key=lambda s: -s.severity_score(self._global_step))[:5]
            ],
        }
