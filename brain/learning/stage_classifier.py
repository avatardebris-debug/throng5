"""
stage_classifier.py — Cluster game states into stages for per-area learner selection.

Detects distinct "areas" of a game (e.g., starting zone, combat zone, exploration zone)
by hashing feature vectors into coarse buckets and tracking transitions.
Each stage gets independent learner performance tracking.

Usage:
    from brain.learning.stage_classifier import StageClassifier

    sc = StageClassifier(n_features=84)
    stage = sc.classify(features)
    sc.record(stage, learner_name="torch_dqn", reward=5.0)
    best = sc.get_best_learner(stage)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StageStats:
    """Performance statistics for a single stage."""
    visits: int = 0
    learner_rewards: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=200)))
    learner_visits: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, learner_name: str, reward: float) -> None:
        self.visits += 1
        self.learner_rewards[learner_name].append(reward)
        self.learner_visits[learner_name] += 1

    def get_best(self) -> Optional[str]:
        """Return the learner with highest mean reward in this stage."""
        best_name = None
        best_mean = -float("inf")
        for name, rewards in self.learner_rewards.items():
            if len(rewards) >= 5:  # Minimum data
                mean = float(np.mean(rewards))
                if mean > best_mean:
                    best_mean = mean
                    best_name = name
        return best_name

    def win_rates(self) -> Dict[str, float]:
        """Return mean reward per learner."""
        result = {}
        for name, rewards in self.learner_rewards.items():
            if rewards:
                result[name] = round(float(np.mean(rewards)), 4)
        return result


class StageClassifier:
    """
    Classifies game states into discrete stages for per-area specialization.

    Method:
    1. Hash features into a coarse bucket (quantize top PCA-like components)
    2. Track stage transitions to detect distinct areas
    3. Maintain per-stage learner performance statistics
    4. Recommend best learner per stage

    Stages are emergent — they form naturally from the state distribution.
    """

    def __init__(
        self,
        n_features: int = 84,
        n_buckets: int = 16,          # Number of distinct stages
        quantize_bits: int = 3,       # Bits per dimension for hashing
        smoothing_window: int = 10,   # Steps before accepting a stage transition
    ):
        self.n_features = n_features
        self.n_buckets = n_buckets
        self._quantize_bits = quantize_bits
        self._smoothing_window = smoothing_window

        # Stage statistics
        self._stages: Dict[int, StageStats] = defaultdict(StageStats)
        self._current_stage: int = 0
        self._stage_history: deque = deque(maxlen=1000)

        # Running mean/std for normalization
        self._running_mean = np.zeros(min(n_features, 8), dtype=np.float64)
        self._running_var = np.ones(min(n_features, 8), dtype=np.float64)
        self._n_seen: int = 0

        # Transition tracking
        self._stage_buffer: deque = deque(maxlen=smoothing_window)
        self._transitions: int = 0

    def classify(self, features: np.ndarray) -> int:
        """
        Classify features into a stage ID.

        Uses the top-8 feature dimensions (highest variance = most discriminative)
        quantized to coarse buckets, then hashed modulo n_buckets.
        """
        features = np.asarray(features, dtype=np.float32).flatten()

        # Use first 8 dims (or fewer) as discriminative features
        k = min(8, len(features))
        top_k = features[:k]

        # Update running statistics
        self._n_seen += 1
        alpha = min(0.01, 1.0 / self._n_seen)
        self._running_mean[:k] += alpha * (top_k.astype(np.float64) - self._running_mean[:k])
        diff = top_k.astype(np.float64) - self._running_mean[:k]
        self._running_var[:k] += alpha * (diff ** 2 - self._running_var[:k])

        # Normalize
        std = np.sqrt(self._running_var[:k] + 1e-8)
        normalized = (top_k - self._running_mean[:k].astype(np.float32)) / std.astype(np.float32)

        # Quantize to integer bins
        bins = np.clip(
            ((normalized + 3) / 6 * (2 ** self._quantize_bits)).astype(int),
            0, (2 ** self._quantize_bits) - 1,
        )

        # Hash to stage ID
        raw_hash = hash(bins.tobytes()) % self.n_buckets

        # Smoothing: only transition if we see the new stage consistently
        self._stage_buffer.append(raw_hash)
        if len(self._stage_buffer) >= self._smoothing_window:
            # Majority vote
            from collections import Counter
            votes = Counter(self._stage_buffer)
            majority, count = votes.most_common(1)[0]
            if count >= self._smoothing_window // 2 + 1:
                if majority != self._current_stage:
                    self._transitions += 1
                self._current_stage = majority

        self._stage_history.append(self._current_stage)
        return self._current_stage

    def record(self, stage_id: int, learner_name: str, reward: float) -> None:
        """Record learner performance at a specific stage."""
        self._stages[stage_id].record(learner_name, reward)

    def get_best_learner(self, stage_id: int) -> Optional[str]:
        """Get the best-performing learner for a given stage."""
        if stage_id in self._stages:
            return self._stages[stage_id].get_best()
        return None

    def get_stage_recommendation(self, features: np.ndarray) -> Optional[str]:
        """Classify features and return best learner for that stage."""
        stage = self.classify(features)
        return self.get_best_learner(stage)

    def report(self) -> Dict[str, Any]:
        """Full report of stage statistics."""
        stage_info = {}
        for stage_id, stats in sorted(self._stages.items()):
            stage_info[f"stage_{stage_id}"] = {
                "visits": stats.visits,
                "best_learner": stats.get_best(),
                "win_rates": stats.win_rates(),
            }

        return {
            "n_stages_seen": len(self._stages),
            "current_stage": self._current_stage,
            "total_transitions": self._transitions,
            "total_classified": self._n_seen,
            "stages": stage_info,
        }
