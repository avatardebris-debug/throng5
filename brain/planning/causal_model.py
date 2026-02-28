"""
causal_model.py — Entity-level effect tracking.

Tracks what *changes* when actions are taken, not just what the next
state is. Critical for puzzle games where understanding causation
is required:
    "Pushing block X causes path Y to open"
    "Collecting item Z before condition W = trap"

Learns cause-effect relationships from observed state transitions:
    (state_diff, action) → effect_type

Usage:
    model = CausalModel()
    model.observe(features_before, action, features_after, reward)
    effects = model.predict_effects(features, action)
    preconditions = model.get_preconditions(goal_hash)
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CausalEffect:
    """A learned cause-effect relationship."""
    action: int
    feature_deltas: np.ndarray  # Which features changed and by how much
    reward_delta: float = 0.0
    observations: int = 0
    led_to_dead_end: int = 0
    led_to_reward: int = 0

    @property
    def is_dangerous(self) -> bool:
        """This action frequently leads to dead ends."""
        if self.observations < 5:
            return False
        return self.led_to_dead_end / self.observations > 0.5

    @property
    def is_beneficial(self) -> bool:
        """This action frequently leads to reward."""
        if self.observations < 5:
            return False
        return self.led_to_reward / self.observations > 0.5


class CausalModel:
    """
    Learns cause-effect relationships from state transitions.

    Instead of predicting the full next state (World Model does that),
    this tracks:
    1. What features CHANGE when an action is taken (deltas)
    2. Whether those changes tend to lead to reward or dead ends
    3. What preconditions (prior state features) are required for effects

    This is a simple statistical model, not a full causal DAG.
    It learns from observation: "when feature X is high and I do action Y,
    feature Z changes — and that usually leads to reward."
    """

    def __init__(
        self,
        n_feature_bins: int = 16,
        max_effects: int = 5000,
    ):
        self._n_bins = n_feature_bins
        self._max_effects = max_effects

        # Effects indexed by (state_bin, action) → list of observed deltas
        self._effects: Dict[Tuple[int, int], CausalEffect] = {}

        # Precondition tracking: what state features must be true for a goal
        # goal_hash → set of (feature_idx, required_range)
        self._preconditions: Dict[int, List[int]] = defaultdict(list)

        # Irreversibility tracking: actions that change state permanently
        self._irreversible_actions: Dict[Tuple[int, int], int] = defaultdict(int)

        self._total_observations: int = 0

        # Running normalization
        self._running_mean = np.zeros(64, dtype=np.float64)
        self._running_var = np.ones(64, dtype=np.float64)
        self._n_seen: int = 0

    def _bin_state(self, features: np.ndarray) -> int:
        """Coarse bin of the state for effect lookup."""
        features = np.asarray(features, dtype=np.float32).flatten()
        k = min(4, len(features))
        top_k = features[:k]
        bins = np.clip((top_k * 4).astype(int), -8, 7) + 8
        return hash(bins.tobytes()) % self._n_bins

    def observe(
        self,
        features_before: np.ndarray,
        action: int,
        features_after: np.ndarray,
        reward: float = 0.0,
        is_dead_end: bool = False,
    ) -> None:
        """
        Record a state transition and learn its causal effects.

        Args:
            features_before: State features before action
            action: Action taken
            features_after: State features after action
            reward: Reward received
            is_dead_end: Whether the resulting state is a dead end
        """
        self._total_observations += 1

        before = np.asarray(features_before, dtype=np.float32).flatten()
        after = np.asarray(features_after, dtype=np.float32).flatten()

        # Compute deltas
        min_len = min(len(before), len(after))
        delta = after[:min_len] - before[:min_len]

        state_bin = self._bin_state(before)
        key = (state_bin, action)

        if key not in self._effects:
            self._effects[key] = CausalEffect(
                action=action,
                feature_deltas=np.zeros_like(delta),
            )

        effect = self._effects[key]
        effect.observations += 1

        # Running average of deltas
        alpha = 1.0 / effect.observations
        effect.feature_deltas = (
            (1 - alpha) * effect.feature_deltas[:min_len]
            + alpha * delta
        )
        effect.reward_delta = (1 - alpha) * effect.reward_delta + alpha * reward

        if is_dead_end:
            effect.led_to_dead_end += 1
        if reward > 0:
            effect.led_to_reward += 1

        # Track irreversibility: large changes that don't revert
        change_magnitude = np.max(np.abs(delta))
        if change_magnitude > 0.5:
            self._irreversible_actions[key] += 1

    def predict_effects(
        self, features: np.ndarray, action: int,
    ) -> Optional[CausalEffect]:
        """
        Predict what effects an action will have at this state.

        Returns the learned CausalEffect or None if never observed.
        """
        state_bin = self._bin_state(features)
        return self._effects.get((state_bin, action))

    def is_action_dangerous(
        self, features: np.ndarray, action: int,
    ) -> bool:
        """Check if this action frequently leads to dead ends."""
        effect = self.predict_effects(features, action)
        if effect is None:
            return False
        return effect.is_dangerous

    def is_action_irreversible(
        self, features: np.ndarray, action: int,
    ) -> bool:
        """Check if this action tends to cause large permanent state changes."""
        state_bin = self._bin_state(features)
        count = self._irreversible_actions.get((state_bin, action), 0)
        return count >= 3  # Seen 3+ irreversible transitions for this action

    def get_safe_actions(
        self, features: np.ndarray, n_actions: int,
    ) -> List[int]:
        """Get actions that are NOT known to be dangerous or irreversible."""
        safe = []
        for a in range(n_actions):
            if not self.is_action_dangerous(features, a):
                safe.append(a)
        return safe if safe else list(range(n_actions))  # Fallback: all actions

    def add_precondition(
        self, goal_hash: int, required_state_hash: int,
    ) -> None:
        """Record that goal_hash requires required_state_hash to be achieved first."""
        if required_state_hash not in self._preconditions[goal_hash]:
            self._preconditions[goal_hash].append(required_state_hash)

    def get_preconditions(self, goal_hash: int) -> List[int]:
        """Get state hashes that must be achieved before this goal."""
        return self._preconditions.get(goal_hash, [])

    def get_dangerous_actions(
        self, features: np.ndarray, n_actions: int,
    ) -> List[Dict[str, Any]]:
        """Get all actions known to be dangerous at this state, with details."""
        dangerous = []
        for a in range(n_actions):
            effect = self.predict_effects(features, a)
            if effect and effect.is_dangerous:
                dangerous.append({
                    "action": a,
                    "dead_end_rate": round(effect.led_to_dead_end / max(effect.observations, 1), 3),
                    "observations": effect.observations,
                    "avg_reward": round(effect.reward_delta, 4),
                })
        return dangerous

    def report(self) -> Dict[str, Any]:
        dangerous_count = sum(1 for e in self._effects.values() if e.is_dangerous)
        beneficial_count = sum(1 for e in self._effects.values() if e.is_beneficial)
        return {
            "total_observations": self._total_observations,
            "unique_effects": len(self._effects),
            "dangerous_effects": dangerous_count,
            "beneficial_effects": beneficial_count,
            "irreversible_actions": len(self._irreversible_actions),
            "preconditions_tracked": sum(len(v) for v in self._preconditions.values()),
        }
