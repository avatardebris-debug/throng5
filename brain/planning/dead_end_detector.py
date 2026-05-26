"""
dead_end_detector.py — Detect unwinnable game states.

After irreversible actions (block pushes, item pickups), runs compressed
rollouts to determine if the level is still solvable. If no rollout from
the current state ever reaches reward/goal in N trials, marks it as dead end.

Critical for puzzle games (Adventures of Lolo) where wrong block placement
makes the level unwinnable — but the game never tells you.

Usage:
    detector = DeadEndDetector(brain, default_trials=500, trap_check_trials=20)
    is_dead = detector.check(features)  # uses default_trials
    if detector.is_trap(features, action, reward):  # uses trap_check_trials (hot path)
        ...
    if is_dead:
        env.load_state(saved_state)  # Undo the bad action
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional

import numpy as np


class DeadEndDetector:
    """
    Detects unwinnable states via compressed rollouts.

    If zero rollouts from a state reach positive reward in N trials,
    the state is flagged as a dead end. This catches:
    - Blocks pushed to wrong positions (puzzle games)
    - Items collected in wrong order (traps)
    - Paths permanently closed

    False positive rate is managed by requiring high trial counts with
    zero successes — very unlikely for solvable states.
    """

    def __init__(
        self,
        brain,
        default_trials: int = 500,
        rollout_length: int = 200,
        reward_threshold: float = 0.0,
        trap_check_trials: int = 20,
        pre_action_trials: Optional[int] = None,
    ):
        self.brain = brain
        self.default_trials = default_trials
        self.rollout_length = rollout_length
        self.reward_threshold = reward_threshold
        # Hot-path trap check uses fewer trials than full dead-end checks.
        self.trap_check_trials = min(trap_check_trials, default_trials)
        # Before/after action checks use a policy-bound baseline trial count.
        self.pre_action_trials = (
            min(200, self.default_trials)
            if pre_action_trials is None
            else min(pre_action_trials, self.default_trials)
        )

        self._checked: Dict[str, bool] = {}  # cache key -> is_dead_end
        self._checks_run: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def check(
        self,
        features: np.ndarray,
        n_trials: Optional[int] = None,
        custom_success_fn: Optional[Callable] = None,
    ) -> bool:
        """
        Check if the current state is a dead end.

        Runs n_trials compressed rollouts. If ZERO succeed, marks as dead end.
        Results are cached by state bytes + rollout policy.

        Args:
            features: Current state features
            n_trials: Number of rollouts (default: ``default_trials``)
            custom_success_fn: Custom success checker

        Returns:
            True if state appears to be a dead end.
        """
        n_trials = n_trials or self.default_trials
        self._checks_run += 1
        cache_key = self._cache_key(features, n_trials, custom_success_fn)
        if cache_key in self._checked:
            self._cache_hits += 1
            return self._checked[cache_key]
        self._cache_misses += 1

        for trial in range(n_trials):
            sim_features = features.copy()
            total_reward = 0.0
            reached_goal = False

            for step in range(self.rollout_length):
                # Use policy with exploration
                action = self._get_action(sim_features, explore=True)

                # Forward predict
                next_features, pred_reward = self._predict_next(sim_features, action)
                total_reward += pred_reward

                if custom_success_fn:
                    if custom_success_fn(next_features, total_reward):
                        reached_goal = True
                        break
                elif pred_reward > self.reward_threshold:
                    reached_goal = True
                    break

                sim_features = next_features

            if reached_goal:
                self._checked[cache_key] = False
                return False

        self._checked[cache_key] = True
        return True

    def check_after_action(
        self,
        features_before: np.ndarray,
        features_after: np.ndarray,
        action: int,
        was_irreversible: bool = False,
    ) -> Dict[str, Any]:
        """
        Check if an action created a dead end.

        Compares solvability before and after the action.
        If the state was solvable before but not after, the action
        caused the dead end. Baseline (before-action) checks use
        ``pre_action_trials`` and after-action checks use ``default_trials``.

        Returns:
            {"is_dead_end": bool, "caused_by_action": bool, "action": int}
        """
        before_trials = self.pre_action_trials
        was_solvable_before = not self.check(features_before, n_trials=before_trials)
        is_dead_after = self.check(features_after, n_trials=self.default_trials)

        return {
            "is_dead_end": is_dead_after,
            "caused_by_action": was_solvable_before and is_dead_after,
            "action": action,
            "was_irreversible": was_irreversible,
        }

    def is_trap(
        self,
        features: np.ndarray,
        action: int,
        immediate_reward: float,
    ) -> bool:
        """
        Detect reward traps: action gives positive reward but leads to dead end.

        The Lolo problem: collecting a heart gives +100 but if you collect
        it before blocking the dragon, the level becomes unsolvable.
        """
        if immediate_reward <= 0:
            return False

        # Simulate the action
        next_features, _ = self._predict_next(features, action)

        # Check if the reward-giving state is a dead end (cheap trials for hot path)
        return self.check(next_features, n_trials=self.trap_check_trials)

    def _get_action(self, features: np.ndarray, explore: bool = True) -> int:
        """Get action from brain's policy."""
        try:
            if self.brain.striatum._torch_dqn is not None:
                action, _ = self.brain.striatum._torch_dqn.select_action(
                    features, explore=explore,
                )
                return action
            q_vals = self.brain.striatum._forward(features)
            if explore and np.random.rand() < 0.3:
                return np.random.randint(len(q_vals))
            return int(np.argmax(q_vals))
        except (AttributeError, IndexError, TypeError, ValueError):
            return np.random.randint(
                getattr(self.brain.striatum, '_n_actions', 4)
            )

    def _predict_next(
        self, features: np.ndarray, action: int,
    ) -> tuple:
        """Predict next state + reward using world model or noise."""
        try:
            wm = self.brain.basal_ganglia._world_model
            if wm is not None:
                return wm.predict(features, action)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass

        # Fallback: noisy perturbation
        noise = np.random.randn(len(features)).astype(np.float32) * 0.05
        return features + noise, float(np.random.randn() * 0.1)

    def report(self) -> Dict[str, Any]:
        return {
            "checks_run": self._checks_run,
            "dead_ends_found": sum(1 for v in self._checked.values() if v),
            "cache_entries": len(self._checked),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "default_trials": self.default_trials,
            "trap_check_trials": self.trap_check_trials,
            "pre_action_trials": self.pre_action_trials,
        }

    def _cache_key(
        self,
        features: np.ndarray,
        n_trials: int,
        custom_success_fn: Optional[Callable],
    ) -> str:
        """Build deterministic cache key for one check policy."""
        success_key = (
            f"custom:{id(custom_success_fn)}"
            if custom_success_fn is not None
            else "reward"
        )
        feature_hash = hashlib.sha1(features.tobytes()).hexdigest()
        return (
            f"{features.dtype}:{features.shape}:{n_trials}:{self.rollout_length}:"
            f"{self.reward_threshold}:{success_key}:{feature_hash}"
        )
