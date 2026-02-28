"""
learner_selector.py — Intelligent RL algorithm selection for the Striatum.

Given an environment fingerprint (action space, observation size, reward
characteristics), selects the best RL algorithm from the registry.

Operates in two modes:
  1. Rule-based (Phase 3): heuristic selection based on env properties
  2. Bandit-based (future): learned selection based on historical performance

Usage:
    from brain.learning.learner_selector import LearnerSelector
    from brain.learning.rl_registry import RLRegistry

    registry = RLRegistry()
    selector = LearnerSelector(registry)

    recommendation = selector.recommend({
        "action_space": "discrete",
        "n_actions": 18,
        "obs_dim": 84,
        "reward_sparsity": "sparse",
        "has_gpu": False,
    })
    print(recommendation)
    # → LearnerRecommendation(name="dqn_builtin", reason="...")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from brain.learning.rl_registry import (
    RLRegistry,
    AlgorithmInfo,
    AlgorithmType,
    ActionSpaceType,
    Learner,
)


@dataclass
class LearnerRecommendation:
    """Result from the learner selector."""
    name: str
    display_name: str
    reason: str
    confidence: float          # 0-1, how confident this is the right choice
    alternatives: List[str]    # Other good options
    hyperparams: Dict[str, Any]  # Suggested hyperparameters


class LearnerSelector:
    """
    Recommends RL algorithms based on environment characteristics.

    Phase 3 (current): Rule-based selection using heuristics.
    Future: Multi-armed bandit that learns which algorithm performs best
    for different environment fingerprints.
    """

    def __init__(self, registry: RLRegistry):
        self._registry = registry
        self._performance_history: Dict[str, List[float]] = {}
        self._selection_count: Dict[str, int] = {}

    def recommend(self, env_fingerprint: Dict[str, Any]) -> LearnerRecommendation:
        """
        Recommend a learner based on environment characteristics.

        env_fingerprint keys:
            action_space: "discrete" or "continuous"
            n_actions: int (if discrete)
            obs_dim: int — observation dimensionality
            reward_sparsity: "dense", "moderate", "sparse"
            avg_episode_length: float (steps per episode)
            has_gpu: bool — GPU available
            compute_budget: "low", "medium", "high"
            is_pomdp: bool — partially observable?
        """
        action_space = env_fingerprint.get("action_space", "discrete")
        n_actions = env_fingerprint.get("n_actions", 18)
        obs_dim = env_fingerprint.get("obs_dim", 84)
        reward_sparsity = env_fingerprint.get("reward_sparsity", "moderate")
        has_gpu = env_fingerprint.get("has_gpu", False)
        compute_budget = env_fingerprint.get("compute_budget", "medium")
        is_pomdp = env_fingerprint.get("is_pomdp", False)
        avg_ep_len = env_fingerprint.get("avg_episode_length", 200)

        # Filter by action space compatibility
        space = ActionSpaceType.CONTINUOUS if action_space == "continuous" else ActionSpaceType.DISCRETE
        candidates = self._registry.list_for_action_space(space)

        if not candidates:
            return self._fallback()

        # Score each candidate
        scored = []
        for algo in candidates:
            score, reason = self._score_algorithm(
                algo, space, reward_sparsity, has_gpu, compute_budget,
                is_pomdp, n_actions, obs_dim, avg_ep_len
            )
            scored.append((score, algo, reason))

        # Sort by score (highest first)
        scored.sort(key=lambda x: -x[0])

        best_score, best_algo, best_reason = scored[0]
        alternatives = [s[1].name for s in scored[1:4]]

        # Generate suggested hyperparameters
        hyperparams = self._suggest_hyperparams(
            best_algo, n_actions, obs_dim, avg_ep_len
        )

        return LearnerRecommendation(
            name=best_algo.name,
            display_name=best_algo.display_name,
            reason=best_reason,
            confidence=min(best_score / 10.0, 1.0),
            alternatives=alternatives,
            hyperparams=hyperparams,
        )

    def create_recommended(self, env_fingerprint: Dict[str, Any]) -> tuple:
        """Recommend and create a learner in one step. Returns (Learner, recommendation)."""
        rec = self.recommend(env_fingerprint)
        learner = self._registry.create(
            rec.name,
            n_features=env_fingerprint.get("obs_dim", 84),
            n_actions=env_fingerprint.get("n_actions", 18),
            **rec.hyperparams,
        )
        return learner, rec

    def record_performance(self, algo_name: str, score: float) -> None:
        """Record how well an algorithm performed (for future bandit selection)."""
        if algo_name not in self._performance_history:
            self._performance_history[algo_name] = []
        self._performance_history[algo_name].append(score)

    # ── Scoring ───────────────────────────────────────────────────────

    def _score_algorithm(
        self,
        algo: AlgorithmInfo,
        space: ActionSpaceType,
        reward_sparsity: str,
        has_gpu: bool,
        compute_budget: str,
        is_pomdp: bool,
        n_actions: int,
        obs_dim: int,
        avg_ep_len: float,
    ) -> tuple:
        """Score an algorithm for the given environment. Returns (score, reason)."""
        score = 5.0  # Base score
        reasons = []

        # ── Compute constraints ───────────────────────────────────────
        if algo.requires_torch and not has_gpu:
            # Penalize heavy PyTorch algorithms on CPU
            score -= algo.compute_cost * 0.5
            reasons.append("GPU recommended but not available")

        if compute_budget == "low" and algo.compute_cost >= 4:
            score -= 2
            reasons.append("too compute-heavy for budget")
        elif compute_budget == "high" and algo.compute_cost <= 2:
            score += 1
            reasons.append("lightweight, could use more compute")

        # ── CPU-only bonus for builtin ────────────────────────────────
        if not algo.requires_torch:
            if not has_gpu:
                score += 2
                reasons.append("no GPU needed")
            score += 1  # Always give builtin a slight bonus for simplicity

        # ── Sparse reward environments ────────────────────────────────
        if reward_sparsity == "sparse":
            if "icm" in algo.name or "rnd" in algo.name or "ngu" in algo.name:
                score += 3
                reasons.append("intrinsic motivation for sparse rewards")
            elif algo.sample_efficiency >= 4:
                score += 1
                reasons.append("good sample efficiency helps with sparse rewards")
            else:
                score -= 1
                reasons.append("poor sample efficiency in sparse reward setting")

        # ── Partial observability ─────────────────────────────────────
        if is_pomdp:
            if algo.supports_recurrent:
                score += 3
                reasons.append("recurrent architecture for POMDPs")
            else:
                score -= 1

        # ── Sample efficiency bonus ───────────────────────────────────
        score += algo.sample_efficiency * 0.3

        # ── Stability bonus ───────────────────────────────────────────
        score += algo.stability * 0.2

        # ── Continuous action space ───────────────────────────────────
        if space == ActionSpaceType.CONTINUOUS:
            if algo.name == "sac":
                score += 2
                reasons.append("SAC is ideal for continuous control")
            elif algo.name == "ppo":
                score += 1
                reasons.append("PPO handles continuous well")

        # ── Historical performance (bandit signal) ────────────────────
        history = self._performance_history.get(algo.name, [])
        if history:
            avg_perf = np.mean(history[-10:])
            score += avg_perf * 0.5
            reasons.append(f"historical avg={avg_perf:.2f}")

        reason = f"{algo.display_name}: " + "; ".join(reasons) if reasons else algo.best_for
        return score, reason

    def _suggest_hyperparams(
        self, algo: AlgorithmInfo, n_actions: int, obs_dim: int, avg_ep_len: float,
    ) -> Dict[str, Any]:
        """Generate suggested hyperparameters based on env characteristics."""
        params: Dict[str, Any] = {}

        # Hidden size scales with observation dimensionality
        if obs_dim <= 32:
            params["hidden_size"] = 64
        elif obs_dim <= 128:
            params["hidden_size"] = 128
        else:
            params["hidden_size"] = 256

        # Batch size scales with available data
        if avg_ep_len < 50:
            params["batch_size"] = 16
        elif avg_ep_len < 200:
            params["batch_size"] = 32
        else:
            params["batch_size"] = 64

        # Learning rate — conservative for value-based, standard for policy
        if algo.algorithm_type == AlgorithmType.VALUE_BASED:
            params["lr"] = 0.001
        elif algo.algorithm_type == AlgorithmType.ACTOR_CRITIC:
            params["lr"] = 0.0003
        else:
            params["lr"] = 0.001

        # Discount factor — higher for longer episodes
        if avg_ep_len > 500:
            params["gamma"] = 0.995
        elif avg_ep_len > 100:
            params["gamma"] = 0.99
        else:
            params["gamma"] = 0.95

        return params

    def _fallback(self) -> LearnerRecommendation:
        return LearnerRecommendation(
            name="dqn_builtin",
            display_name="DQN (Built-in NumPy)",
            reason="Fallback: no compatible algorithms found",
            confidence=0.3,
            alternatives=[],
            hyperparams={"hidden_size": 128, "lr": 0.001, "gamma": 0.99, "batch_size": 32},
        )
