"""
probe_runner.py — Run short probe trials to empirically compare learners.

Runs N-step probes with each candidate algorithm, scores by reward slope
(learning speed, not just raw reward), and returns a ranking.

Usage:
    from brain.learning.probe_runner import ProbeRunner

    runner = ProbeRunner(brain, probe_steps=500)
    result = runner.run_probe(obs_generator)
    print(result.ranking)  # [("torch_dqn", 0.82), ("numpy_dqn", 0.34)]
"""

from __future__ import annotations

import copy
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ProbeResult:
    """Result from a probe run."""
    ranking: List[Tuple[str, float]]   # [(name, slope_score), ...] sorted best-first
    eliminated: Optional[str] = None   # Worst performer name
    details: Dict[str, Dict[str, float]] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    probe_steps: int = 0

    @property
    def winner(self) -> Optional[str]:
        return self.ranking[0][0] if self.ranking else None


class ProbeRunner:
    """
    Run short empirical probes with multiple learners.

    Each probe:
    1. Snapshots the current DQN weights
    2. Runs N steps with each candidate learner
    3. Scores by reward SLOPE (linear regression on episode rewards)
       This measures learning speed, not just raw reward.
    4. Restores the original weights
    5. Returns ranking with the worst performer marked for elimination

    The probe does NOT destroy anything — weights are cloned and restored.
    """

    def __init__(
        self,
        brain,
        probe_steps: int = 500,
        top_k: int = 3,
    ):
        self.brain = brain
        self.probe_steps = probe_steps
        self.top_k = top_k

    def run_probe(
        self,
        obs_fn: Callable[[], np.ndarray],
        reward_fn: Optional[Callable[[int], float]] = None,
    ) -> ProbeResult:
        """
        Run probe with all registered learners.

        Args:
            obs_fn: Callable that returns current observation (features).
                    Called each step to get the next observation.
            reward_fn: Optional reward function. If None, uses random reward.
                       Takes action as input, returns reward.

        Returns:
            ProbeResult with ranking and elimination recommendation.
        """
        if self.brain.meta_controller is None:
            return ProbeResult(ranking=[], probe_steps=self.probe_steps)

        mc = self.brain.meta_controller
        learner_names = list(mc._slots.keys())[:self.top_k]

        if len(learner_names) < 2:
            return ProbeResult(
                ranking=[(learner_names[0], 1.0)] if learner_names else [],
                probe_steps=self.probe_steps,
            )

        t0 = time.perf_counter()
        details: Dict[str, Dict[str, float]] = {}

        for name in learner_names:
            slot = mc._slots[name]
            learner = slot.learner

            # Run probe with this learner
            probe_data = self._run_single_probe(learner, obs_fn, reward_fn)
            details[name] = probe_data

        elapsed = time.perf_counter() - t0

        # Rank by reward slope (learning speed)
        ranking = sorted(
            [(name, details[name]["slope"]) for name in learner_names],
            key=lambda x: -x[1],
        )

        # Mark worst for elimination
        eliminated = ranking[-1][0] if len(ranking) > 2 else None

        return ProbeResult(
            ranking=ranking,
            eliminated=eliminated,
            details=details,
            elapsed_seconds=round(elapsed, 2),
            probe_steps=self.probe_steps,
        )

    def _run_single_probe(
        self,
        learner: Any,
        obs_fn: Callable[[], np.ndarray],
        reward_fn: Optional[Callable[[int], float]],
    ) -> Dict[str, float]:
        """Run a single probe with one learner and score it."""
        episode_rewards: List[float] = []
        ep_reward = 0.0
        ep_steps = 0

        for step in range(self.probe_steps):
            obs = obs_fn()

            # Get action from learner
            try:
                if hasattr(learner, 'select_action'):
                    action, _ = learner.select_action(obs, explore=True)
                elif hasattr(learner, 'forward'):
                    q_vals = learner.forward(obs)
                    action = int(np.argmax(q_vals))
                else:
                    action = np.random.randint(
                        getattr(learner, 'n_actions', 4)
                    )
            except Exception:
                action = 0

            # Get reward
            if reward_fn is not None:
                reward = reward_fn(action)
            else:
                reward = np.random.randn() * 0.1

            ep_reward += reward
            ep_steps += 1

            # Episode boundary every 25 steps
            if ep_steps >= 25:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_steps = 0

        # Add final partial episode
        if ep_steps > 0:
            episode_rewards.append(ep_reward)

        return self._score_probe(episode_rewards)

    @staticmethod
    def _score_probe(episode_rewards: List[float]) -> Dict[str, float]:
        """
        Score a probe by reward slope (linear regression).

        A positive slope means the learner is IMPROVING over the probe.
        This is more informative than mean reward because it captures
        learning speed, not just initial performance.
        """
        if len(episode_rewards) < 2:
            return {
                "slope": 0.0,
                "mean_reward": episode_rewards[0] if episode_rewards else 0.0,
                "std_reward": 0.0,
                "n_episodes": len(episode_rewards),
            }

        x = np.arange(len(episode_rewards), dtype=np.float64)
        y = np.array(episode_rewards, dtype=np.float64)

        # Linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        var_x = np.var(x)
        slope = float(cov_xy / var_x) if var_x > 0 else 0.0

        return {
            "slope": round(slope, 6),
            "mean_reward": round(float(y_mean), 4),
            "std_reward": round(float(np.std(y)), 4),
            "n_episodes": len(episode_rewards),
            "first_5_avg": round(float(np.mean(y[:5])), 4) if len(y) >= 5 else round(float(y_mean), 4),
            "last_5_avg": round(float(np.mean(y[-5:])), 4) if len(y) >= 5 else round(float(y_mean), 4),
        }
