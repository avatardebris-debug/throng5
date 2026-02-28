"""
counterfactual.py — "What would have happened if I did X instead?"

Uses the world model to simulate alternative histories:
  - "What if I hadn't pushed that block?"
  - "What if I went right instead of left?"
  - "Would I have survived if I jumped earlier?"

This enables:
  - Credit assignment (which action caused the failure?)
  - Regret analysis (was there a better alternative?)
  - Trap verification (does the "good" path actually lead somewhere?)

Usage:
    cf = CounterfactualReasoner(brain)
    result = cf.what_if(state_features, alternative_action, n_steps=50)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class CounterfactualReasoner:
    """
    Simulates alternative action histories using the world model.

    Given a state and an alternative action, rolls out the world model
    to see what WOULD have happened. Compares to what actually happened.
    """

    def __init__(self, brain=None, rollout_length: int = 100):
        self.brain = brain
        self._rollout_length = rollout_length
        self._total_queries: int = 0
        self._regrets_found: int = 0

    def what_if(
        self,
        state_features: np.ndarray,
        alternative_action: int,
        n_steps: int = 0,
        policy_fn=None,
    ) -> Dict[str, Any]:
        """
        Simulate: "What if I took action X from this state?"

        Rolls out the world model for n_steps using the given
        alternative first action, then the normal policy.

        Returns comparison of alt vs. actual trajectory.
        """
        self._total_queries += 1
        n_steps = n_steps or self._rollout_length

        # Simulate alternative trajectory
        alt_reward, alt_survived, alt_states = self._rollout(
            state_features, alternative_action, n_steps, policy_fn,
        )

        return {
            "alternative_action": alternative_action,
            "alt_total_reward": round(alt_reward, 4),
            "alt_survived_steps": alt_survived,
            "n_steps": n_steps,
        }

    def compare_actions(
        self,
        state_features: np.ndarray,
        actions: List[int],
        n_steps: int = 50,
        policy_fn=None,
    ) -> Dict[str, Any]:
        """
        Compare multiple actions from the same state.

        Returns ranked list of actions by expected outcome.
        """
        results = []
        for action in actions:
            reward, survived, _ = self._rollout(
                state_features, action, n_steps, policy_fn,
            )
            results.append({
                "action": action,
                "expected_reward": round(reward, 4),
                "survival_steps": survived,
            })

        results.sort(key=lambda x: -x["expected_reward"])
        best = results[0]["action"]
        worst = results[-1]["action"]

        return {
            "rankings": results,
            "best_action": best,
            "worst_action": worst,
            "best_reward": results[0]["expected_reward"],
            "worst_reward": results[-1]["expected_reward"],
            "regret": results[0]["expected_reward"] - results[-1]["expected_reward"],
        }

    def find_regret(
        self,
        state_features: np.ndarray,
        actual_action: int,
        actual_reward: float,
        n_alternatives: int = 4,
        n_steps: int = 50,
    ) -> Dict[str, Any]:
        """
        Check if the actual action was suboptimal.

        Compares the actual outcome to all alternative actions.
        If an alternative would have been better, reports regret.
        """
        # Get all possible actions
        n_actions = getattr(self.brain.striatum, '_n_actions', 4) if self.brain else 4

        best_alt_reward = -float("inf")
        best_alt_action = actual_action

        for action in range(min(n_actions, n_alternatives)):
            if action == actual_action:
                continue
            reward, survived, _ = self._rollout(state_features, action, n_steps)
            if reward > best_alt_reward:
                best_alt_reward = reward
                best_alt_action = action

        regret = max(0, best_alt_reward - actual_reward)
        if regret > 0:
            self._regrets_found += 1

        return {
            "actual_action": actual_action,
            "actual_reward": round(actual_reward, 4),
            "best_alternative": best_alt_action,
            "best_alt_reward": round(best_alt_reward, 4),
            "regret": round(regret, 4),
            "was_optimal": regret < 0.01,
        }

    def _rollout(
        self,
        features: np.ndarray,
        first_action: int,
        n_steps: int,
        policy_fn=None,
    ) -> tuple:
        """Run a world model rollout."""
        sim_features = np.asarray(features, dtype=np.float32).copy()
        total_reward = 0.0
        states = [sim_features.copy()]

        for step in range(n_steps):
            action = first_action if step == 0 else self._get_action(sim_features, policy_fn)
            next_features, pred_reward = self._predict(sim_features, action)
            total_reward += pred_reward
            sim_features = next_features
            states.append(sim_features.copy())

            # Stop if "death" predicted
            if pred_reward < -5.0:
                return total_reward, step + 1, states

        return total_reward, n_steps, states

    def _predict(self, features: np.ndarray, action: int) -> tuple:
        try:
            wm = self.brain.basal_ganglia._world_model
            if wm is not None:
                return wm.predict(features, action)
        except Exception:
            pass
        noise = np.random.randn(len(features)).astype(np.float32) * 0.05
        return features + noise, float(np.random.randn() * 0.1)

    def _get_action(self, features: np.ndarray, policy_fn=None) -> int:
        if policy_fn:
            return policy_fn(features)
        try:
            q = self.brain.striatum._forward(features)
            return int(np.argmax(q))
        except Exception:
            return np.random.randint(4)

    def report(self) -> Dict[str, Any]:
        return {
            "total_queries": self._total_queries,
            "regrets_found": self._regrets_found,
        }
