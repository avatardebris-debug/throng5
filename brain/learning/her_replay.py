"""
her_replay.py — Hindsight Experience Replay (HER, Andrychowicz et al 2017).

HER converts failed trajectories into successful ones by relabelling the
goal: "I was trying to reach G but ended up at s_T → pretend I was trying
to reach s_T all along, so my reward is +1."

This creates dense learning signal in sparse-reward environments like:
  • FrozenLake (almost never reaches goal randomly)
  • MountainCar  (almost never reaches top randomly)
  • Any goal-conditioned task

Strategy used: "future" (sample k random future states from the same episode
as alternative goals).  This is the default recommended strategy in the paper.

Usage
-----
    her = HERReplayBuffer(k=4, sparse_reward_threshold=0.0)

    # At end of each episode, if reward was sparse:
    relabelled = her.relabel_episode(episode_transitions)
    for t in relabelled:
        striatum._nstep.push(*t)   # inject into main training pipeline
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class HERReplayBuffer:
    """
    Hindsight Experience Replay: relabels sparse-reward episodes.

    Parameters
    ----------
    k : int
        Number of HER-relabelled goals per real transition (default 4).
        Higher k → more synthetic transitions, better for very sparse rewards.
    sparse_reward_threshold : float
        Episodes with total_reward <= this are treated as sparse/failed and
        eligible for HER relabelling.  Default 0.0.
    goal_reward : float
        Reward assigned to the relabelled goal-reaching transition. Default +1.0.
    goal_dims : optional list of int
        If states have a goal embedded (e.g. GoalEnv), these are the state
        indices representing the achieved goal for distance-based reward.
        When None, the fallback is: any relabelled goal-reach gives goal_reward.
    """

    def __init__(
        self,
        k: int = 4,
        sparse_reward_threshold: float = 0.0,
        goal_reward: float = 1.0,
        goal_dims: Optional[List[int]] = None,
        strategy: str = "future",   # "future" or "final"
    ):
        self.k = k
        self.sparse_reward_threshold = sparse_reward_threshold
        self.goal_reward = goal_reward
        self.goal_dims = goal_dims
        self.strategy = strategy
        self._episodes_relabelled = 0
        self._transitions_generated = 0

    # ── Main API ──────────────────────────────────────────────────────

    def should_relabel(self, total_reward: float) -> bool:
        """Returns True if the episode is sparse enough to warrant HER."""
        return total_reward <= self.sparse_reward_threshold

    def relabel_episode(
        self,
        transitions: List[Transition],
        total_reward: Optional[float] = None,
        strategy: Optional[str] = None,
    ) -> List[Transition]:
        """
        Apply HER to a complete episode.

        Strategies:
        - "future"  (default): for each step t, sample k random future states
        - "final":  use the last episode state as goal for all steps (best for
                    deterministic goal envs like GridWorld where last state = goal)

        Returns
        -------
        List of relabelled Transition tuples.
        """
        if len(transitions) < 2:
            return []

        strat = strategy or self.strategy
        relabelled: List[Transition] = []

        # "final": the last next_state is the achieved goal for the whole episode
        if strat == "final":
            final_goal = transitions[-1][3]
            for s, a, r, s2, done in transitions[:-1]:
                rl_reward = self._compute_reward(s2, final_goal)
                relabelled.append((
                    np.asarray(s, dtype=np.float32), a,
                    float(rl_reward),
                    np.asarray(final_goal, dtype=np.float32),
                    True,   # MINOR-5: goal is always "reached" in hindsight
                ))
                self._transitions_generated += 1
            if relabelled:
                self._episodes_relabelled += 1
            return relabelled

        # "future": default strategy
        for t_idx, (s, a, r, s2, done) in enumerate(transitions):
            future_range = list(range(t_idx + 1, len(transitions)))
            if not future_range:
                continue

            k = min(self.k, len(future_range))
            goal_indices = random.sample(future_range, k)

            for g_idx in goal_indices:
                achieved_state = transitions[g_idx][3]  # next_state at g_idx
                rl_reward = self._compute_reward(s2, achieved_state)
                # MINOR-5: done=True — the agent "reached" this relabelled goal
                relabelled.append((
                    np.asarray(s, dtype=np.float32),
                    a,
                    float(rl_reward),
                    np.asarray(achieved_state, dtype=np.float32),
                    True,
                ))
                self._transitions_generated += 1

        if relabelled:
            self._episodes_relabelled += 1

        return relabelled

    # ── Reward function ───────────────────────────────────────────────

    def _compute_reward(
        self,
        achieved: np.ndarray,
        goal: np.ndarray,
    ) -> float:
        """
        Compute hindsight reward for reaching a goal state.

        - With goal_dims: L2 distance on those dims → +1 if close, -0.01 otherwise
        - Without goal_dims (MAJOR-4 fix): L2-distance-weighted shaping so the
          agent gets gradient toward actual goal states, not a uniform +1 for
          any future state regardless of position.
        """
        if self.goal_dims is not None:
            dist = float(np.linalg.norm(
                achieved[self.goal_dims] - goal[self.goal_dims]
            ))
            return self.goal_reward if dist < 0.1 else -0.01
        # No structured goal: shape reward by L2 similarity between achieved and goal.
        # At achieved==goal: reward=goal_reward. At large distance: reward→0.
        dist = float(np.linalg.norm(achieved - goal))
        scale = float(np.linalg.norm(goal)) + 1e-8
        return self.goal_reward * float(np.clip(1.0 - dist / scale, 0.0, 1.0))

    # ── Stats ─────────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        return {
            "episodes_relabelled": self._episodes_relabelled,
            "transitions_generated": self._transitions_generated,
            "k": self.k,
        }

    def __repr__(self) -> str:
        return (
            f"HERReplayBuffer(k={self.k}, "
            f"eps_relabelled={self._episodes_relabelled})"
        )
