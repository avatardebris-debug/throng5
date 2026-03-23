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
    ):
        self.k = k
        self.sparse_reward_threshold = sparse_reward_threshold
        self.goal_reward = goal_reward
        self.goal_dims = goal_dims
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
    ) -> List[Transition]:
        """
        Apply HER "future" strategy to a complete episode.

        For each transition (s_t, a_t, r_t, s_{t+1}, done_t):
          Sample k future states s_g from the episode (t < g ≤ T)
          Relabel:  goal = s_g
                    reward = goal_reward  (reached whatever we were trying to reach)
                    done = True           (episode ends when goal reached)

        The relabelled transitions are ADDED to the original transitions —
        call sites push both original and relabelled sets into the replay buffer.

        Returns
        -------
        List of relabelled Transition tuples (may be empty if episode too short).
        """
        if len(transitions) < 2:
            return []

        relabelled: List[Transition] = []

        for t_idx, (s, a, r, s2, done) in enumerate(transitions):
            # Future states are those after this step
            future_range = list(range(t_idx + 1, len(transitions)))
            if not future_range:
                continue

            # Sample up to k future indices
            k = min(self.k, len(future_range))
            goal_indices = random.sample(future_range, k)

            for g_idx in goal_indices:
                achieved_state = transitions[g_idx][3]  # next_state at g_idx
                # Compute relabelled reward
                rl_reward = self._compute_reward(s2, achieved_state)
                rl_done = (g_idx == len(transitions) - 1)  # done at last goal step

                relabelled.append((
                    np.asarray(s, dtype=np.float32),
                    a,
                    float(rl_reward),
                    np.asarray(achieved_state, dtype=np.float32),
                    rl_done,
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

        Default: simple sparse — +goal_reward if states match closely.
        If goal_dims is set: use L2 distance on those dims.
        """
        if self.goal_dims is not None:
            dist = float(np.linalg.norm(
                achieved[self.goal_dims] - goal[self.goal_dims]
            ))
            return self.goal_reward if dist < 0.1 else -0.01
        # No structured goal — the future-strategy makes any future state a goal,
        # so just give a flat positive signal for relabelled transitions
        return self.goal_reward

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
