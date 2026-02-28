"""
montezuma_bandit.py
===================
Adaptive episode allocator — A/B tournament between two policies.

Policy A: PortableNNAgent + SurvivorAmygdala  (current baseline)
Policy B: MetaStackPipeline + DreamerTeacher + amygdala  (meta-stack)

After each episode the shaped reward is recorded for whichever policy ran.
A softmax over EMA rewards determines the probability of selecting each
policy for the next episode.  Over time the better policy gets more episodes.

Both policies share the same replay buffer (off-policy), so neither falls
behind when it isn't selected.

Usage::

    bandit = PolicyBandit(alpha=0.2)
    for ep in range(n_episodes):
        policy = bandit.select()          # "A" or "B"
        agent  = agent_A if policy == "A" else agent_B
        result = run_episode(..., agent=agent, ...)
        bandit.record(policy, result["shaped_reward"])
        print(bandit.status_line())
    print(bandit.summary())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal

Policy = Literal["A", "B"]


@dataclass
class _PolicyStats:
    name:       str
    ema:        float = 0.0          # exponential moving average of shaped reward
    n_episodes: int   = 0
    total_reward: float = 0.0
    wins:       int   = 0            # episodes where this policy had higher EMA


class PolicyBandit:
    """
    Softmax-weighted episode bandit over two policies.

    Weight update (EMA) after each episode::

        ema_A += alpha * (reward - ema_A)   # if policy A ran
        weight_A = softmax(ema_A, ema_B)[0]

    Parameters
    ----------
    alpha : float
        EMA smoothing factor.  0.2 → ~5-episode memory window.
    temp : float
        Softmax temperature.  Higher = more uniform (more exploration).
        Lower = winner-takes-all faster.
    force : str | None
        If "A" or "B", always selects that policy (for solo runs).
    """

    def __init__(
        self,
        alpha: float = 0.2,
        temp:  float = 1.0,
        force: "Policy | None" = None,
    ) -> None:
        self._alpha = alpha
        self._temp  = temp
        self._force = force
        self._A     = _PolicyStats(name="A")
        self._B     = _PolicyStats(name="B")
        self._last:  Policy = "A"
        self._ep    = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select(self) -> Policy:
        """Select a policy for the next episode."""
        self._ep += 1
        if self._force is not None:
            self._last = self._force
            return self._force

        # First two episodes: run each once to seed EMA
        if self._ep <= 2:
            self._last = "A" if self._ep == 1 else "B"
            return self._last

        w = self.weight_A()
        self._last = "A" if random.random() < w else "B"
        return self._last

    def record(self, policy: Policy, reward: float) -> None:
        """Update EMA after an episode completes."""
        stat = self._A if policy == "A" else self._B
        stat.n_episodes  += 1
        stat.total_reward += reward

        # EMA update
        if stat.n_episodes == 1:
            stat.ema = reward   # cold start
        else:
            stat.ema += self._alpha * (reward - stat.ema)

        # Track wins (which policy had higher EMA this episode)
        if self._A.n_episodes > 0 and self._B.n_episodes > 0:
            winner = "A" if self._A.ema >= self._B.ema else "B"
            (self._A if winner == "A" else self._B).wins += 1

    # ------------------------------------------------------------------
    # Weight / status
    # ------------------------------------------------------------------

    def weight_A(self) -> float:
        """Probability of selecting Policy A (softmax over EMA)."""
        if self._A.n_episodes == 0 and self._B.n_episodes == 0:
            return 0.5
        if self._A.n_episodes == 0:
            return 0.1
        if self._B.n_episodes == 0:
            return 0.9
        a = self._A.ema / self._temp
        b = self._B.ema / self._temp
        # Numerically stable softmax
        m = max(a, b)
        ea, eb = math.exp(a - m), math.exp(b - m)
        return ea / (ea + eb)

    @property
    def last_policy(self) -> Policy:
        return self._last

    def status_line(self) -> str:
        wa = self.weight_A()
        return (
            f"[bandit] A={wa:.2f}({self._A.n_episodes}ep ema={self._A.ema:+.1f})  "
            f"B={1-wa:.2f}({self._B.n_episodes}ep ema={self._B.ema:+.1f})  "
            f"last={self._last}"
        )

    def summary(self) -> str:
        wa = self.weight_A()
        lines = [
            "  [bandit] A/B Tournament Summary",
            f"    Policy A (PortableNN+Amygdala):    "
            f"eps={self._A.n_episodes:3d}  ema={self._A.ema:+7.2f}  "
            f"avg={self._A.total_reward/max(1,self._A.n_episodes):+7.2f}  "
            f"wins={self._A.wins}",
            f"    Policy B (MetaStack+Dreamer):       "
            f"eps={self._B.n_episodes:3d}  ema={self._B.ema:+7.2f}  "
            f"avg={self._B.total_reward/max(1,self._B.n_episodes):+7.2f}  "
            f"wins={self._B.wins}",
            f"    Final weights: A={wa:.3f}  B={1-wa:.3f}  "
            f"→ {'A wins' if wa > 0.5 else 'B wins' if wa < 0.5 else 'tied'}",
        ]
        return "\n".join(lines)
