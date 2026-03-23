"""
replay_adapters.py — Drop-in replacements / wrappers used by Striatum.

Provides two classes:

  StratifiedReplayDeque
    Wraps PrioritizedReplayBuffer behind the exact same (s, a, r, s', done)
    tuple interface the Striatum deque already uses, so learn() needs zero
    changes to its batch-unpacking logic.  Adds near_death and td_error
    priority updates on top.

  NStepBuffer
    Accumulates n consecutive transitions then flushes a single
    n-step return transition to a downstream buffer.
    G_t = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n max Q(s_{t+n})
    The Bellman backup becomes:
      target = G_t + γ^n * max Q(s_{t+n}) * (1-done)
    which the learn() loop handles just like a 1-step target.

Usage in Striatum.__init__:
    from brain.learning.replay_adapters import StratifiedReplayDeque, NStepBuffer
    self._replay = StratifiedReplayDeque(capacity=buffer_size)
    self._nstep  = NStepBuffer(n=4, gamma=gamma, downstream=self._replay)

Usage in Striatum.learn() push:
    self._nstep.push(state, action, reward, next_state, done, near_death=near)
    # (replaces self._replay.append(...))
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


# ── Stratified Replay Deque ───────────────────────────────────────────────


class StratifiedReplayDeque:
    """
    Drop-in replacement for deque(maxlen=N) that uses stratified priority
    sampling internally.

    Exposes the minimal interface Striatum uses:
      len()           → number of stored transitions
      __getitem__     → (s, a, r, s', done) tuple at index
      append(tuple)   → stores a plain transition with default priority
      push(...)       → stores with near_death/td_error metadata
      update_priorities(indices, td_errors)  → update after learn()
      sample(n)       → list of (s, a, r, s', done) tuples
    """

    # priority alpha (how much to weight TD errors)
    _ALPHA = 0.6
    # importance-sampling beta (starts 0.4, anneals to 1.0)
    _BETA_START = 0.4
    _BETA_END = 1.0
    _BETA_ANNEAL_STEPS = 50_000

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: Deque[Transition] = deque(maxlen=capacity)
        self._priorities: Deque[float] = deque(maxlen=capacity)
        self._near_death: Deque[bool] = deque(maxlen=capacity)
        self._steps = 0     # for beta annealing

    # ── Append interface (compatible with plain deque usage) ──────────

    def append(self, transition: Transition) -> None:
        """Append a plain (s, a, r, s', done) tuple with default priority."""
        s, a, r, s2, done = transition
        near = bool(r < -0.5)   # heuristic near-death from reward
        self._push_internal(s, a, r, s2, done, near_death=near, td_error=1.0)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        near_death: bool = False,
        td_error: float = 1.0,
    ) -> None:
        """Push with explicit metadata."""
        self._push_internal(state, action, reward, next_state, done, near_death, td_error)

    def _push_internal(
        self, state, action, reward, next_state, done, near_death, td_error
    ) -> None:
        priority = self._compute_priority(near_death, td_error)
        self._buf.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))
        self._priorities.append(priority)
        self._near_death.append(near_death)

    def _compute_priority(self, near_death: bool, td_error: float) -> float:
        base = (abs(td_error) + 1e-6) ** self._ALPHA
        nd_mult = 2.0 if near_death else 1.0
        return float(base * nd_mult)

    # ── Sample ────────────────────────────────────────────────────────

    def sample(self, n: int) -> Tuple[List[Transition], np.ndarray]:
        """
        Stratified priority sample.  Returns (transitions, indices).
        Stratification: 30% near-death, 70% normal (if enough of each).
        """
        buf = list(self._buf)
        prios = np.array(self._priorities, dtype=np.float64)
        nd = np.array(self._near_death, dtype=bool)

        total = len(buf)
        if total == 0 or n == 0:
            return [], np.array([], dtype=int)

        n = min(n, total)
        probs = prios / prios.sum()

        # Simple stratified: try to give 30% of batch to near-death transitions
        nd_idx = np.where(nd)[0]
        norm_idx = np.where(~nd)[0]

        nd_target = min(round(n * 0.30), len(nd_idx))
        norm_target = n - nd_target

        chosen = []
        if nd_target > 0 and len(nd_idx) > 0:
            nd_probs = probs[nd_idx] / probs[nd_idx].sum()
            chosen_nd = np.random.choice(nd_idx, size=nd_target,
                                         replace=(len(nd_idx) < nd_target), p=nd_probs)
            chosen.extend(chosen_nd.tolist())

        if norm_target > 0:
            if len(norm_idx) >= norm_target:
                np_probs = probs[norm_idx] / probs[norm_idx].sum()
                chosen_n = np.random.choice(norm_idx, size=norm_target,
                                            replace=False, p=np_probs)
            else:
                # Fallback: sample from all
                chosen_n = np.random.choice(total, size=norm_target,
                                            replace=(total < norm_target), p=probs)
            chosen.extend(chosen_n.tolist())

        indices = np.array(chosen[:n], dtype=int)
        transitions = [buf[i] for i in indices]
        return transitions, indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities for sampled indices after a learn step."""
        prios = list(self._priorities)
        nd = list(self._near_death)
        for i, td in zip(indices, td_errors):
            if 0 <= i < len(prios):
                prios[i] = self._compute_priority(nd[i], float(td))
        self._priorities = deque(prios, maxlen=self.capacity)

    # ── Deque-compatible indexing ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx: int) -> Transition:
        return list(self._buf)[idx]


# ── N-Step Return Buffer ──────────────────────────────────────────────────


class NStepBuffer:
    """
    Accumulates n steps and flushes n-step return transitions to a downstream buffer.

    The flushed transition is:
      (s_t, a_t, G_t, s_{t+n}, done_t_or_later)
    where G_t = Σ_{k=0}^{n-1} γ^k * r_{t+k}

    The downstream buffer (StratifiedReplayDeque or plain deque) receives
    standard (s, a, r, s', done) tuples — the n-step G_t just replaces r.

    Parameters
    ----------
    n : int
        Number of steps to accumulate (default 4).
    gamma : float
        Discount factor (default 0.99).
    downstream : buffer
        Must support .push(s, a, r, s', done, near_death) or .append(tuple).
    """

    def __init__(self, n: int = 4, gamma: float = 0.99, downstream=None):
        self.n = n
        self.gamma = gamma
        self._downstream = downstream
        self._buf: List = []   # List of (s, a, r, s', done, near_death)
        self._gammas = np.array([gamma ** k for k in range(n)], dtype=np.float64)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        near_death: bool = False,
    ) -> None:
        """Add a new transition. Flushes to downstream when buffer reaches n steps."""
        self._buf.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
            bool(near_death),
        ))

        # On every push after n steps, or when done
        if len(self._buf) >= self.n or done:
            self._flush_one()

        if done:
            # Flush remaining partial sequences
            while self._buf:
                self._flush_one()

    def _flush_one(self) -> None:
        """Pop the oldest transition as an n-step return and push downstream."""
        if not self._buf:
            return

        # n-step discounted return from position 0
        n = min(self.n, len(self._buf))
        rewards = np.array([self._buf[k][2] for k in range(n)], dtype=np.float64)
        G = float(np.dot(self._gammas[:n], rewards))

        s0, a0, _, _, _, nd0 = self._buf[0]
        s_n, _, _, _, done_n, _ = self._buf[n - 1]
        # done = True if any step in the window ended the episode
        any_done = any(self._buf[k][4] for k in range(n))

        self._buf.pop(0)

        if self._downstream is not None:
            if hasattr(self._downstream, 'push'):
                self._downstream.push(s0, a0, G, s_n, any_done, near_death=nd0)
            else:
                self._downstream.append((s0, a0, G, s_n, any_done))

    def set_downstream(self, buf) -> None:
        """Attach or replace the downstream buffer."""
        self._downstream = buf
