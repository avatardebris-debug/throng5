"""
throng4/learning/prioritized_replay.py
=======================================
Prioritized experience replay buffer — drop-in replacement for the
uniform ``ReplayBuffer`` in ``portable_agent.py``.

Two backends
------------
* **In-memory** (default, ``db_path=None``):
  Stores transitions in a Python deque for fast prototype runs.
  All priority logic is computed in Python and stored per-entry.

* **SQLite-backed** (``db_path=<path>``):
  Delegates to the ``transition_metrics`` view populated by
  ``HumanPlayLogger.compute_derived()``.  All writes go through
  the logger — this buffer is read-only; call ``push_db()`` only
  when you also have a logger open.

Stratified sampling
-------------------
Transitions are bucketed by flag type, then sampled proportionally
to ``priority_clipped``:

  Bucket A — ``human_agent_disagree=True``  (multiplier 1.5×)
  Bucket B — ``near_death_flag=True``        (multiplier 2.0×)
  Bucket C — everything else                 (multiplier 1.0×)

Each bucket draws ``batch_size // n_buckets`` samples (remainder
from high-priority bucket A).  If a bucket is too small, its
allocation is redistributed to the others.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Priority formula constants (v1 — before value head)
_DISAGREE_MULT = 1.5
_NEAR_DEATH_MULT = 2.0


# ------------------------------------------------------------------ #
# In-memory entry
# ------------------------------------------------------------------ #

class _Entry:
    """Single transition stored in the in-memory buffer."""

    __slots__ = (
        "x", "reward", "next_x_list", "done",
        "human_action", "agent_action",
        "near_death", "novelty_score",
        "priority_raw", "priority_clipped",
        "human_agent_disagree",
    )

    def __init__(
        self,
        x: np.ndarray,
        reward: float,
        next_x_list: List[np.ndarray],
        done: bool,
        human_action: Optional[int] = None,
        agent_action: Optional[int] = None,
        near_death: bool = False,
        novelty_score: float = 0.0,
    ):
        self.x = x
        self.reward = reward
        self.next_x_list = next_x_list
        self.done = done
        self.human_action = human_action
        self.agent_action = agent_action
        self.near_death = near_death
        self.novelty_score = novelty_score

        disagree = (
            human_action is not None
            and agent_action is not None
            and human_action != agent_action
        )
        self.human_agent_disagree = disagree
        self.priority_raw = _compute_priority(disagree, near_death, novelty_score)
        self.priority_clipped = float(min(max(self.priority_raw, 0.1), 10.0))


def _compute_priority(
    disagree: bool, near_death: bool, novelty: float
) -> float:
    return (
        (_DISAGREE_MULT if disagree else 1.0)
        * (_NEAR_DEATH_MULT if near_death else 1.0)
        * (1.0 + novelty)
    )


# ------------------------------------------------------------------ #
# PrioritizedReplayBuffer
# ------------------------------------------------------------------ #

class PrioritizedReplayBuffer:
    """
    Stratified priority-weighted experience replay.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to keep (in-memory mode).
    rng : np.random.RandomState
        Shared RNG (passed in from the agent to keep runs reproducible).
    db_path : str or None
        If given, also opens an SQLite connection for DB-backed sampling.
        In-memory push/sample still works regardless of this flag.
    """

    def __init__(
        self,
        capacity: int,
        rng: np.random.RandomState,
        db_path: Optional[str] = None,
    ):
        self.capacity = capacity
        self.rng = rng
        self._buffer: deque[_Entry] = deque(maxlen=capacity)

        # SQLite connection (optional — for hybrid logger + replay workflows)
        self._conn: Optional[sqlite3.Connection] = None
        if db_path is not None:
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------ #
    # Push (in-memory path)
    # ------------------------------------------------------------------ #

    def push(
        self,
        x: np.ndarray,
        reward: float,
        next_x_list: List[np.ndarray],
        done: bool,
        human_action: Optional[int] = None,
        agent_action: Optional[int] = None,
        near_death: bool = False,
        novelty_score: float = 0.0,
    ) -> None:
        """
        Add a transition to the in-memory buffer.

        This is the primary write path used by ``PortableNNAgent`` (mirrors
        the ``ReplayBuffer.push`` signature with extra optional fields).
        """
        entry = _Entry(
            x=x.copy(),
            reward=reward,
            next_x_list=next_x_list,
            done=done,
            human_action=human_action,
            agent_action=agent_action,
            near_death=near_death,
            novelty_score=novelty_score,
        )
        self._buffer.append(entry)

    # ------------------------------------------------------------------ #
    # Sample (in-memory path)
    # ------------------------------------------------------------------ #

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, float, List[np.ndarray], bool]]:
        """
        Return a stratified priority-weighted batch.

        Return format is identical to the old ``ReplayBuffer.sample()``
        so that ``_train_batch`` requires no changes::

            [(x, reward, next_x_list, done), ...]

        Sampling strategy
        -----------------
        Split ``batch_size`` across three flag buckets; within each
        bucket draw proportional to ``priority_clipped``.  If a bucket
        has fewer entries than its allocation, the deficit is added to
        the largest bucket.
        """
        buf = list(self._buffer)
        n = len(buf)
        if n == 0:
            return []

        # Bucket membership
        bucket_a = [e for e in buf if e.human_agent_disagree]
        bucket_b = [e for e in buf if not e.human_agent_disagree and e.near_death]
        bucket_c = [e for e in buf if not e.human_agent_disagree and not e.near_death]

        alloc_a = batch_size // 3
        alloc_b = batch_size // 3
        alloc_c = batch_size - alloc_a - alloc_b

        samples: List[_Entry] = []
        samples += self._sample_bucket(bucket_a, alloc_a)
        samples += self._sample_bucket(bucket_b, alloc_b)
        # Drain bucket_c with whatever's left (includes alloc_c + any a/b shortfall)
        remaining_c = batch_size - len(samples)
        samples += self._sample_bucket(bucket_c, remaining_c)

        # Final fallback: guarantee exactly batch_size via uniform with replacement
        remaining = batch_size - len(samples)
        if remaining > 0 and buf:
            extra_idx = self.rng.choice(n, size=remaining, replace=True)
            samples += [buf[i] for i in extra_idx]

        # Return in (x, reward, next_x_list, done) format to match old API
        return [(e.x, e.reward, e.next_x_list, e.done) for e in samples]

    def _sample_bucket(
        self, bucket: List[_Entry], k: int
    ) -> List[_Entry]:
        """Priority-weighted sample of k items from a bucket (no replacement)."""
        if not bucket or k <= 0:
            return []
        k = min(k, len(bucket))
        priorities = np.array([e.priority_clipped for e in bucket], dtype=np.float64)
        total = priorities.sum()
        if total <= 0.0:
            probs = None  # uniform
        else:
            probs = priorities / total
            # Guard against float rounding making sum != 1.0 exactly
            probs = probs / probs.sum()
        replace = k > len(bucket)   # replacement only if we MUST over-sample
        indices = self.rng.choice(len(bucket), size=k, replace=replace, p=probs)
        return [bucket[i] for i in indices]

    # ------------------------------------------------------------------ #
    # Priority update hook (for TD-error PER — future use)
    # ------------------------------------------------------------------ #

    def update_priorities(
        self,
        indices: Sequence[int],
        td_errors: Sequence[float],
    ) -> None:
        """
        Update priority for specific buffer entries by index.

        Currently a no-op stub — will be wired to TD-error PER once the
        value head is added.  Signature matches standard PER libraries.
        """
        pass  # TODO: wire to |td_error| once value head exists

    # ------------------------------------------------------------------ #
    # DB-backed sampling (optional)
    # ------------------------------------------------------------------ #

    def sample_from_db(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Sample from the SQLite ``v_priority_replay`` view.

        Requires ``db_path`` to have been provided at construction.
        Returns dicts with keys matching column names (not the simple
        tuple format — use this path for offline training from logged data).
        """
        if self._conn is None:
            raise RuntimeError(
                "db_path was not provided — cannot sample from DB"
            )
        rows = self._conn.execute(
            """
            SELECT id, episode_id, step_idx, obs_ref, next_obs_ref,
                   executed_action, reward, done,
                   priority_clipped, sampling_weight,
                   near_death_flag, recovery_flag, human_agent_disagree,
                   abstract_vec_json
            FROM v_priority_replay
            LIMIT ?
            """,
            (batch_size * 10,),   # over-fetch then re-sample by priority
        ).fetchall()

        if not rows:
            return []

        priorities = np.array(
            [r["priority_clipped"] if r["priority_clipped"] else 1.0 for r in rows],
            dtype=np.float64,
        )
        probs = priorities / priorities.sum()
        k = min(batch_size, len(rows))
        chosen = self.rng.choice(len(rows), size=k, replace=False, p=probs)
        return [dict(rows[i]) for i in chosen]

    # ------------------------------------------------------------------ #
    # Dunder
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._buffer)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]
