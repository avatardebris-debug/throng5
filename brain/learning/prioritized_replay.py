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

    def seed_from_db(
        self,
        db_path: str,
        env_name: str,
        n_actions: int,
        max_rows: int = 5000,
        verbose: bool = True,
    ) -> int:
        """
        Load human play transitions from replay_db.sqlite into the in-memory buffer.

        Feature reconstruction
        ----------------------
        The DB stores the raw RAM vector (128 floats) as ``abstract_vec_json``
        in ``transition_metrics``.  We reconstruct the full agent feature as::

            feat = [ram_vec (128) | one_hot(executed_action, n_actions)]

        For the *next* state we shift by one step: the next row's ``abstract_vec_json``
        is used, with one feature per possible action (so the agent can score them all).
        Consecutive transitions within the same episode are matched by step order.

        Parameters
        ----------
        db_path   : Path to replay_db.sqlite
        env_name  : Game ID to filter (e.g. 'ALE/Breakout-v5')
        n_actions : Action space size — used to build one-hot action encoding
        max_rows  : Maximum number of transitions to load (newest first)
        verbose   : Print summary line when done

        Returns
        -------
        Number of transitions loaded.
        """
        import json as _json
        import sqlite3 as _sqlite3

        con = _sqlite3.connect(db_path)
        con.row_factory = _sqlite3.Row

        # Pull: state_vec, next_state_vec (lead by 1 within episode),
        # executed_action, reward, done, near_death, disagree
        rows = con.execute("""
            SELECT
                t.episode_id,
                t.step_idx,
                t.executed_action,
                t.reward,
                t.done,
                t.human_action,
                t.agent_action,
                m.near_death_flag,
                m.human_agent_disagree,
                m.priority_clipped,
                m.abstract_vec_json
            FROM transitions t
            JOIN transition_metrics m ON m.transition_id = t.id
            JOIN episodes e ON e.episode_id = t.episode_id
            JOIN sessions s ON s.session_id = e.session_id
            WHERE s.env_name = ?
              AND m.abstract_vec_json IS NOT NULL
              AND t.executed_action IS NOT NULL
            ORDER BY t.episode_id, t.step_idx
        """, (env_name,)).fetchall()
        con.close()

        if not rows:
            if verbose:
                print(f"  [seed_from_db] No rows found for {env_name}")
            return 0

        # Build per-episode step lookup for next-state matching
        # key: (episode_id, step_idx) → row
        lookup: dict = {}
        for r in rows:
            lookup[(r["episode_id"], r["step_idx"])] = r

        loaded = 0
        for r in rows[-max_rows:]:   # newest max_rows
            try:
                ram = np.array(_json.loads(r["abstract_vec_json"]),
                               dtype=np.float32)
                if len(ram) != 128:
                    continue
                # Current feature
                ah = np.zeros(n_actions, dtype=np.float32)
                act = int(r["executed_action"])
                if 0 <= act < n_actions:
                    ah[act] = 1.0
                feat = np.concatenate([ram, ah])

                # Next state features (one per possible action for Bellman max)
                nxt_key = (r["episode_id"], r["step_idx"] + 1)
                next_feats: List[np.ndarray] = []
                if nxt_key in lookup and not r["done"]:
                    nxt_row = lookup[nxt_key]
                    nxt_vec = nxt_row["abstract_vec_json"]
                    if nxt_vec:
                        nxt_ram = np.array(_json.loads(nxt_vec),
                                           dtype=np.float32)
                        for a in range(n_actions):
                            nah = np.zeros(n_actions, dtype=np.float32)
                            nah[a] = 1.0
                            nf = np.concatenate([nxt_ram, nah])
                            if np.isfinite(nf).all():
                                next_feats.append(nf)

                self.push(
                    x=feat,
                    reward=float(r["reward"]),
                    next_x_list=next_feats,
                    done=bool(r["done"]),
                    human_action=r["human_action"],
                    agent_action=r["agent_action"],
                    near_death=bool(r["near_death_flag"]),
                )
                loaded += 1
            except Exception:
                continue

        if verbose:
            print(f"  [seed_from_db] Loaded {loaded}/{len(rows)} transitions "
                  f"for {env_name} (buffer size now {len(self._buffer)})")
        return loaded



    # ------------------------------------------------------------------ #
    # Sample (in-memory path)
    # ------------------------------------------------------------------ #

    def sample(
        self, batch_size: int
    ) -> Tuple[List[Tuple[np.ndarray, float, List[np.ndarray], bool]], List[int]]:
        """
        Return a stratified priority-weighted batch AND the buffer indices
        so callers can feed back TD errors via update_priorities().

        Returns
        -------
        (transitions, indices)
        transitions : List of (x, reward, next_x_list, done)
        indices     : List[int] — positions in _buffer for TD-error feedback
        """
        buf = list(self._buffer)
        n = len(buf)
        if n == 0:
            return [], []

        # Bucket membership (with buffer-level indices tracked)
        bucket_a = [(i, e) for i, e in enumerate(buf) if e.human_agent_disagree]
        bucket_b = [(i, e) for i, e in enumerate(buf) if not e.human_agent_disagree and e.near_death]
        bucket_c = [(i, e) for i, e in enumerate(buf) if not e.human_agent_disagree and not e.near_death]

        alloc_a = batch_size // 3
        alloc_b = batch_size // 3
        alloc_c = batch_size - alloc_a - alloc_b

        sampled: List[Tuple[int, '_Entry']] = []
        sampled += self._sample_bucket_indexed(bucket_a, alloc_a)
        sampled += self._sample_bucket_indexed(bucket_b, alloc_b)
        remaining_c = batch_size - len(sampled)
        sampled += self._sample_bucket_indexed(bucket_c, remaining_c)

        # Final fallback: priority-weighted (MINOR-3 fix — was uniform)
        remaining = batch_size - len(sampled)
        if remaining > 0 and buf:
            priorities = np.array([e.priority_clipped for e in buf], dtype=np.float64)
            probs = priorities / priorities.sum()
            extra_idx = self.rng.choice(n, size=remaining, replace=True, p=probs)
            sampled += [(int(i), buf[i]) for i in extra_idx]

        indices = [i for i, _ in sampled]
        transitions = [(e.x, e.reward, e.next_x_list, e.done) for _, e in sampled]
        return transitions, indices

    def _sample_bucket_indexed(
        self, bucket: List[Tuple[int, '_Entry']], k: int
    ) -> List[Tuple[int, '_Entry']]:
        """Priority-weighted sample of k (index, entry) pairs from a bucket."""
        if not bucket or k <= 0:
            return []
        k = min(k, len(bucket))
        priorities = np.array([e.priority_clipped for _, e in bucket], dtype=np.float64)
        total = priorities.sum()
        if total <= 0.0:
            probs = None
        else:
            probs = priorities / total
            probs = probs / probs.sum()   # guard float rounding
        replace = k > len(bucket)
        chosen = self.rng.choice(len(bucket), size=k, replace=replace, p=probs)
        return [bucket[i] for i in chosen]

    # Legacy helper — kept for internal compatibility
    def _sample_bucket(
        self, bucket: List['_Entry'], k: int
    ) -> List['_Entry']:
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
            probs = probs / probs.sum()
        replace = k > len(bucket)
        indices = self.rng.choice(len(bucket), size=k, replace=replace, p=probs)
        return [bucket[i] for i in indices]

    # ------------------------------------------------------------------ #
    # Priority update hook (for TD-error PER — future use)
    # ------------------------------------------------------------------ #

    def update_priorities(
        self,
        indices: Sequence[int],
        td_errors: Sequence[float],
        alpha: float = 0.6,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Update priority for specific buffer entries by index (CRITICAL-1 fix).

        Uses standard PER formula: priority = (|td_error| + epsilon)^alpha
        This ensures high-error transitions are sampled more frequently,
        providing the core PER benefit that was previously a no-op stub.

        Parameters
        ----------
        indices   : Buffer positions returned by sample()
        td_errors : Corresponding |TD-error| values from train_step()
        alpha     : Priority exponent (0.6 standard, 0=uniform, 1=full PER)
        epsilon   : Small constant preventing zero priority
        """
        buf = list(self._buffer)   # needed to index by position
        for idx, err in zip(indices, td_errors):
            if 0 <= idx < len(buf):
                raw = float((abs(float(err)) + epsilon) ** alpha)
                buf[idx].priority_raw = raw
                buf[idx].priority_clipped = float(min(max(raw, 0.1), 10.0))

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
