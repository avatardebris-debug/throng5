"""
throng4/storage/human_play_logger.py
=====================================
Human play logging + SQLite indexer for the throng4 replay system.

Usage pattern
-------------
    logger = HumanPlayLogger()          # opens/creates replay_db.sqlite
    sid = logger.open_session("tetris", "human_play")
    eid = logger.open_episode(sid, episode_index=0, seed=42)

    for step_idx, (state_vec, human_act, agent_act, exec_act, reward, done) in enumerate(play):
        logger.log_step(sid, eid, step_idx, state_vec,
                        human_action=human_act, agent_action=agent_act,
                        executed_action=exec_act, reward=reward, done=done,
                        action_source="human", action_space_n=len(valid_actions))

    logger.close_episode(eid, total_reward, total_steps)
    logger.compute_derived(eid)        # fills transition_metrics
    logger.close()
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Default paths — overridable via constructor
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _REPO_ROOT / "experiments" / "replay_db.sqlite"
_DEFAULT_SCHEMA_PATH = _REPO_ROOT / "experiments" / "REPLAY_DB_SCHEMA.sql"


class HumanPlayLogger:
    """
    Logs human/agent play to SQLite using the Tetra-designed replay DB schema.

    The DB is created (or opened) on construction; the DDL comes from
    ``experiments/REPLAY_DB_SCHEMA.sql``.  All writes are committed lazily
    via ``close()`` — call it in a ``finally`` block or use the context
    manager form::

        with HumanPlayLogger() as logger:
            ...
    """

    # ------------------------------------------------------------------ #
    # Construction / teardown
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        schema_sql_path: Optional[str | Path] = None,
    ):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.schema_sql_path = (
            Path(schema_sql_path) if schema_sql_path else _DEFAULT_SCHEMA_PATH
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        """Execute DDL from REPLAY_DB_SCHEMA.sql (idempotent — uses IF NOT EXISTS)."""
        sql = self.schema_sql_path.read_text(encoding="utf-8")
        # sqlite3 module executes one statement at a time via executescript
        self._conn.executescript(sql)
        self._conn.commit()

    def close(self) -> None:
        """Commit any pending writes and close the connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "HumanPlayLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Session helpers
    # ------------------------------------------------------------------ #

    def open_session(
        self,
        env_name: str,
        source: str = "human_play",
        env_version: Optional[str] = None,
        rom_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session row.

        Parameters
        ----------
        env_name : str
            e.g. ``"tetris"`` or ``"ALE/Breakout-v5"``
        source : str
            ``"human_play"`` | ``"agent_selfplay"`` | ``"mixed"``
        config : dict, optional
            Arbitrary run config (JSON-serialised and stored in `config_json`).

        Returns
        -------
        str
            ``session_id`` (UUID4 string prefixed with ``"ses_"``).
        """
        session_id = "ses_" + str(uuid.uuid4())
        created_ms = _now_ms()
        config_json = json.dumps(config) if config else None
        self._conn.execute(
            """
            INSERT INTO sessions
                (session_id, created_at_ms, source, env_name, env_version, rom_id, config_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, created_ms, source, env_name, env_version, rom_id, config_json),
        )
        self._conn.commit()
        return session_id

    def open_episode(
        self,
        session_id: str,
        episode_index: int,
        seed: Optional[int] = None,
    ) -> str:
        """
        Create a new episode row.

        Returns
        -------
        str
            ``episode_id`` (UUID4 string prefixed with ``"ep_"``).
        """
        episode_id = "ep_" + str(uuid.uuid4())
        started_ms = _now_ms()
        self._conn.execute(
            """
            INSERT INTO episodes
                (episode_id, session_id, episode_index, started_at_ms, seed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (episode_id, session_id, episode_index, started_ms, seed),
        )
        self._conn.commit()
        return episode_id

    def close_episode(
        self,
        episode_id: str,
        total_reward: float,
        total_steps: int,
        final_score: Optional[float] = None,
        final_lives: Optional[int] = None,
        terminated: bool = True,
        truncated: bool = False,
    ) -> None:
        """Stamp the episode with end-of-game statistics."""
        ended_ms = _now_ms()
        self._conn.execute(
            """
            UPDATE episodes
            SET ended_at_ms  = ?,
                total_reward = ?,
                total_steps  = ?,
                final_score  = ?,
                final_lives  = ?,
                terminated   = ?,
                truncated    = ?
            WHERE episode_id = ?
            """,
            (
                ended_ms,
                total_reward,
                total_steps,
                final_score,
                final_lives,
                int(terminated),
                int(truncated),
                episode_id,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Step logging
    # ------------------------------------------------------------------ #

    def log_step(
        self,
        session_id: str,
        episode_id: str,
        step_idx: int,
        state_vec: Sequence[float],
        executed_action: int,
        action_source: str,
        action_space_n: int,
        reward: float,
        done: bool,
        *,
        human_action: Optional[int] = None,
        agent_action: Optional[int] = None,
        truncated: bool = False,
        score: Optional[float] = None,
        lives: Optional[int] = None,
        value_pred: Optional[float] = None,
        policy_entropy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> int:
        """
        Log one environment step.

        Writes one row to ``transitions`` and one skeleton row to
        ``transition_metrics`` (with placeholder priority = 1.0).
        Call :meth:`compute_derived` after the episode closes to fill
        the remaining metric columns.

        Parameters
        ----------
        state_vec : sequence of float
            The 84-dim abstract feature vector (or however long your adapter
            produces).  Stored inline in ``transition_metrics.abstract_vec_json``.

        Returns
        -------
        int
            ``transitions.id`` of the newly inserted row.
        """
        ts_ms = _now_ms()
        obs_ref = f"abstract://{session_id}/{episode_id}/{step_idx:06d}"
        tags_json = json.dumps(tags) if tags else None
        vec_json = json.dumps([float(v) for v in state_vec])
        ext_slots_active = _count_ext_slots_active(state_vec)

        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO transitions
                (session_id, episode_id, step_idx, timestamp_ms,
                 obs_ref, next_obs_ref, obs_shape_json,
                 human_action, agent_action, executed_action,
                 action_source, action_space_n,
                 reward, done, truncated,
                 score, lives,
                 value_pred, policy_entropy,
                 tags_json, notes)
            VALUES (?,?,?,?,  ?,?,?,  ?,?,?,  ?,?,  ?,?,?,  ?,?,  ?,?,  ?,?)
            """,
            (
                session_id, episode_id, step_idx, ts_ms,
                obs_ref, None, f"[{len(state_vec)}]",
                human_action, agent_action, executed_action,
                action_source, action_space_n,
                reward, int(done), int(truncated),
                score, lives,
                value_pred, policy_entropy,
                tags_json, notes,
            ),
        )
        transition_id = cur.lastrowid

        # Skeleton metrics row — will be completed by compute_derived()
        self._conn.execute(
            """
            INSERT OR IGNORE INTO transition_metrics
                (transition_id, abstract_vec_json, ext_slots_active,
                 priority_raw, priority_clipped, last_updated_ms)
            VALUES (?, ?, ?, 1.0, 1.0, ?)
            """,
            (transition_id, vec_json, ext_slots_active, ts_ms),
        )
        self._conn.commit()
        return transition_id

    # ------------------------------------------------------------------ #
    # Derived metrics (backward pass after episode close)
    # ------------------------------------------------------------------ #

    def compute_derived(
        self,
        episode_id: str,
        gamma: float = 0.99,
        n_step_5: int = 5,
        n_step_20: int = 20,
        near_death_threshold: float = -1.0,
        recovery_window: int = 5,
    ) -> int:
        """
        Compute and persist derived metrics for every transition in *episode_id*.

        Metrics computed
        ----------------
        - ``n_step_return_5`` / ``n_step_return_20``
        - ``time_to_terminal``  (steps remaining until done=1)
        - ``near_death_flag``   (reward < near_death_threshold)
        - ``recovery_flag``     (near_death followed by positive reward within window)
        - ``human_agent_disagree`` (human_action != agent_action, both not None)
        - ``priority_raw``      (v1 formula — see HUMAN_PLAY_SCHEMA.md)
        - ``priority_clipped``  (same as raw, clamped to [0.1, 10.0])
        - ``novelty_score``     (always 0.0 until embedding layer exists)

        Returns
        -------
        int
            Number of transitions updated.
        """
        # Fetch all transitions for this episode ordered by step_idx
        rows = self._conn.execute(
            """
            SELECT t.id, t.step_idx, t.reward, t.done,
                   t.human_action, t.agent_action
            FROM transitions t
            WHERE t.episode_id = ?
            ORDER BY t.step_idx ASC
            """,
            (episode_id,),
        ).fetchall()

        if not rows:
            return 0

        n = len(rows)
        ids = [r["id"] for r in rows]
        rewards = [r["reward"] for r in rows]
        dones = [r["done"] for r in rows]
        human_actions = [r["human_action"] for r in rows]
        agent_actions = [r["agent_action"] for r in rows]

        # --- n-step returns ---
        def nstep(i: int, horizon: int) -> float:
            ret, discount = 0.0, 1.0
            for j in range(i, min(i + horizon, n)):
                ret += discount * rewards[j]
                discount *= gamma
                if dones[j]:
                    break
            return ret

        ret5 = [nstep(i, n_step_5) for i in range(n)]
        ret20 = [nstep(i, n_step_20) for i in range(n)]

        # --- time_to_terminal ---
        # scan backwards: how many steps until the episode ends
        time_to_term = [0] * n
        t_remaining = 0
        for i in range(n - 1, -1, -1):
            time_to_term[i] = t_remaining
            if not dones[i]:
                t_remaining += 1
            else:
                t_remaining = 0

        # --- near_death_flag ---
        near_death = [int(rewards[i] < near_death_threshold) for i in range(n)]

        # --- recovery_flag: near_death then positive reward within window ---
        recovery = [0] * n
        for i in range(n):
            if near_death[i]:
                for j in range(i + 1, min(i + 1 + recovery_window, n)):
                    if rewards[j] > 0.0:
                        recovery[i] = 1
                        break

        # --- human_agent_disagree ---
        disagree = [
            int(
                human_actions[i] is not None
                and agent_actions[i] is not None
                and human_actions[i] != agent_actions[i]
            )
            for i in range(n)
        ]

        # --- priority formula v1 ---
        novelty = 0.0  # placeholder
        now_ms = _now_ms()

        updates = []
        for i in range(n):
            p_raw = (
                (1.5 if disagree[i] else 1.0)
                * (2.0 if near_death[i] else 1.0)
                * (1.0 + novelty)
            )
            p_clipped = float(min(max(p_raw, 0.1), 10.0))

            updates.append((
                ret5[i],
                ret20[i],
                time_to_term[i],
                near_death[i],
                recovery[i],
                disagree[i],
                novelty,
                p_raw,
                p_clipped,
                now_ms,
                ids[i],   # WHERE transition_id = ?
            ))

        self._conn.executemany(
            """
            UPDATE transition_metrics
            SET n_step_return_5   = ?,
                n_step_return_20  = ?,
                time_to_terminal  = ?,
                near_death_flag   = ?,
                recovery_flag     = ?,
                human_agent_disagree = ?,
                novelty_score     = ?,
                priority_raw      = ?,
                priority_clipped  = ?,
                last_updated_ms   = ?
            WHERE transition_id = ?
            """,
            updates,
        )
        self._conn.commit()
        return n


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

def _now_ms() -> int:
    return int(time.time() * 1000)


def _count_ext_slots_active(state_vec: Sequence[float]) -> int:
    """
    Count how many mask slots are set (=1.0) in the abstract feature vector.

    Layout: [core(20) | ext(32) | mask(32)] = 84 total.
    If the vector is shorter, returns 0.
    """
    CORE_SIZE = 20
    EXT_MAX = 32
    if len(state_vec) < CORE_SIZE + EXT_MAX * 2:
        return 0
    mask_start = CORE_SIZE + EXT_MAX
    mask_end = mask_start + EXT_MAX
    return int(sum(1 for v in state_vec[mask_start:mask_end] if v > 0.5))
