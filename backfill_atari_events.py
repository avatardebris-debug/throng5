"""
backfill_atari_events.py
========================
Retroactively generate JSONL event files from existing replay_db.sqlite
sessions, so Tetra gets analysis data on everything played before the
AtariEventLogger was added.

What we reconstruct per step
-----------------------------
human_action  ← t.human_action  (exact)
agent_topk    ← synthesised from t.agent_action  (we know the greedy choice;
                estimated confidence based on disagree flag)
entropy       ← estimated (low if agree / high if disagree)
disagree      ← m.human_agent_disagree  (exact DB flag)
high_conf_disagree ← same as disagree (agent was taking a definite action)
reward        ← t.reward  (exact)
done          ← t.done   (exact)
near_death    ← m.near_death_flag  (exact)

Caveats
-------
agent_topk only shows the greedy action, not full distribution rankings.
entropy is estimated, not computed from real Q-values.
Both fields are marked with "_reconstructed": true in the event.

Usage
-----
    python backfill_atari_events.py                  # all sessions in DB
    python backfill_atari_events.py --game ALE/MontezumaRevenge-v5
    python backfill_atari_events.py --dry-run        # count only, no writes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT    = Path(__file__).resolve().parent
_DB_PATH = _ROOT / "experiments" / "replay_db.sqlite"
_EVENTS_DIR = _ROOT / "experiments" / "atari_events"

# ─────────────────────────────────────────────────────────────────────
# ALE action meanings cache (avoid re-making envs repeatedly)
# ─────────────────────────────────────────────────────────────────────

_ACTION_CACHE: dict[str, list[str]] = {}

def _get_action_meanings(game_id: str) -> list[str]:
    if game_id in _ACTION_CACHE:
        return _ACTION_CACHE[game_id]
    try:
        import gymnasium as gym
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make(game_id, obs_type="ram", render_mode=None)
        meanings = list(env.unwrapped.get_action_meanings())
        env.close()
        _ACTION_CACHE[game_id] = meanings
        return meanings
    except Exception as exc:
        print(f"  [warn] Could not get action meanings for {game_id}: {exc}")
        return [str(i) for i in range(18)]   # fallback: numeric strings


# ─────────────────────────────────────────────────────────────────────
# Event reconstruction from DB row
# ─────────────────────────────────────────────────────────────────────

def _estimate_confidence(disagree: bool) -> float:
    """
    Agent confidence estimate when we don't have full Q-values.
    Agree → high confidence (agent was right).
    Disagree → moderate confidence (it still chose something).
    """
    return 0.85 if not disagree else 0.70


def _make_event(
    row: sqlite3.Row,
    game_id: str,
    episode_num: int,
    action_meanings: list[str],
) -> dict:
    n = len(action_meanings)
    human_idx   = int(row["human_action"] or row["executed_action"] or 0)
    agent_idx   = int(row["agent_action"] or 0)
    disagree    = bool(row["human_agent_disagree"])
    near_death  = bool(row["near_death_flag"])
    conf        = _estimate_confidence(disagree)

    human_idx  = min(human_idx, n - 1)
    agent_idx  = min(agent_idx, n - 1)

    # Synthesise topk — just the greedy action at estimated confidence
    topk = [{"action": action_meanings[agent_idx], "p": round(conf, 4)}]
    if disagree and human_idx != agent_idx:
        # Show the human action too so Tetra can see the split
        topk.append({"action": action_meanings[human_idx],
                     "p": round(1.0 - conf - 0.05, 4)})

    # Entropy estimate: peaked if confident, flatter if disagreeing
    entropy_est = round(0.3 + (1.8 if disagree else 0.2), 4)

    return {
        "game":               game_id,
        "episode":            episode_num,
        "step":               int(row["step_idx"]),
        "human_action":       action_meanings[human_idx],
        "agent_topk":         topk,
        "entropy":            entropy_est,
        "disagree":           disagree,
        "high_conf_disagree": disagree,   # agent always took a definite action
        "reward":             round(float(row["reward"]), 4),
        "done":               bool(row["done"]),
        "flags": {
            "near_death":  near_death,
            "novel_state": False,
        },
        "_reconstructed": True,   # marks this as backfilled, not live-logged
    }


# ─────────────────────────────────────────────────────────────────────
# Per-session backfill
# ─────────────────────────────────────────────────────────────────────

def backfill_session(
    con: sqlite3.Connection,
    session_id: str,
    game_id: str,
    dry_run: bool = False,
) -> int:
    """
    Write one JSONL file for a single DB session.
    Returns number of events written.
    """
    action_meanings = _get_action_meanings(game_id)
    n = len(action_meanings)

    rows = con.execute("""
        SELECT
            t.step_idx, t.executed_action, t.human_action, t.agent_action,
            t.reward, t.done, t.episode_id,
            m.near_death_flag, m.human_agent_disagree
        FROM transitions t
        JOIN transition_metrics m ON m.transition_id = t.id
        WHERE t.session_id = ?
          AND m.abstract_vec_json IS NOT NULL
        ORDER BY t.episode_id, t.step_idx
    """, (session_id,)).fetchall()

    if not rows:
        return 0

    # Map episode_id → sequential episode number
    ep_order: dict[str, int] = {}
    for r in rows:
        ep_id = r["episode_id"]
        if ep_id not in ep_order:
            ep_order[ep_id] = len(ep_order)

    if dry_run:
        return len(rows)

    # Write JSONL
    slug = game_id.replace("/", "_").replace("-", "_")
    out_dir = _EVENTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}_backfill.jsonl"

    # Skip if already done
    if out_path.exists():
        return 0

    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            ep_num = ep_order[r["episode_id"]]
            event = _make_event(r, game_id, ep_num, action_meanings)
            fh.write(json.dumps(event) + "\n")

    return len(rows)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def backfill(
    game_filter: str | None = None,
    db_path: Path = _DB_PATH,
    dry_run: bool = False,
) -> None:
    from throng4.storage.atari_event import update_brief

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    sessions = con.execute(
        "SELECT session_id, env_name FROM sessions WHERE env_name IS NOT NULL"
    ).fetchall()

    # Group by game
    by_game: dict[str, list[str]] = defaultdict(list)
    for s in sessions:
        game = s["env_name"]
        if game_filter and game != game_filter:
            continue
        by_game[game].append(s["session_id"])

    total_events = 0
    for game, sess_ids in sorted(by_game.items()):
        game_events = 0
        for sid in sess_ids:
            n = backfill_session(con, sid, game, dry_run=dry_run)
            game_events += n

        total_events += game_events
        status = "would write" if dry_run else "wrote"
        print(f"  {game:<45} {status} {game_events:>6} events "
              f"({len(sess_ids)} session{'s' if len(sess_ids)!=1 else ''})")

        # Update brief for this game (skip in dry run)
        if not dry_run and game_events > 0:
            try:
                update_brief(game)
                print(f"    +-- brief updated")
            except Exception as exc:
                print(f"    +-- brief update failed: {exc}")

    con.close()
    print(f"\nTotal: {total_events} events {'(dry run)' if dry_run else 'backfilled'}")


def _parse():
    p = argparse.ArgumentParser(
        description="Backfill Atari event JSONL from existing replay_db.sqlite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game", default=None, help="Only backfill this game ID")
    p.add_argument("--db",   default=str(_DB_PATH))
    p.add_argument("--dry-run", action="store_true",
                   help="Count events but don't write any files")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    import sys
    sys.path.insert(0, str(_ROOT))
    print(f"Backfilling from {args.db}")
    print(f"{'DRY RUN — ' if args.dry_run else ''}Writing to {_EVENTS_DIR}\n")
    backfill(
        game_filter=args.game,
        db_path=Path(args.db),
        dry_run=args.dry_run,
    )
