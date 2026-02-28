"""
generate_atari_cron_brief.py
============================
Cron step 4 — generate a fresh atari_brief_cron.json for sending to the LLM.

Combines:
  1. experiments/atari_brief.json           (aggregate alignment stats, all games)
  2. experiments/atari_events/<game>/       latest *_semantic.json eval file per game
  3. experiments/atari_hypotheses.json      active/retired hypothesis ledger

Output: experiments/atari_brief_cron.json

Usage:
    python generate_atari_cron_brief.py

Cron sequence:
    4. python generate_atari_cron_brief.py
    5. [LLM reads ATARI_PROMPT.md + atari_brief_cron.json, writes atari_inbox.json]
    6. python ingest_atari_inbox.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT       = Path(__file__).resolve().parent
_EXP        = _ROOT / "experiments"
_BRIEF_SRC  = _EXP / "atari_brief.json"
_HYPS_FILE  = _EXP / "atari_hypotheses.json"
_EVENTS_DIR = _EXP / "atari_events"
_OUT        = _EXP / "atari_brief_cron.json"


def _load_latest_semantic(game_id: str) -> dict | None:
    """Find the most recently written *_semantic.json for the given game."""
    slug = game_id.replace("/", "_").replace("-", "_")
    game_dir = _EVENTS_DIR / slug
    if not game_dir.exists():
        return None
    candidates = sorted(game_dir.glob("*_semantic.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        # Tag which file we used
        data["_source_file"] = candidates[0].name
        return data
    except Exception:
        return None


def _load_hypotheses() -> dict:
    """Load atari_hypotheses.json (hypothesis ledger for Atari)."""
    if _HYPS_FILE.exists():
        try:
            return json.loads(_HYPS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active": [], "retired": [], "candidates": []}


def main() -> None:
    if not _BRIEF_SRC.exists():
        print(f"ERROR: {_BRIEF_SRC} not found.")
        print("Run: python generate_atari_brief.py   (or eval_atari_agent.py) first.")
        sys.exit(1)

    base_brief   = json.loads(_BRIEF_SRC.read_text(encoding="utf-8"))
    hypotheses   = _load_hypotheses()

    # Enrich game entries with latest semantic eval data
    enriched_games = []
    n_with_eval = 0
    for game_entry in base_brief.get("games", []):
        game_id = game_entry["game"]
        semantic = _load_latest_semantic(game_id)
        entry = dict(game_entry)

        if semantic:
            n_with_eval += 1
            # Summarise episodes for LLM (trim trajectory snapshots for size)
            eps_summary = []
            for ep in semantic.get("episodes", []):
                snapshots = ep.get("trajectory_snapshots", [])
                # Keep at most 4 snapshots to keep brief size reasonable
                eps_summary.append({
                    "episode":           ep["episode"],
                    "total_reward":      ep["total_reward"],
                    "total_steps":       ep["total_steps"],
                    "rooms_visited":     ep.get("rooms_visited", []),
                    "max_room":          ep.get("max_room", 0),
                    "key_collected_any": ep.get("key_collected_any", False),
                    "mean_entropy":      ep.get("mean_entropy"),
                    "trajectory_snapshots": snapshots[:4],
                })
            entry["latest_eval"] = {
                "label":        semantic.get("label"),
                "decoder":      semantic.get("decoder"),
                "source_file":  semantic.get("_source_file"),
                "episodes":     eps_summary,
            }
        enriched_games.append(entry)

    # Build the cron brief
    cron_brief = {
        "schema_version":    "atari_v1",
        "brief_type":        "atari_hypothesis_brief",
        "generated_for":     "cron",
        "n_games":           base_brief.get("n_games", len(enriched_games)),
        "last_updated":      base_brief.get("last_updated"),
        "games":             enriched_games,
        "hypothesis_ledger": hypotheses,
        "inbox_schema": {
            "description": (
                "Respond with ONLY a valid JSON array of ops. "
                "Read experiments/ATARI_PROMPT.md for full instructions."
            ),
            "supported_ops": ["ADD", "RETIRE", "MUTATE"],
            "required_per_op": {
                "ADD":    ["op", "name", "description", "game",
                           "llm_score", "llm_priority", "llm_notes", "enaction"],
                "RETIRE": ["op", "name", "llm_notes"],
                "MUTATE": ["op", "parent", "name", "description", "game",
                           "llm_score", "llm_priority", "llm_notes", "enaction"],
            },
            "enaction_types": {
                "priority_boost": {
                    "fields": {"condition": "near_death|room_boundary|high_entropy|disagree",
                               "multiplier": "float (e.g. 3.0)"},
                },
                "imitation_weight": {
                    "fields": {"action": "action name or ALL", "alpha": "float 0-1"},
                },
                "phase_extend": {
                    "fields": {"phase_steps": "int (override imitation_phase_steps)"},
                },
            },
        },
    }

    _OUT.write_text(json.dumps(cron_brief, indent=2), encoding="utf-8")

    n_active   = len(hypotheses.get("active", []))
    n_cand     = len(hypotheses.get("candidates", []))
    n_retired  = len(hypotheses.get("retired", []))
    size_kb    = _OUT.stat().st_size / 1024

    print(f"atari_brief_cron.json written ({size_kb:.1f} KB)")
    print(f"  Games: {len(enriched_games)}  |  With eval results: {n_with_eval}")
    print(f"  Hypotheses: {n_active} active  "
          f"{n_cand} candidates  {n_retired} retired")
    print()
    print("Next steps:")
    print("  1. Copy system prompt from experiments/ATARI_PROMPT.md")
    print("  2. Paste experiments/atari_brief_cron.json as user message")
    print("  3. Save the LLM JSON array response to experiments/atari_inbox.json")
    print("  4. Run: python ingest_atari_inbox.py")


if __name__ == "__main__":
    main()
