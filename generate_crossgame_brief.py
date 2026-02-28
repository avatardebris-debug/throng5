"""
generate_crossgame_brief.py
============================
Phase 3: Generate a multi-game brief for Tetra that enables cross-domain
principle extraction. Aggregates episode histories from all round-robin
games plus any existing per-game brief data.

Output: experiments/crossgame_brief.json

Usage:
    python generate_crossgame_brief.py
"""
from __future__ import annotations
import json, time
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent
_RESULTS = _ROOT / "benchmark_results"
_EXP     = _ROOT / "experiments"
_OUT     = _EXP / "crossgame_brief.json"

# Same roster as round_robin_runner to stay in sync
from round_robin_runner import GAME_ROSTER


def _game_summary(cfg: dict) -> dict:
    slug  = cfg["id"].replace("/","_").replace("-","_")
    ep_f  = _RESULTS / f"{slug}_rr_episodes.json"
    episodes = []
    if ep_f.exists():
        try:
            episodes = json.loads(ep_f.read_text(encoding="utf-8"))
        except Exception:
            pass

    n = len(episodes)
    if n == 0:
        return {
            "game":             cfg["id"],
            "environment_type": cfg,
            "n_episodes":       0,
            "status":           "no_data",
        }

    rewards  = [e.get("game_reward", 0)    for e in episodes]
    steps    = [e.get("steps", 0)          for e in episodes]
    reliance = [e.get("dreamer_reliance",0) for e in episodes if e.get("dreamer_reliance") is not None]

    recent_20 = episodes[-20:]
    trend_dir = "improving" if (
        len(recent_20) >= 10 and
        sum(e["game_reward"] for e in recent_20[-10:]) >
        sum(e["game_reward"] for e in recent_20[:10])
    ) else "flat_or_declining"

    return {
        "game":             cfg["id"],
        "environment_type": {
            "stochastic":   cfg["stochastic"],
            "reward_type":  cfg["reward_type"],
            "notes":        cfg.get("notes",""),
        },
        "n_episodes":       n,
        "reward_stats": {
            "mean":    round(sum(rewards)/n,    3),
            "max":     round(max(rewards),      3),
            "min":     round(min(rewards),      3),
            "recent20_mean": round(sum(e["game_reward"] for e in recent_20)/len(recent_20), 3),
            "trend":   trend_dir,
        },
        "step_stats": {
            "mean": round(sum(steps)/n, 1),
            "max":  max(steps),
        },
        "dreamer_stats": {
            "mean_reliance":  round(sum(reliance)/len(reliance), 4) if reliance else None,
            "final_reliance": round(reliance[-1], 4) if reliance else None,
            "calibrated_by_ep": next((e["episode"] for e in episodes
                                      if e.get("dreamer_calibrated")), None),
        },
        "episode_sample": episodes[-5:],   # last 5 for context
    }


def main() -> None:
    import importlib
    game_summaries = [_game_summary(cfg) for cfg in GAME_ROSTER]
    n_with_data    = sum(1 for g in game_summaries if g.get("n_episodes",0) > 0)

    # Load existing principles
    principles_path = _EXP / "principles.json"
    principles = []
    if principles_path.exists():
        try:
            principles = json.loads(principles_path.read_text(encoding="utf-8")).get("principles",[])
        except Exception:
            pass

    brief = {
        "schema_version":  "crossgame_v1",
        "brief_type":      "crossgame_principle_brief",
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_games":         len(GAME_ROSTER),
        "n_games_with_data": n_with_data,
        "games":           game_summaries,
        "existing_principles": principles,
        "inbox_schema": {
            "description": (
                "Respond with a valid JSON array of ops. "
                "See experiments/CROSSGAME_PROMPT.md for full instructions. "
                "Supported ops: ADD (game hypothesis), RETIRE, MUTATE, PRINCIPLE (cross-game)."
            ),
            "principle_op_format": {
                "op":         "PRINCIPLE",
                "id":         "p_<slug>",
                "text":       "Human-readable principle statement",
                "env_class":  {"stochastic": "bool", "sparse": "bool"},
                "params": {
                    "dream_interval":    "int",
                    "advisory_rate":     "float 0-1",
                    "promote_threshold": "int",
                    "epsilon_decay":     "float",
                    "gamma":             "float",
                    "label":             "str",
                },
                "evidence":    ["list of game+ep range strings"],
                "confidence":  "float 0-1",
            },
        },
    }

    _EXP.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(brief, indent=2), encoding="utf-8")
    size_kb = _OUT.stat().st_size / 1024

    print(f"crossgame_brief.json written ({size_kb:.1f} KB)")
    print(f"  Games: {len(GAME_ROSTER)}  With data: {n_with_data}")
    print(f"  Existing principles: {len(principles)}")
    print()
    print("Next steps:")
    print("  1. Copy system prompt from experiments/CROSSGAME_PROMPT.md")
    print("  2. Paste experiments/crossgame_brief.json as user message to Tetra")
    print("  3. Save Tetra's JSON array response to experiments/crossgame_inbox.json")
    print("  4. Run: python ingest_crossgame_inbox.py")


if __name__ == "__main__":
    main()
