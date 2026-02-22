"""
generate_atari_brief.py
=======================
Aggregates per-step AtariEventLogger JSONL files into a structured
brief for Tetra to analyze.

Reads:  experiments/atari_events/<game_slug>/*.jsonl
Also:   benchmark_results/comparison.json  (if available)
Writes: experiments/atari_brief.json

Usage
-----
    python generate_atari_brief.py
    python generate_atari_brief.py --games ALE/MontezumaRevenge-v5 ALE/Breakout-v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from throng4.storage.atari_event import aggregate_all, _EVENTS_DIR

BRIEF_PATH      = _ROOT / "experiments" / "atari_brief.json"
COMPARISON_PATH = _ROOT / "benchmark_results" / "comparison.json"

# ─────────────────────────────────────────────────────────────────────
# Load comparison data (baseline vs human-seeded, if available)
# ─────────────────────────────────────────────────────────────────────

def _load_comparison() -> dict:
    if not COMPARISON_PATH.exists():
        return {}
    try:
        data = json.loads(COMPARISON_PATH.read_text())
        return {d["game"]: d for d in data}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────
# Build brief
# ─────────────────────────────────────────────────────────────────────

def build_brief(game_ids: list[str] | None = None) -> dict:
    summaries = aggregate_all(game_ids)
    comparison = _load_comparison()

    games_out = []
    for s in summaries:
        game = s["game"]
        entry = {**s}   # all aggregated stats

        # Attach benchmark comparison if available
        if game in comparison:
            c = comparison[game]
            entry["benchmark"] = {
                "baseline_mean_reward": c.get("baseline", {}).get("mean"),
                "human_mean_reward":    c.get("human",    {}).get("mean"),
                "n_seeded":             c.get("human",    {}).get("n_seeded", 0),
            }

        # Hypothesis request — what Tetra should focus on
        entry["analysis_hints"] = _analysis_hints(s, comparison.get(game))
        games_out.append(entry)

    brief = {
        "schema_version": "atari_v1",
        "n_games": len(games_out),
        "games": games_out,
        "inbox_schema": _INBOX_SCHEMA,
        "instructions": _INSTRUCTIONS,
    }
    return brief


def _analysis_hints(s: dict, comp: dict | None) -> list[str]:
    """Generate specific questions for Tetra based on observed data."""
    hints = []

    if s["n_steps"] == 0:
        hints.append("No event data collected yet — play sessions needed.")
        return hints

    if s["alignment_rate"] < 0.5:
        hints.append(
            f"Low alignment ({s['alignment_rate']:.0%}) — agent and human disagree "
            f"more than half the time. What strategy does the human have that the "
            f"agent is missing?"
        )
    elif s["alignment_rate"] > 0.85:
        hints.append(
            f"High alignment ({s['alignment_rate']:.0%}) — agent closely mirrors "
            f"human. Investigate whether the agent's Q-values reflect human "
            f"intent or just happen to correlate."
        )

    if s["high_conf_disagree_rate"] > 0.1:
        hints.append(
            f"High-confidence disagreement rate = {s['high_conf_disagree_rate']:.0%}. "
            f"Agent is SURE about an action the human avoids. These are the most "
            f"valuable disagreements — the agent has learned a wrong confident habit."
        )

    if s["disagree_near_terminal_rate"] > 0.3:
        hints.append(
            f"Disagreement near terminal/death = {s['disagree_near_terminal_rate']:.0%}. "
            f"Agent diverges from human exactly when survival matters most. "
            f"Suggest: elevate near_death transitions in priority buffer."
        )

    if s["mean_entropy"] > 2.0:
        hints.append(
            f"High mean entropy ({s['mean_entropy']:.2f}) — agent is uncertain "
            f"across the board. Imitation training may need more steps, or "
            f"the action space is too large for current training data volume."
        )

    top_dis = s.get("top_disagreement_actions", {})
    if top_dis:
        top_action = next(iter(top_dis))
        hints.append(
            f"Most disagreed-on human action: '{top_action}' "
            f"({top_dis[top_action]} times). Why does the agent avoid this action?"
        )

    if comp:
        bl = (comp.get("baseline") or {}).get("mean", 0) or 0
        hu = (comp.get("human") or {}).get("mean", 0) or 0
        if bl == 0 and hu == 0:
            pass
        elif bl == 0:
            hints.append(
                f"Baseline scored 0 — pure RL never escaped zero reward. "
                f"Human-seeded arm scored {hu:.1f}. The human seed was essential."
            )
        elif hu > bl * 1.2:
            delta = (hu - bl) / abs(bl) * 100
            hints.append(
                f"Human seeding improved reward by +{delta:.0f}% ({bl:.1f} -> {hu:.1f}). "
                f"Which actions correlate with the reward improvement?"
            )
        elif hu < bl * 0.9:
            hints.append(
                f"Human seeding HURT performance ({bl:.1f} -> {hu:.1f}). "
                f"Possible negative transfer — human strategy may conflict with RL objective."
            )

    return hints


# ─────────────────────────────────────────────────────────────────────
# Tetra inbox schema + instructions
# ─────────────────────────────────────────────────────────────────────

_INBOX_SCHEMA = {
    "description": "Respond with a JSON array of hypothesis operations.",
    "operations": ["ADD", "RETIRE", "MUTATE"],
    "required_fields": {
        "ADD":    ["op", "name", "description", "game", "llm_score",
                   "llm_priority", "llm_notes", "enaction"],
        "RETIRE": ["op", "name", "llm_notes"],
        "MUTATE": ["op", "parent", "name", "description", "game",
                   "llm_score", "llm_priority", "llm_notes", "enaction"],
    },
    "enaction_types": {
        "priority_boost": {
            "description": "Boost replay priority for transitions matching condition",
            "fields": {"condition": "str (e.g. 'near_death')",
                       "multiplier": "float"}
        },
        "imitation_weight": {
            "description": "Scale imitation loss contribution for specific action",
            "fields": {"action": "str (action name)", "alpha": "float 0-1"}
        },
        "explore_suppress": {
            "description": "Reduce epsilon for specific human actions the agent avoids",
            "fields": {"action": "str", "epsilon_scale": "float 0-1"}
        },
    },
}

_INSTRUCTIONS = (
    "You are Tetra analyzing Atari human-play vs agent alignment data. "
    "For each game, you receive: alignment_rate, high_conf_disagree_rate, "
    "disagree_near_terminal_rate, mean_entropy, calibration_proxy, "
    "reward_on_agree/disagree, top human actions, top disagreement actions, "
    "and benchmark comparison (baseline RL vs human-seeded RL). "
    "Write 2-4 focused hypothesis operations per game. "
    "Reference specific numbers. Output ONLY a valid JSON array."
)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Generate Tetra brief from Atari event logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", nargs="*", default=None,
                   help="Specific game IDs (default: all with event data)")
    p.add_argument("--out", default=str(BRIEF_PATH),
                   help="Output path for atari_brief.json")
    p.add_argument("--print", action="store_true", dest="print_brief",
                   help="Also print summary table to stdout")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    brief = build_brief(args.games)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(brief, indent=2))
    print(f"Brief written -> {out_path}  ({brief['n_games']} games)")

    if args.print_brief:
        print(f"\n{'='*70}")
        print(f"  {'Game':<30} {'Align':>7} {'HCDis':>7} {'Entropy':>8} {'Calib':>7}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")
        for g in brief["games"]:
            if g.get("n_steps", 0) == 0:
                continue
            name = g["game"].replace("ALE/", "")[:30]
            print(f"  {name:<30} "
                  f"{g['alignment_rate']:>7.2%} "
                  f"{g['high_conf_disagree_rate']:>7.2%} "
                  f"{g['mean_entropy']:>8.3f} "
                  f"{g['calibration_proxy']:>7.2%}")
