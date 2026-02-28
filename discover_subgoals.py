"""
discover_subgoals.py
====================
Autonomous RAM-based subgoal discovery from human play data.

Reads either:
  A) reward_ram_log JSONL (always available) — gives RAM state at reward events
  B) atari_events JSONL with "ram" field (opt-in via ram_obs= in log_step)

Algorithm
---------
For each reward event in the data:
  1. Find the RAM state just before the reward ('ram_before')
  2. Compare against the "baseline" RAM distribution across non-reward steps
  3. For each byte b:
       score(b, v) = P(ram[b] == v | within N steps of reward) /
                     P(ram[b] == v | baseline)
  4. Top-K (byte, value) pairs with score > threshold become subgoal candidates

Output: experiments/subgoal_candidates.json
  [{
    "byte": 56,
    "value": 255,
    "score": 12.4,
    "p_near_reward": 0.94,
    "p_baseline": 0.076,
    "n_reward_events": 3,
    "label": "estimated",          # "calibrated" if byte is known
    "x_at_reward": [192, 201],     # RAM[42] values at reward (player_x)
    "y_at_reward": [14, 14],       # RAM[43] values at reward (player_y)
    "room_at_reward": [1],         # RAM[3]  values at reward (room)
    "suggested_subgoal": {
      "name": "byte_56_eq_255",
      "description": "RAM[56] == 255  (score=12.4, p=0.94)",
      "reward": 5.0,               # placeholder, tune manually
      "spatial": {"x_range": [185, 210], "y_range": [10, 18], "room": 1}
    }
  }, ...]

Usage
-----
  # From reward_ram_log only (always available):
  python discover_subgoals.py --game ALE/MontezumaRevenge-v5

  # Include full-episode JSONL files with RAM field (opt-in logging):
  python discover_subgoals.py --game ALE/MontezumaRevenge-v5 --use-episode-ram

  # Output to a different path:
  python discover_subgoals.py --game ALE/MontezumaRevenge-v5 --out experiments/my_candidates.json

  # Lower threshold (more candidates):
  python discover_subgoals.py --game ALE/MontezumaRevenge-v5 --min-score 3.0

After running, inspect experiments/subgoal_candidates.json and paste the
relevant candidates into experiments/atari_inbox.json to feed Tetra.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT        = Path(__file__).resolve().parent
_EVENTS_DIR  = _ROOT / "experiments" / "atari_events"
_REWARD_LOG  = _ROOT / "experiments" / "save_states" / "reward_ram_log"
_OUT_DEFAULT = _ROOT / "experiments" / "subgoal_candidates.json"

# Known bytes (from calibration) — used to label candidates
_KNOWN_BYTES: dict[int, str] = {
    3:  "room",
    42: "player_x",
    43: "player_y",
    56: "key_flag_primary",
    58: "lives",
    65: "key_flag_secondary",
}


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────

def _game_slug(game_id: str) -> str:
    return game_id.replace("/", "_").replace("-", "_")


def _load_reward_ram_log(game_id: str) -> list[dict]:
    """
    Load all reward_ram_log JSONL entries for this game.
    Each entry: {step, reward, action, ram_before: [128 ints], ram_after: [128 ints]}
    """
    slug   = _game_slug(game_id)
    log_dir = _REWARD_LOG / slug
    if not log_dir.exists():
        return []
    events = []
    for f in sorted(log_dir.glob("*_rewards.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


def _load_episode_ram_events(game_id: str) -> list[dict]:
    """
    Load JSONL events that include a 'ram' field.
    Only returns events that have the ram field (opt-in logging).
    """
    slug    = _game_slug(game_id)
    ev_dir  = _EVENTS_DIR / slug
    if not ev_dir.exists():
        return []
    events = []
    for f in sorted(ev_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "ram" in obj:
                        events.append(obj)
                except json.JSONDecodeError:
                    pass
    return events


# ─────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────

def _analyse_reward_log(
    reward_events: list[dict],
    min_score: float = 5.0,
    top_k: int = 20,
) -> list[dict]:
    """
    Analyse ram_before/ram_after at reward moments.

    For each byte that transitions at reward time, compute a lift score:
      lift = p(byte transitions to value V at reward) / p(any transition to V)

    Since we only have the reward-moment snapshot (no baseline distribution
    from full episodes), we use a simpler heuristic:
      - bytes that change at reward time are candidates
      - we rank by |ram_after[b] - ram_before[b]| to prefer large/discrete jumps
      - if after_value is 0 or 255 (flag pattern), score x2
    """
    if not reward_events:
        return []

    # Collect changed bytes across all reward events
    byte_changes: dict[int, list[dict]] = defaultdict(list)
    for ev in reward_events:
        before = ev.get("ram_before", [])
        after  = ev.get("ram_after",  [])
        if len(before) < 128 or len(after) < 128:
            continue
        reward = ev.get("reward", 0.0)
        for b in range(128):
            vb, va = before[b], after[b]
            if vb != va:
                byte_changes[b].append({
                    "before": vb,
                    "after":  va,
                    "reward": reward,
                    "x": before[42],
                    "y": before[43],
                    "room": before[3],
                })

    n_events = len(reward_events)
    candidates = []

    for byte_idx, changes in byte_changes.items():
        n_changed = len(changes)
        freq      = n_changed / n_events   # how often this byte changes at reward

        # Group by (before, after) transition pairs
        transitions: dict[tuple, list] = defaultdict(list)
        for c in changes:
            transitions[(c["before"], c["after"])].append(c)

        for (vb, va), instances in transitions.items():
            n_inst = len(instances)
            # Score: frequency of this exact transition × flag bonus
            flag_bonus = 2.0 if va in (0, 255, 1, 2) else 1.0
            size_bonus = 1.0 + abs(va - vb) / 128.0   # larger change = more signal
            score      = (n_inst / n_events) * flag_bonus * size_bonus * 10.0

            if score < min_score * 0.5:   # pre-filter
                continue

            xs    = [i["x"]    for i in instances]
            ys    = [i["y"]    for i in instances]
            rooms = [i["room"] for i in instances]

            label = _KNOWN_BYTES.get(byte_idx, "unknown")

            # Spatial summary for subgoal threshold suggestion
            x_lo, x_hi = min(xs), max(xs)
            y_lo, y_hi = min(ys), max(ys)
            pad = 15
            spatial = {
                "x_range": [max(0, x_lo - pad), min(159, x_hi + pad)],
                "y_range": [max(0, y_lo - pad), min(255, y_hi + pad)],
                "room":    sorted(set(rooms)),
            }

            candidates.append({
                "byte":            byte_idx,
                "value_before":    vb,
                "value_after":     va,
                "score":           round(score, 3),
                "n_reward_events": n_inst,
                "n_total_events":  n_events,
                "freq":            round(n_inst / n_events, 4),
                "label":           label,
                "x_at_reward":     xs,
                "y_at_reward":     ys,
                "room_at_reward":  sorted(set(rooms)),
                "suggested_subgoal": {
                    "name":        f"byte_{byte_idx}_transition_{vb}_to_{va}",
                    "description": (
                        f"RAM[{byte_idx}] {vb}→{va}  "
                        f"({label}, score={score:.1f}, "
                        f"freq={n_inst}/{n_events} reward events)"
                    ),
                    "reward":      round(min(score, 15.0), 1),
                    "spatial":     spatial,
                },
            })

    # Sort by score descending, deduplicate by byte keeping best
    candidates.sort(key=lambda c: c["score"], reverse=True)
    seen_bytes: set[int] = set()
    deduped = []
    for c in candidates:
        if c["byte"] not in seen_bytes:
            seen_bytes.add(c["byte"])
            deduped.append(c)

    return deduped[:top_k]


def _analyse_episode_ram(
    episode_events: list[dict],
    min_score: float = 5.0,
    lookback: int = 50,
    top_k: int = 20,
) -> list[dict]:
    """
    Analyse full-episode JSONL where each event includes a 'ram' field.

    For each step with reward > 0:
      - collect RAM states for the lookback window before the reward
      - compare byte value distributions in that window vs. the whole episode

    lift(b, v) = P(ram[b]==v | within lookback of reward) /
                 P(ram[b]==v | anywhere in episode)
    """
    if not episode_events:
        return []

    # Index events by (episode, step)
    by_ep: dict[int, list] = defaultdict(list)
    for ev in episode_events:
        by_ep[ev["episode"]].append(ev)

    # byte_value → {near_reward_count, total_count}
    byte_stats: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"near": 0, "total": 0, "x": [], "y": [], "rooms": []}
    )

    for ep, evs in by_ep.items():
        evs_sorted = sorted(evs, key=lambda e: e["step"])
        n = len(evs_sorted)

        # Total counts — baseline
        for ev in evs_sorted:
            ram = ev["ram"]
            for b in range(128):
                byte_stats[(b, ram[b])]["total"] += 1

        # Near-reward counts
        reward_steps = {ev["step"] for ev in evs_sorted if ev.get("reward", 0.0) > 0}
        near_reward_steps = set()
        for rs in reward_steps:
            for offset in range(lookback):
                near_reward_steps.add(rs - offset)

        for ev in evs_sorted:
            if ev["step"] in near_reward_steps:
                ram = ev["ram"]
                for b in range(128):
                    key = (b, ram[b])
                    byte_stats[key]["near"] += 1
                    byte_stats[key]["x"].append(ram[42])
                    byte_stats[key]["y"].append(ram[43])
                    byte_stats[key]["rooms"].append(ram[3])

    n_total = sum(len(evs) for evs in by_ep.values())
    n_near  = max(1, sum(
        len([e for e in evs if e.get("reward", 0.0) > 0]) * lookback
        for evs in by_ep.values()
    ))

    candidates = []
    for (byte_idx, value), stats in byte_stats.items():
        if stats["near"] < 2:
            continue
        p_near     = stats["near"] / n_near
        p_baseline = max(stats["total"] / n_total, 1e-9)
        lift       = p_near / p_baseline

        if lift < min_score:
            continue

        xs    = stats["x"][:50]
        ys    = stats["y"][:50]
        rooms = stats["rooms"][:50]
        label = _KNOWN_BYTES.get(byte_idx, "unknown")

        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)
        pad = 15
        spatial = {
            "x_range": [max(0, x_lo - pad), min(159, x_hi + pad)],
            "y_range": [max(0, y_lo - pad), min(255, y_hi + pad)],
            "room":    sorted(set(rooms)),
        }

        candidates.append({
            "byte":          byte_idx,
            "value":         value,
            "lift":          round(lift, 3),
            "p_near_reward": round(p_near,     4),
            "p_baseline":    round(p_baseline, 4),
            "n_near":        stats["near"],
            "label":         label,
            "x_sample":      xs[:10],
            "y_sample":      ys[:10],
            "room_sample":   sorted(set(rooms))[:5],
            "suggested_subgoal": {
                "name":        f"byte_{byte_idx}_eq_{value}",
                "description": (
                    f"RAM[{byte_idx}]=={value}  "
                    f"({label}, lift={lift:.1f}x, p_near={p_near:.3f})"
                ),
                "reward":  round(min(math.log2(lift + 1) * 2, 15.0), 1),
                "spatial": spatial,
            },
        })

    candidates.sort(key=lambda c: c["lift"], reverse=True)
    seen_bytes: set[int] = set()
    deduped = []
    for c in candidates:
        if c["byte"] not in seen_bytes:
            seen_bytes.add(c["byte"])
            deduped.append(c)

    return deduped[:top_k]


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def discover(
    game_id: str,
    use_episode_ram: bool = False,
    min_score: float = 5.0,
    top_k: int = 20,
    out_path: Path = _OUT_DEFAULT,
) -> list[dict]:
    print(f"[discover_subgoals] game={game_id}")

    # Source A: reward_ram_log
    reward_events = _load_reward_ram_log(game_id)
    print(f"  Reward log events: {len(reward_events)}")
    candidates_a = _analyse_reward_log(reward_events, min_score=min_score, top_k=top_k)

    # Source B: full-episode JSONL with RAM (opt-in)
    candidates_b: list[dict] = []
    if use_episode_ram:
        ep_events = _load_episode_ram_events(game_id)
        print(f"  Episode RAM events: {len(ep_events)}")
        if ep_events:
            candidates_b = _analyse_episode_ram(ep_events, min_score=min_score, top_k=top_k)

    # Merge: prefer reward-log candidates (more precise), append novel ep candidates
    seen_bytes = {c["byte"] for c in candidates_a}
    merged = list(candidates_a)
    for c in candidates_b:
        if c["byte"] not in seen_bytes:
            merged.append(c)
            seen_bytes.add(c["byte"])

    output = {
        "game":             game_id,
        "n_reward_events":  len(reward_events),
        "n_episode_events": len(_load_episode_ram_events(game_id)) if use_episode_ram else 0,
        "method":           "reward_log" + ("+episode_ram" if use_episode_ram else ""),
        "n_candidates":     len(merged),
        "candidates":       merged,
        "next_steps": [
            "1. Review candidates — focus on high score/lift with known labels",
            "2. Copy promising ones to experiments/atari_inbox.json as ADD ops",
            "3. Set op.enaction.type = 'shaped_reward_ram' with byte/value/bonus",
            "4. Run: python ingest_atari_inbox.py",
            "5. Update montezuma_subgoals.py with confirmed coordinates",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\n  Found {len(merged)} candidate subgoal signals:")
    for c in merged[:10]:
        label = c.get("label", "?")
        if "lift" in c:
            print(f"    byte={c['byte']:3d} ({label:25s}) value={c['value']:3d}  lift={c['lift']:.1f}x")
        else:
            score = c.get("score", 0)
            vb, va = c.get("value_before",0), c.get("value_after",0)
            print(f"    byte={c['byte']:3d} ({label:25s}) {vb}→{va}  score={score:.1f}  freq={c['freq']:.2f}")

    print(f"\n  Written: {out_path}")
    print()
    print("  Next: review candidates.json, paste into atari_inbox.json as ADD ops")
    return merged


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous RAM subgoal discovery from human play data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game", default="ALE/MontezumaRevenge-v5",
                   help="Game ID (e.g. ALE/MontezumaRevenge-v5)")
    p.add_argument("--use-episode-ram", action="store_true",
                   help="Also analyse full-episode JSONL files with 'ram' field "
                        "(requires opt-in logging via ram_obs= in log_step)")
    p.add_argument("--min-score", type=float, default=5.0,
                   help="Minimum score/lift to include a candidate")
    p.add_argument("--top-k", type=int, default=20,
                   help="Max candidates to output")
    p.add_argument("--out", type=Path, default=_OUT_DEFAULT,
                   help="Output JSON path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    discover(
        game_id         = args.game,
        use_episode_ram = args.use_episode_ram,
        min_score       = args.min_score,
        top_k           = args.top_k,
        out_path        = args.out,
    )
