"""
compare_benchmarks.py
======================
Side-by-side comparison of baseline vs human-seeded benchmark results.

Usage
-----
    python compare_benchmarks.py

Reads all baseline_<game>.json and human_<game>.json files from
benchmark_results/ and prints a formatted table plus saves
benchmark_results/comparison.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = _ROOT / "benchmark_results"


# ─────────────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────────────

def load_results() -> dict[str, dict]:
    """Return {game_id: {baseline: {...}, human: {...}}} for all matched games."""
    baselines = {
        p.stem.replace("baseline_", "").replace("_v5", "-v5").replace("_", "/", 1): p
        for p in RESULTS_DIR.glob("baseline_*.json")
    }
    humans = {
        p.stem.replace("human_", "").replace("_v5", "-v5").replace("_", "/", 1): p
        for p in RESULTS_DIR.glob("human_*.json")
    }

    matched: dict[str, dict] = {}

    # Match by reading the actual game field from each file
    all_baseline = {}
    for p in RESULTS_DIR.glob("baseline_*.json"):
        try:
            d = json.loads(p.read_text())
            all_baseline[d["game"]] = d
        except Exception:
            pass

    all_human = {}
    for p in RESULTS_DIR.glob("human_*.json"):
        try:
            d = json.loads(p.read_text())
            all_human[d["game"]] = d
        except Exception:
            pass

    all_games = set(all_baseline) | set(all_human)
    for game in sorted(all_games):
        matched[game] = {
            "baseline": all_baseline.get(game),
            "human":    all_human.get(game),
        }
    return matched


# ─────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────

def _get(d: dict | None, key: str, default=float("nan")):
    if d is None:
        return default
    return d.get(key, default)


def delta_pct(base: float, human: float) -> str:
    if base == 0 or np.isnan(base) or np.isnan(human):
        return "    N/A"
    d = (human - base) / abs(base) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:5.1f}%"


def significance(base_eps: list, human_eps: list) -> str:
    """Rough t-test p-value label."""
    if not base_eps or not human_eps:
        return ""
    try:
        from scipy import stats as _stats
        _, p = _stats.ttest_ind(human_eps, base_eps)
        if p < 0.001:
            return "***"
        if p < 0.01:
            return " **"
        if p < 0.05:
            return "  *"
        return "   "
    except ImportError:
        # No scipy — skip
        return "   "


# ─────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────

def print_table(matched: dict[str, dict]) -> list[dict]:
    rows = []

    # Header
    print()
    print("=" * 90)
    print("  BENCHMARK COMPARISON: Baseline (pure RL) vs Human-Seeded (Option A)")
    print("=" * 90)
    print(f"  {'Game':<28} {'Base Mean':>9} {'Human Mean':>10} "
          f"{'Delta':>8} {'Sig':>4}  {'Seeded':>7}  "
          f"{'Base Max':>8} {'Human Max':>9}")
    print(f"  {'-'*28} {'-'*9} {'-'*10} {'-'*8} {'-'*4}  {'-'*7}  {'-'*8} {'-'*9}")

    for game, pair in sorted(matched.items()):
        bl = pair["baseline"]
        hu = pair["human"]

        bl_mean = _get(bl, "mean_reward")
        hu_mean = _get(hu, "mean_reward")
        bl_max  = _get(bl, "max_reward")
        hu_max  = _get(hu, "max_reward")
        seeded  = _get(hu, "n_seeded", 0)

        bl_eps  = [e["reward"] for e in (bl or {}).get("episodes", [])]
        hu_eps  = [e["reward"] for e in (hu or {}).get("episodes", [])]

        dpct = delta_pct(bl_mean, hu_mean)
        sig  = significance(bl_eps, hu_eps)

        # Highlight positive delta
        marker = "▲" if not np.isnan(hu_mean) and not np.isnan(bl_mean) and hu_mean > bl_mean else " "

        name = game.replace("ALE/", "")[:28]
        print(f"  {name:<28} {bl_mean:>9.2f} {hu_mean:>10.2f} "
              f"{dpct:>8} {sig:>4}{marker} {seeded:>7}  "
              f"{bl_max:>8.2f} {hu_max:>9.2f}")

        rows.append({
            "game": game,
            "baseline_mean": bl_mean,
            "human_mean": hu_mean,
            "delta_pct": dpct.strip(),
            "n_seeded": seeded,
            "baseline_max": bl_max,
            "human_max": hu_max,
        })

    print("=" * 90)
    print("  ▲ = human-seeded outperformed baseline   * p<0.05  ** p<0.01  *** p<0.001")
    print()

    # Summary counts
    complete = [(r["game"], r["baseline_mean"], r["human_mean"])
                for r in rows
                if not np.isnan(r["baseline_mean"]) and not np.isnan(r["human_mean"])]
    if complete:
        wins   = sum(1 for _, b, h in complete if h > b)
        losses = sum(1 for _, b, h in complete if h < b)
        ties   = len(complete) - wins - losses
        print(f"  Human-seeded won {wins}/{len(complete)} games, "
              f"lost {losses}, tied {ties}")
        if complete:
            avg_delta = np.mean([(h - b) / max(abs(b), 1)
                                  for _, b, h in complete]) * 100
            print(f"  Average delta: {avg_delta:+.1f}%")

    return rows


# ─────────────────────────────────────────────────────────────────────
# Per-game learning curve (optional, saved as compact JSON)
# ─────────────────────────────────────────────────────────────────────

def save_comparison(matched: dict[str, dict], rows: list[dict]) -> None:
    out = []
    for game, pair in matched.items():
        bl = pair["baseline"]
        hu = pair["human"]
        out.append({
            "game": game,
            "baseline": {
                "per_ep_reward": [e["reward"] for e in (bl or {}).get("episodes", [])],
                "mean": _get(bl, "mean_reward"),
                "std":  _get(bl, "std_reward"),
                "max":  _get(bl, "max_reward"),
            },
            "human": {
                "per_ep_reward": [e["reward"] for e in (hu or {}).get("episodes", [])],
                "mean": _get(hu, "mean_reward"),
                "std":  _get(hu, "std_reward"),
                "max":  _get(hu, "max_reward"),
                "n_seeded": _get(hu, "n_seeded", 0),
            },
        })
    p = RESULTS_DIR / "comparison.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"  Full comparison data → {p}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not RESULTS_DIR.exists():
        print(f"No benchmark_results/ directory found. "
              f"Run benchmark_baseline.py and benchmark_human.py first.")
        sys.exit(1)

    matched = load_results()
    if not matched:
        print("No results found. Run the benchmarks first.")
        sys.exit(1)

    rows = print_table(matched)
    save_comparison(matched, rows)
