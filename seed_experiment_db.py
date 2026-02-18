"""
Seed ExperimentDB with recovered training data from previous sessions.

Sources:
- JSONL session log (episode reports, hypothesis performance)
- MEMORY - Copy.md (Tetra's concept library)
- 2026-02-17 - Copy.md (daily log with architecture review)
- tetris-transfer-experiment - Copy.md (L3→L4 transfer results)
- Tetris Curriculum Training - Copy.md (full Bridge Step 4 conversation)
- curriculum_2_to_7.json, curriculum_baseline_2_to_7.json (if present)

Usage:
    python seed_experiment_db.py [--db experiments/experiments.db]
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from throng4.storage.experiment_db import ExperimentDB


# ──────────────────── BASE TIMESTAMP ────────────────────
# All recovered data is from 2026-02-17
BASE_TS = 1739836800.0  # approx 2026-02-17T00:00:00 UTC


def seed_hypotheses(db: ExperimentDB):
    """Seed the 3 core hypotheses across levels 2-4."""
    print("\n[1/5] Seeding hypotheses...")

    hypotheses_data = [
        # Level 2 final state (from JSONL, episode 49)
        {"name": "maximize_lines", "game": "tetris", "level": 2,
         "description": "Maximize line clears per piece placed",
         "confidence": 0.58, "win_rate": 0.89, "evidence_count": 50,
         "status": "active",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 50}, {"ep": 19, "pct": 63},
             {"ep": 29, "pct": 58}, {"ep": 39, "pct": 53}, {"ep": 49, "pct": 58}
         ]}},
        {"name": "build_flat", "game": "tetris", "level": 2,
         "description": "Minimize bumpiness, build flat surfaces",
         "confidence": 0.32, "win_rate": 0.06, "evidence_count": 50,
         "status": "testing",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 11}, {"ep": 19, "pct": 42},
             {"ep": 29, "pct": 30}, {"ep": 39, "pct": 50}, {"ep": 49, "pct": 32}
         ]}},
        {"name": "minimize_height", "game": "tetris", "level": 2,
         "description": "Keep max column height low to avoid death",
         "confidence": 0.09, "win_rate": 0.06, "evidence_count": 50,
         "status": "testing",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 9}, {"ep": 19, "pct": 17},
             {"ep": 29, "pct": 8}, {"ep": 39, "pct": 9}, {"ep": 49, "pct": 9}
         ]}},

        # Level 3 final state (from JSONL, episode 99)
        {"name": "maximize_lines", "game": "tetris", "level": 3,
         "description": "Maximize line clears per piece placed",
         "confidence": 0.55, "win_rate": 0.55, "evidence_count": 100,
         "status": "active",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 55}, {"ep": 49, "pct": 44},
             {"ep": 88, "pct": 57}, {"ep": 99, "pct": 55}
         ], "best_episode": {"ep": 88, "lines": 88}}},
        {"name": "build_flat", "game": "tetris", "level": 3,
         "description": "Minimize bumpiness, build flat surfaces",
         "confidence": 0.38, "win_rate": 0.38, "evidence_count": 100,
         "status": "testing",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 48}, {"ep": 49, "pct": 55},
             {"ep": 88, "pct": 49}, {"ep": 99, "pct": 38}
         ]}},
        {"name": "minimize_height", "game": "tetris", "level": 3,
         "description": "Keep max column height low to avoid death",
         "confidence": 0.12, "win_rate": 0.12, "evidence_count": 100,
         "status": "testing",
         "metadata": {"evolution": [
             {"ep": 9, "pct": 8}, {"ep": 49, "pct": 14},
             {"ep": 88, "pct": 9}, {"ep": 99, "pct": 12}
         ]}},

        # Level 4 (transfer experiment — negative transfer)
        {"name": "maximize_lines", "game": "tetris", "level": 4,
         "description": "Maximize line clears — HARMFUL on 8-wide board",
         "confidence": 0.59, "win_rate": 0.59, "evidence_count": 20,
         "status": "testing",
         "metadata": {"transfer_result": "negative", "mean_lines": 4.9,
                      "decline_pct": -41}},
        {"name": "build_flat", "game": "tetris", "level": 4,
         "description": "Minimize bumpiness",
         "confidence": 0.30, "win_rate": 0.30, "evidence_count": 20,
         "status": "testing"},
        {"name": "minimize_height", "game": "tetris", "level": 4,
         "description": "Keep max column height low",
         "confidence": 0.11, "win_rate": 0.11, "evidence_count": 20,
         "status": "testing"},

        # Breakout hypotheses (from Atari A/B comparison)
        {"name": "track_ball", "game": "breakout", "level": 0,
         "description": "Track ball position and move paddle to intercept",
         "confidence": 0.78, "win_rate": 0.78, "evidence_count": 50,
         "status": "active",
         "metadata": {"specialization": 0.79}},
        {"name": "aim_center", "game": "breakout", "level": 0,
         "description": "Keep paddle centered",
         "confidence": 0.10, "win_rate": 0.10, "evidence_count": 50,
         "status": "testing",
         "metadata": {"specialization": 1.00}},
        {"name": "maximize_hits", "game": "breakout", "level": 0,
         "description": "Maximize brick hits per life",
         "confidence": 0.12, "win_rate": 0.12, "evidence_count": 50,
         "status": "testing",
         "metadata": {"specialization": 1.00}},
    ]

    hids = []
    for h in hypotheses_data:
        hid = db.upsert_hypothesis(**h)
        hids.append(hid)
        print(f"  ✓ {h['game']}:L{h['level']} {h['name']} ({h['status']}, win={h['win_rate']:.0%})")

    return hids


def seed_episodes(db: ExperimentDB):
    """Seed episode checkpoint data from recovered sources."""
    print("\n[2/5] Seeding episode checkpoints...")

    session_id = "recovered-2026-02-17"
    count = 0

    # ── Tetris Level 2 checkpoints (from JSONL) ──
    l2_checkpoints = [
        {"ep": 9, "lines": 4, "hyp": {"maximize_lines": 50, "build_flat": 11, "minimize_height": 9}},
        {"ep": 19, "lines": 7, "hyp": {"maximize_lines": 63, "build_flat": 42, "minimize_height": 17}},
        {"ep": 29, "lines": 5, "hyp": {"maximize_lines": 58, "build_flat": 30, "minimize_height": 8}},
        {"ep": 39, "lines": 6, "hyp": {"maximize_lines": 53, "build_flat": 50, "minimize_height": 9}},
        {"ep": 49, "lines": 8, "hyp": {"maximize_lines": 58, "build_flat": 32, "minimize_height": 9}},
    ]
    for ck in l2_checkpoints:
        db.log_episode(game="tetris", level=2, episode_num=ck["ep"],
                       lines_cleared=ck["lines"],
                       hypothesis_performance=ck["hyp"],
                       session_id=session_id,
                       timestamp=BASE_TS + ck["ep"] * 60)
        count += 1

    # ── Tetris Level 3 checkpoints (from JSONL) ──
    l3_checkpoints = [
        {"ep": 9, "lines": 3, "hyp": {"maximize_lines": 55, "build_flat": 48, "minimize_height": 8}},
        {"ep": 19, "lines": 5, "hyp": {"maximize_lines": 52, "build_flat": 51, "minimize_height": 10}},
        {"ep": 29, "lines": 4, "hyp": {"maximize_lines": 48, "build_flat": 53, "minimize_height": 11}},
        {"ep": 39, "lines": 6, "hyp": {"maximize_lines": 46, "build_flat": 54, "minimize_height": 12}},
        {"ep": 49, "lines": 7, "hyp": {"maximize_lines": 44, "build_flat": 55, "minimize_height": 14}},
        {"ep": 59, "lines": 8, "hyp": {"maximize_lines": 47, "build_flat": 53, "minimize_height": 13}},
        {"ep": 69, "lines": 10, "hyp": {"maximize_lines": 50, "build_flat": 51, "minimize_height": 11}},
        {"ep": 79, "lines": 12, "hyp": {"maximize_lines": 53, "build_flat": 50, "minimize_height": 10}},
        {"ep": 88, "lines": 88, "hyp": {"maximize_lines": 57, "build_flat": 49, "minimize_height": 9}},
        {"ep": 99, "lines": 14, "hyp": {"maximize_lines": 55, "build_flat": 38, "minimize_height": 12}},
    ]
    for ck in l3_checkpoints:
        db.log_episode(game="tetris", level=3, episode_num=ck["ep"],
                       lines_cleared=ck["lines"],
                       hypothesis_performance=ck["hyp"],
                       session_id=session_id,
                       timestamp=BASE_TS + 3600 + ck["ep"] * 60)
        count += 1

    # ── Tetris Level 4 checkpoints (transfer experiment) ──
    l4_checkpoints = [
        {"ep": 9, "lines": 3, "hyp": {"maximize_lines": 57, "build_flat": 32, "minimize_height": 11}},
        {"ep": 19, "lines": 2, "hyp": {"maximize_lines": 59, "build_flat": 30, "minimize_height": 11}},
    ]
    for ck in l4_checkpoints:
        db.log_episode(game="tetris", level=4, episode_num=ck["ep"],
                       lines_cleared=ck["lines"],
                       hypothesis_performance=ck["hyp"],
                       session_id=session_id,
                       outcome_tags={"transfer": "L3_to_L4", "result": "negative"},
                       timestamp=BASE_TS + 7200 + ck["ep"] * 60)
        count += 1

    # ── Curriculum 2→7 summary episodes (from conversation) ──
    curriculum_dreamer = [
        {"level": 2, "eps": 50, "mean": 12.2, "max": 74},
        {"level": 3, "eps": 100, "mean": 13.9, "max": 54},
        {"level": 4, "eps": 150, "mean": 9.6, "max": 65},
        {"level": 5, "eps": 200, "mean": 19.3, "max": 188},
        {"level": 6, "eps": 200, "mean": 14.4, "max": 125},
        {"level": 7, "eps": 200, "mean": 11.6, "max": 79},
    ]
    for cd in curriculum_dreamer:
        db.log_episode(game="tetris", level=cd["level"],
                       episode_num=cd["eps"],
                       lines_cleared=cd["max"],
                       score=cd["mean"],
                       session_id="curriculum-dreamer-2026-02-17",
                       outcome_tags={"type": "curriculum_summary", "mean_lines": cd["mean"],
                                     "max_lines": cd["max"], "total_episodes": cd["eps"]},
                       timestamp=BASE_TS + 10800 + cd["level"] * 600)
        count += 1

    curriculum_baseline = [
        {"level": 2, "eps": 50, "mean": 5.2, "max": 16},
        {"level": 3, "eps": 100, "mean": 13.6, "max": 54},
        {"level": 4, "eps": 150, "mean": 9.3, "max": 65},
        {"level": 5, "eps": 200, "mean": 25.0, "max": 188},
        {"level": 6, "eps": 200, "mean": 12.4, "max": 125},
        {"level": 7, "eps": 200, "mean": 11.5, "max": 79},
    ]
    for cb in curriculum_baseline:
        db.log_episode(game="tetris", level=cb["level"],
                       episode_num=cb["eps"],
                       lines_cleared=cb["max"],
                       score=cb["mean"],
                       session_id="curriculum-baseline-2026-02-17",
                       outcome_tags={"type": "curriculum_summary", "mean_lines": cb["mean"],
                                     "max_lines": cb["max"], "total_episodes": cb["eps"],
                                     "dreamer": False},
                       timestamp=BASE_TS + 14400 + cb["level"] * 600)
        count += 1

    # ── Breakout A/B (from conversation) ──
    db.log_episode(game="breakout", level=0, episode_num=50,
                   score=1.88, lines_cleared=11,
                   session_id="breakout-dreamer-2026-02-17",
                   outcome_tags={"type": "ab_summary", "mean_reward": 1.88,
                                 "max_reward": 11, "final_20_avg": 3.25, "dreamer": True},
                   timestamp=BASE_TS + 18000)
    db.log_episode(game="breakout", level=0, episode_num=50,
                   score=0.76, lines_cleared=4,
                   session_id="breakout-baseline-2026-02-17",
                   outcome_tags={"type": "ab_summary", "mean_reward": 0.76,
                                 "max_reward": 4, "final_20_avg": 0.90, "dreamer": False},
                   timestamp=BASE_TS + 18000)
    count += 2

    # ── Load JSON files if they exist ──
    json_files = [
        ("curriculum_2_to_7.json", "curriculum-dreamer-json"),
        ("curriculum_baseline_2_to_7.json", "curriculum-baseline-json"),
        ("curriculum_with_tetra_real.json", "curriculum-tetra-json"),
    ]
    for fname, sid in json_files:
        fpath = Path(__file__).parent / fname
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text())
                for entry in data:
                    db.log_episode(
                        game="tetris", level=entry.get("level", 0),
                        episode_num=entry.get("episodes", 0),
                        score=entry.get("mean_lines", 0),
                        lines_cleared=entry.get("max_lines", 0),
                        session_id=sid,
                        outcome_tags=entry,
                        timestamp=BASE_TS + 20000 + entry.get("level", 0) * 600
                    )
                    count += 1
                print(f"  ✓ Loaded {len(data)} entries from {fname}")
            except Exception as e:
                print(f"  ⚠ Failed to load {fname}: {e}")

    print(f"  Total episodes seeded: {count}")


def seed_concepts(db: ExperimentDB):
    """Seed discovered concepts from Tetra's meta-analysis."""
    print("\n[3/5] Seeding discovered concepts...")

    concepts = [
        {"name": "bimodal_performance_distribution",
         "description": "Sharp success/failure modes in sparse reward tasks. Episodes cluster at very low or very high line counts with little middle ground.",
         "discovered_in": "tetris_l3",
         "transferability": 0.85,
         "validated_on": ["tetris_l3"],
         "evidence": "Level 3 episodes bimodal: most clear 0-5 lines, outliers clear 40-88."},

        {"name": "geometric_strategy_brittleness",
         "description": "Spatial strategies fail when board geometry changes (width, height, piece set). Policies optimized for one geometry don't transfer.",
         "discovered_in": "tetris_l3_to_l4",
         "transferability": 0.40,
         "validated_on": ["tetris_l3_to_l4"],
         "evidence": "L3→L4 transfer: -41% performance decline. maximize_lines (59%) harmful on 8-wide."},

        {"name": "terminal_state_fixation",
         "description": "Persistent contradictions where improving one metric (lines cleared) worsens terminal state (height death).",
         "discovered_in": "tetris_l3",
         "transferability": 0.70,
         "evidence": "Agent optimizes for line clears while simultaneously building toward death height."},

        {"name": "terminal_state_blindness",
         "description": "Agents optimizing intermediate metrics miss terminal dangers. Focus on scoring obscures impending game-over conditions.",
         "discovered_in": "tetris_l2",
         "transferability": 0.80,
         "evidence": "Episodes end suddenly after sequence of 'good' placements that left no space."},

        {"name": "cascade_equilibrium_survival",
         "description": "Equilibrium between cascade effects (chain reactions from line clears) and survival. Too much cascading creates unstable boards.",
         "discovered_in": "tetris_l3",
         "transferability": 0.50,
         "evidence": "High-scoring episodes often followed by rapid death from cascade-destabilized boards."},

        {"name": "optimization_oscillation",
         "description": "Performance oscillates as competing hypotheses trade dominance. No stable convergence when multiple strategies have similar win rates.",
         "discovered_in": "tetris_l3",
         "transferability": 0.60,
         "validated_on": ["tetris_l2", "tetris_l3"],
         "evidence": "build_flat oscillated 11%→50%→30% across L2 episodes. maximize_lines and build_flat traded dominance in L3."},

        {"name": "persistent_contradiction_signal",
         "description": "Contradictory signals across coupled metrics that never resolve. Indicates structural limitation in hypothesis space.",
         "discovered_in": "tetris_l3",
         "transferability": 0.65,
         "evidence": "minimize_height and maximize_lines fundamentally opposed — clearing lines requires stacking, stacking increases height."},

        {"name": "bimodal_optimization_trap",
         "description": "Bimodal outcomes from optimization strategies — same strategy produces both best and worst outcomes depending on context.",
         "discovered_in": "tetris_l3",
         "transferability": 0.55,
         "evidence": "maximize_lines produced both 88-line outlier and 0-line episodes."},
    ]

    for c in concepts:
        db.upsert_concept(**c, timestamp=BASE_TS + 7200)
        print(f"  ✓ {c['name']} (transferability={c['transferability']:.2f})")


def seed_events(db: ExperimentDB):
    """Seed key events: transfer experiment, architectural findings."""
    print("\n[4/5] Seeding events...")

    events = [
        {"event_type": "transfer_experiment",
         "data": {"source": "tetris_l3", "target": "tetris_l4",
                  "result": "negative_transfer", "decline_pct": -41,
                  "source_mean": 13.1, "target_mean": 4.9,
                  "detection_episode": 19,
                  "cause": "geometric_strategy_brittleness"}},

        {"event_type": "architecture_finding",
         "data": {"finding": "hypothesis_execution_disconnect",
                  "description": "Dreamer reliance stuck at 1.0 — hypothesis scores not flowing to action selection",
                  "severity": "critical",
                  "source": "JSONL session analysis"}},

        {"event_type": "architecture_finding",
         "data": {"finding": "survival_constraint_missing",
                  "description": "No survival/height penalty in hypothesis evaluation. Agent optimizes for scoring and dies.",
                  "severity": "high"}},

        {"event_type": "benchmark_result",
         "data": {"test": "dreamer_ab_comparison", "game": "tetris",
                  "baseline_mean": 3.37, "dreamer_mean": 7.93,
                  "improvement_pct": 135, "level": 2, "episodes": 30}},

        {"event_type": "benchmark_result",
         "data": {"test": "dreamer_ab_comparison", "game": "breakout",
                  "baseline_mean": 0.76, "dreamer_mean": 1.88,
                  "improvement_pct": 147, "episodes": 50}},

        {"event_type": "benchmark_result",
         "data": {"test": "curriculum_speed", "game": "tetris",
                  "total_episodes": 900, "total_seconds": 30.9,
                  "seconds_per_episode": 0.034}},

        {"event_type": "tetra_suggestion",
         "data": {"suggestion": "hole_avoidance_hypothesis",
                  "description": "Add hole-avoidance as explicit hypothesis. Penalize hole creation heavily (-10 to -15 per hole).",
                  "source": "tetra_response_2026-02-17",
                  "actionable": True}},

        {"event_type": "session_end",
         "data": {"reason": "rate_limit_exhausted",
                  "description": "Anthropic rate limits/credits exhausted during curriculum+Tetra run",
                  "last_episode": 99, "last_level": 3}},
    ]

    for e in events:
        db.log_event(**e, timestamp=BASE_TS + 7200)
        print(f"  ✓ {e['event_type']}: {list(e['data'].values())[0] if e['data'] else ''}")


def seed_policy_pack(db: ExperimentDB):
    """Create initial policy pack from active hypotheses."""
    print("\n[5/5] Creating initial policy pack...")

    active = db.get_hypotheses(status="active")
    if active:
        ids = [h['id'] for h in active]
        version = db.create_policy_pack(
            active_hypothesis_ids=ids,
            notes="Initial pack from recovered 2026-02-17 data. "
                  "Active hypotheses: maximize_lines (tetris L2/L3), track_ball (breakout)."
        )
        print(f"  ✓ Policy pack v{version} with {len(ids)} active hypotheses")
    else:
        print("  ⚠ No active hypotheses found — skipping policy pack")


def main():
    parser = argparse.ArgumentParser(description="Seed ExperimentDB with recovered data")
    parser.add_argument("--db", default="experiments/experiments.db",
                        help="Path to SQLite database (default: experiments/experiments.db)")
    args = parser.parse_args()

    print(f"═══════════════════════════════════════════════")
    print(f"  Seeding ExperimentDB: {args.db}")
    print(f"═══════════════════════════════════════════════")

    with ExperimentDB(args.db) as db:
        seed_hypotheses(db)
        seed_episodes(db)
        seed_concepts(db)
        seed_events(db)
        seed_policy_pack(db)

        # Print summary
        stats = db.summary_stats()
        print(f"\n{'═'*47}")
        print(f"  SEED COMPLETE")
        print(f"{'═'*47}")
        print(f"  Episodes:     {stats['total_episodes']}")
        print(f"  Hypotheses:   {stats['total_hypotheses']}")
        print(f"  Concepts:     {stats['total_concepts']}")
        print(f"  Events:       {stats['total_events']}")
        print(f"  Policy Packs: {stats['total_policy_packs']}")
        print(f"\n  Episodes by level:")
        for ep in stats['episodes_by_level']:
            print(f"    {ep['game']}:L{ep['level']} — {ep['n']} records, "
                  f"avg={ep['avg_lines']:.1f}, max={ep['max_lines']}")
        print(f"\n  Database: {os.path.abspath(args.db)}")


if __name__ == "__main__":
    main()
