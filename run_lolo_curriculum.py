"""
run_lolo_curriculum.py — Run Lolo curriculum training.

Launches graduated puzzle training through WholeBrain.
Generates procedural puzzles at increasing difficulty tiers.

Usage:
    python run_lolo_curriculum.py [--episodes 500] [--tier 1] [--verbose]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.games.lolo.lolo_adapter import LoloAdapter
from brain.games.lolo.lolo_curriculum import LoloCurriculum
from brain.games.lolo.lolo_generator import LoloPuzzleGenerator
from brain.orchestrator import WholeBrain


def main():
    parser = argparse.ArgumentParser(description="Lolo Curriculum Training")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per tier")
    parser.add_argument("--tier", type=int, default=1, help="Starting tier (1-7)")
    parser.add_argument("--max-tier", type=int, default=7, help="Max tier to attempt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--session", type=str, default="lolo_train", help="Session name")
    args = parser.parse_args()

    print("=" * 60)
    print("Adventures of Lolo — Curriculum Training")
    print("=" * 60)
    print(f"  Episodes/tier: {args.episodes}")
    print(f"  Starting tier: {args.tier}")
    print(f"  Max tier:      {args.max_tier}")
    print(f"  Session:       {args.session}")
    print()

    # Initialize brain
    brain = WholeBrain(
        n_features=84,
        n_actions=18,
        session_name=args.session,
        game_mode="puzzle",
    )

    # Initialize curriculum
    gen = LoloPuzzleGenerator(seed=args.seed)
    adapter = LoloAdapter(feature_dim=84)
    curriculum = LoloCurriculum(brain, generator=gen, adapter=adapter, seed=args.seed)

    # Set starting tier
    curriculum._current_tier = args.tier
    gen.complexity_tier = args.tier

    # Run curriculum
    start = time.time()
    try:
        results = curriculum.run_full_curriculum(
            episodes_per_tier=args.episodes,
            max_tiers=args.max_tier,
            verbose=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        results = curriculum.report()

    elapsed = time.time() - start

    # Final report
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed:     {elapsed:.1f}s")
    print(f"  Total episodes:   {curriculum._total_episodes}")
    print(f"  Total steps:      {curriculum._total_steps}")
    print(f"  Dead-ends caught: {curriculum._dead_ends_detected}")
    print(f"  Flagged puzzles:  {len(curriculum.flagged_puzzles)}")
    print()

    # Per-tier summary
    for tier, stats in curriculum.tier_stats.items():
        print(f"  Tier {tier}: {stats['success_rate']:.0%} success, "
              f"avg_reward={stats['avg_reward']:.1f}, "
              f"episodes={stats['episodes']}")

    # Brain report
    report = brain.report()
    print()
    print(f"  Brain sections: {len(report)}")
    for section, data in report.items():
        if isinstance(data, dict):
            info = ", ".join(f"{k}={v}" for k, v in list(data.items())[:3])
            print(f"    {section}: {info}")

    brain.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
