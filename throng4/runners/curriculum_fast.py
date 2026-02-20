"""
curriculum_fast.py — FastLoop curriculum runner: L2→L7 at full speed.

Runs each level with the FastLoop (zero LLM overhead), then fires the
SlowLoop after each level to update hypotheses and refresh the PolicyPack.

Usage:
    python throng4/runners/curriculum_fast.py
    python throng4/runners/curriculum_fast.py --start-level 5 --max-level 7
    python throng4/runners/curriculum_fast.py --overnight   # full L2-L7 run
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from throng4.storage.experiment_db import ExperimentDB
from throng4.storage.telemetry_logger import TelemetryLogger
from throng4.storage.policy_pack import PolicyPack, PromotionGates
from throng4.runners.fast_loop import FastLoop
from throng4.runners.slow_loop import SlowLoop

# ── Curriculum config ──────────────────────────────────────────────────────

EPISODES_PER_LEVEL = {
    2: 500,
    3: 500,
    4: 500,
    5: 500,
    6: 500,
    7: 2000,   # Final level gets the bulk of training
}

ADVANCE_THRESHOLD = {
    2: 5.0,
    3: 15.0,
    4: 10.0,
    5: 20.0,
    6: 10.0,
    7: 0.0,    # Always completes
}


def run_curriculum(
    start_level: int = 2,
    max_level: int = 7,
    db_path: str = 'experiments/experiments.db',
    log_dir: str = 'experiments/logs',
    consolidate_between_levels: bool = True,
    save_weights: bool = True,
    verbose: bool = True,
):
    """
    Run FastLoop curriculum from start_level to max_level.

    After each level, runs a SlowLoop 'nightly' pass to update hypotheses
    and write a new PolicyPack. The next level loads the fresh pack.

    Args:
        start_level:                  First curriculum level.
        max_level:                    Last curriculum level (inclusive).
        db_path:                      ExperimentDB path.
        log_dir:                      TelemetryLogger directory.
        consolidate_between_levels:   Run SlowLoop after each level.
        save_weights:                 Save agent weights after each level.
        verbose:                      Print progress.
    """
    db     = ExperimentDB(db_path)
    logger = TelemetryLogger(log_dir)
    gates  = PromotionGates(min_evidence=30, min_win_rate=0.20)

    all_stats = []
    t_total   = time.time()

    print(f"\n{'='*70}")
    print(f"FAST CURRICULUM  L{start_level}→L{max_level}")
    print(f"DB: {db_path}")
    print(f"{'='*70}")

    prev_agent = None   # Carry agent weights across levels

    for level in range(start_level, max_level + 1):
        episodes = EPISODES_PER_LEVEL.get(level, 500)
        threshold = ADVANCE_THRESHOLD.get(level, 0.0)

        print(f"\n{'─'*70}")
        print(f"Level {level}  ({episodes} episodes, advance threshold: {threshold:.1f} lines)")
        print(f"{'─'*70}")

        # Load latest PolicyPack for this level
        pack = PolicyPack.load_latest(db, game='tetris')
        if pack:
            print(f"  Using PolicyPack v{pack.version} ({len(pack)} hypotheses)")

        # Create FastLoop — transfer agent weights from previous level
        loop = FastLoop(
            game='tetris',
            level=level,
            db=db,
            logger=logger,
            pack=pack,
            pack_refresh_interval=200,
        )

        # Transfer weights from previous level if available.
        # Note: levels with different board widths have different state sizes,
        # so weight transfer fails on shape mismatch (e.g. L4 6-wide → L5 8-wide).
        # This is expected and correct — agent starts fresh at new board width.
        if prev_agent is not None:
            weights_path = f'experiments/weights_L{level - 1}.npz'
            if os.path.exists(weights_path):
                try:
                    loop.agent.load_weights(weights_path)
                    print(f"  ✓ Loaded weights from L{level - 1}")
                except ValueError as e:
                    print(f"  ⚠ L{level-1}→L{level} weight transfer skipped "
                          f"(board width changed, state size mismatch) — starting fresh")
                except Exception as e:
                    print(f"  ⚠ Weight load failed: {e} — starting fresh")

        # Run the level
        stats = loop.run(n_episodes=episodes, verbose=verbose)
        all_stats.append({'level': level, **stats})

        # Save weights for next level
        if save_weights:
            weights_path = f'experiments/weights_L{level}.npz'
            try:
                loop.agent.save_weights(weights_path)
                print(f"  ✓ Weights saved → {weights_path}")
            except Exception as e:
                print(f"  ⚠ Weight save failed: {e}")
        prev_agent = loop.agent

        # Threshold check (informational — never blocks progress)
        mean_lines = stats['mean_lines']
        final_mean = stats['final_mean_lines']
        if mean_lines >= threshold:
            print(f"  ✅ L{level}: mean={mean_lines:.1f} >= {threshold:.1f} — threshold met")
        else:
            print(f"  ⚠️  L{level}: mean={mean_lines:.1f} < {threshold:.1f} — below threshold (continuing anyway)")

        # SlowLoop consolidation between levels
        if consolidate_between_levels and level < max_level:
            print(f"\n  [SlowLoop] Consolidating after L{level}...")
            slow   = SlowLoop(db, game='tetris', gates=gates)
            report = slow.consolidate(mode='nightly')
            print(f"  [SlowLoop] PolicyPack v{report.pack_version}: "
                  f"{report.pack_promoted} active, "
                  f"{len(report.candidates_generated)} new candidates")

    # Final consolidation
    print(f"\n  [SlowLoop] Final full consolidation...")
    slow   = SlowLoop(db, game='tetris', gates=gates)
    report = slow.consolidate(mode='full')

    # Print final table
    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"FAST CURRICULUM COMPLETE  ({elapsed:.0f}s total)")
    print(f"{'='*70}")
    print(f"  {'Level':<8} {'Episodes':<12} {'Mean Lines':<15} {'Final Mean':<15} {'Max':<8} {'ep/s':<8}")
    print(f"  {'─'*8} {'─'*12} {'─'*15} {'─'*15} {'─'*8} {'─'*8}")
    for s in all_stats:
        print(f"  {s['level']:<8} {s['episodes']:<12} {s['mean_lines']:<15.2f} "
              f"{s['final_mean_lines']:<15.2f} {s['max_lines']:<8} {s['eps_per_s']:<8.1f}")
    print(f"{'='*70}\n")

    # DB summary
    db_stats = db.summary_stats()
    print(f"DB after run:")
    print(f"  {db_stats['total_episodes']} total episodes, "
          f"{db_stats['total_hypotheses']} hypotheses, "
          f"PolicyPack v{report.pack_version} ({report.pack_promoted} active)")
    for row in db_stats['episodes_by_level']:
        print(f"  L{row['level']}: {row['n']} eps, "
              f"avg={row['avg_lines']:.1f}, max={row['max_lines']}")

    # Save summary JSON
    summary_path = 'experiments/curriculum_fast_results.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'levels': all_stats,
            'elapsed_s': round(elapsed, 1),
            'db_stats': db_stats,
            'pack_version': report.pack_version,
        }, f, indent=2)
    print(f"\nResults saved → {summary_path}")

    db.close()
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='FastLoop curriculum runner')
    parser.add_argument('--start-level', type=int, default=2)
    parser.add_argument('--max-level',   type=int, default=7)
    parser.add_argument('--db',          default='experiments/experiments.db')
    parser.add_argument('--log-dir',     default='experiments/logs')
    parser.add_argument('--overnight',   action='store_true',
                        help='Full L2-L7 run with L7=5000 episodes')
    parser.add_argument('--no-consolidate', action='store_true',
                        help='Skip SlowLoop between levels')
    args = parser.parse_args()

    if args.overnight:
        EPISODES_PER_LEVEL[7] = 5000
        print("Overnight mode: L7 gets 5000 episodes")

    run_curriculum(
        start_level=args.start_level,
        max_level=args.max_level,
        db_path=args.db,
        consolidate_between_levels=not args.no_consolidate,
    )


if __name__ == '__main__':
    main()
