"""
consolidation_cron.py — Standalone entry point for scheduled consolidation.

Run this on a schedule (e.g., Task Scheduler on Windows) to keep the
SlowLoop running independently of the FastLoop:

  # Hourly lightweight pass
  python -m throng4.runners.consolidation_cron --mode hourly

  # Nightly full consolidation
  python -m throng4.runners.consolidation_cron --mode nightly

The cron script:
  1. Opens the DB
  2. Runs SlowLoop.consolidate(mode)
  3. Saves report to experiments/consolidation/YYYY-MM-DD_HH-MM_{mode}.md
  4. Prints summary and exits with code 0 (success) or 1 (error)

Exit codes:
  0 — consolidation completed successfully
  1 — error (check stderr)
  2 — no new episodes to process (not an error, just nothing to do)
"""

import sys
import os
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from throng4.storage.experiment_db import ExperimentDB
from throng4.storage.policy_pack import PromotionGates
from throng4.runners.slow_loop import SlowLoop


def main():
    parser = argparse.ArgumentParser(
        description='Consolidation cron — runs SlowLoop and exits'
    )
    parser.add_argument('--mode', default='hourly',
                        choices=['hourly', 'nightly', 'full'],
                        help='Consolidation window')
    parser.add_argument('--game', default='tetris',
                        help='Game to consolidate for')
    parser.add_argument('--db',   default='experiments/experiments.db',
                        help='ExperimentDB path')
    parser.add_argument('--min-evidence', type=int,   default=30)
    parser.add_argument('--min-win-rate', type=float, default=0.20)
    parser.add_argument('--report-dir', default='experiments/consolidation',
                        help='Directory for markdown reports')
    args = parser.parse_args()

    try:
        gates = PromotionGates(
            min_evidence=args.min_evidence,
            min_win_rate=args.min_win_rate,
        )

        with ExperimentDB(args.db) as db:
            slow = SlowLoop(
                db=db,
                game=args.game,
                gates=gates,
                report_dir=args.report_dir,
            )
            report = slow.consolidate(mode=args.mode)

        if report.episodes_processed == 0:
            print("No new episodes — nothing to consolidate.")
            sys.exit(2)

        print(f"\nConsolidation complete.")
        print(f"  Episodes processed: {report.episodes_processed}")
        print(f"  PolicyPack v{report.pack_version}: "
              f"{report.pack_promoted} active, {report.pack_rejected} rejected")
        print(f"  New candidates: {len(report.candidates_generated)}")
        print(f"  Elapsed: {report.elapsed_s:.1f}s")
        sys.exit(0)

    except Exception:
        print("ERROR during consolidation:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
