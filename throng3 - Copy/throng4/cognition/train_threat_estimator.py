"""
train_threat_estimator.py — Train ThreatEstimator from ExperimentDB and save.

Trains on all Tetris episodes in the DB (all levels), then saves per-level
estimators for use in FastLoop.

Usage:
    python throng4/cognition/train_threat_estimator.py
    python throng4/cognition/train_threat_estimator.py --level 7
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from throng4.storage import ExperimentDB
from throng4.cognition.threat_estimator import ThreatEstimator
import numpy as np


def train_for_level(db, level: int, n_episodes: int = 3000, verbose: bool = True):
    te = ThreatEstimator(n_features=11, k_steps=5, threshold=0.60)
    summary = te.train_from_db(db, game='tetris', n_episodes=n_episodes,
                               level=level, verbose=verbose)

    if 'error' in summary:
        print(f"  L{level}: {summary['error']}")
        return None

    path = f'experiments/threat_estimator_L{level}.npz'
    te.save(path)
    print(f"  L{level}: acc={summary['final_acc']:.1%}  "
          f"loss={summary['final_loss']:.4f}  "
          f"n={summary['n_samples']}  "
          f"(+{summary['n_positive']} threats, -{summary['n_negative']} safe)")
    return te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db',      default='experiments/experiments.db')
    parser.add_argument('--level',   type=int, default=None,
                        help='Train for specific level (default: all levels)')
    parser.add_argument('--episodes', type=int, default=3000)
    args = parser.parse_args()

    db = ExperimentDB(args.db)

    print(f"Training ThreatEstimator from {args.db}...")
    print()

    if args.level:
        levels = [args.level]
    else:
        # Train for all levels that have enough data
        stats = db.summary_stats()
        levels = [r['level'] for r in stats['episodes_by_level'] if r['n'] >= 100]
        print(f"Found levels with ≥100 episodes: {levels}")
        print()

    # Also train a cross-level "universal" estimator
    if args.level is None:
        print("=== Universal estimator (all levels) ===")
        te_all = ThreatEstimator(n_features=11, k_steps=5, threshold=0.60)
        s = te_all.train_from_db(db, game='tetris', n_episodes=5000, level=None)
        if 'error' not in s:
            te_all.save('experiments/threat_estimator_all.npz')
            print(f"Universal: acc={s['final_acc']:.1%}  n={s['n_samples']}\n")

    print("=== Per-level estimators ===")
    for level in levels:
        train_for_level(db, level, n_episodes=args.episodes)

    db.close()
    print("\nDone. Estimators saved to experiments/threat_estimator_L*.npz")

    # Quick self-test
    print("\n=== Self-test: loading + predicting ===")
    import glob
    for path in sorted(glob.glob('experiments/threat_estimator_*.npz')):
        te = ThreatEstimator.load(path)
        # Test: full board (high threat)
        high_threat_features = np.array([
            0.95,  # max_height / board_h  (nearly full)
            0.30,  # hole density (lots of holes)
            0.40,  # bumpiness norm
            0.05,  # few lines (bad game)
            0.10,  # few pieces (died early)
            0.90,  # height^2
            1.0,   # critical height flag
            1.0,   # many holes flag
            1.0,   # very many holes flag
            1.0,   # high bumpiness flag
            0.80,  # level 5-6
        ], dtype=np.float32)
        # Test: flat board (low threat)
        low_threat_features = np.array([
            0.20,  # max_height / board_h  (mostly empty)
            0.01,  # very few holes
            0.05,  # very flat
            0.80,  # many lines cleared (good game)
            0.70,  # lots of pieces placed (survived long)
            0.04,  # height^2 (low)
            0.0,   # not critical
            0.0,   # no holes flag
            0.0,   # no very-many-holes flag
            0.0,   # not bumpy
            0.40,  # level ~3
        ], dtype=np.float32)

        p_high = te.predict(high_threat_features)
        p_low  = te.predict(low_threat_features)
        mode_h = te.mode(high_threat_features)
        mode_l = te.mode(low_threat_features)
        print(f"  {os.path.basename(path):<40}  "
              f"P(threat|full_board)={p_high:.3f} [{mode_h}]  "
              f"P(threat|flat_board)={p_low:.3f} [{mode_l}]")


if __name__ == '__main__':
    main()
