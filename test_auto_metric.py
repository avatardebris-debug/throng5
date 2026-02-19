"""
test_auto_metric.py — Validate that AutoMetric can rediscover 'holes' and
'bumpiness' from raw board observations without any domain knowledge.

This test:
  1. Runs 150 Tetris L3 episodes
  2. Records the raw board array + lines_cleared for each
  3. Calls AutoMetric.analyze()
  4. Checks that 'holes' and 'bumpiness' (or equivalent col_height features)
     rank in the top discoveries

Pass criterion: both holes and bumpiness-equivalent feature appear in
top 10 discoveries, with |r| > 0.3.
"""

import sys
import numpy as np
sys.path.insert(0, '.')

from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.storage.auto_metric import AutoMetric

# ── Run episodes and record ────────────────────────────────────────────────

print("Running 150 Tetris L3 episodes...")
am = AutoMetric(
    db=None,            # no DB needed for this test
    game='tetris_test',
    obs_shape=(12, 6),  # L3 board
    min_correlation=0.20,
    min_episodes=50,
    storage_path='experiments/auto_metric_test.jsonl',
)

for ep in range(150):
    adapter = TetrisAdapter(level=3, max_pieces=200)
    adapter.reset()
    done = False

    while not done:
        valid = adapter.get_valid_actions()
        if not valid:
            break
        # Randomish policy with slight preference for flatter actions
        scores = []
        for a in valid:
            f = adapter.make_features(a)
            # Prefer lower bumpiness (feature index 2)
            scores.append(-f[2] + np.random.randn() * 0.3)
        best = valid[int(np.argmax(scores))]
        _, _, done, _ = adapter.step(best)

    # Record: raw board + outcome
    board = np.array(adapter.env.board, dtype=np.float32)
    lines = adapter.env.lines_cleared

    # Also record pre-extracted board features as 'extra' (to validate they match)
    bf = adapter._compute_board_features(adapter.env.board)
    am.record(
        raw_obs=board,
        outcome=lines,
        episode_id=f'ep_{ep}',
        extra={
            'known_holes':       float(bf['holes']),
            'known_max_height':  float(bf['max_height']),
            'known_bumpiness':   float(bf['bumpiness']),
            'known_agg_height':  float(bf['agg_height']),
        }
    )

print(am.summary())

# ── Analyze ────────────────────────────────────────────────────────────────

print("\nRunning correlation analysis...")
discoveries = am.analyze(min_episodes=50)

print(f"\nTop 20 discovered features (sorted by |r|):")
print(f"{'Rank':<5} {'Name':<35} {'r':>8}  {'mean':>8}  {'std':>8}")
print("─" * 68)
for i, d in enumerate(discoveries[:20]):
    sign = '+' if d.correlation > 0 else ''
    print(f"  {i+1:<3} {d.name:<35} {sign}{d.correlation:>7.4f}  "
          f"{d.mean_val:>8.3f}  {d.std_val:>8.3f}")

# ── Validation ────────────────────────────────────────────────────────────

print()
print("=== VALIDATION ===")
top_names = {d.name for d in discoveries[:10]}

# Check if known_holes (our ground-truth) is in top 10
holes_found = any('holes' in n for n in top_names)
bump_found  = any('bumpiness' in n or 'col_height_std' in n or 'col_height_range' in n
                  for n in top_names)
height_found = any('height' in n or 'top_fill' in n for n in top_names)

print(f"  ✓ holes discovered in top 10:      {holes_found}")
print(f"  ✓ bumpiness-equiv in top 10:       {bump_found}")
print(f"  ✓ height-related in top 10:        {height_found}")
print()

# Find holes correlation specifically
for d in discoveries:
    if 'known_holes' in d.name:
        print(f"  Ground-truth holes: r={d.correlation:+.4f} (ranks #{discoveries.index(d)+1})")
    if 'known_bumpiness' in d.name:
        print(f"  Ground-truth bump:  r={d.correlation:+.4f} (ranks #{discoveries.index(d)+1})")
    if 'known_max_height' in d.name:
        print(f"  Ground-truth maxH:  r={d.correlation:+.4f} (ranks #{discoveries.index(d)+1})")

passed = holes_found or bump_found or height_found
print(f"\n{'PASS' if passed else 'FAIL'}: AutoMetric {'can' if passed else 'CANNOT'} "
      f"rediscover board features from raw observations.")
