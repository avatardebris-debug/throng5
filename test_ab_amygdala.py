"""
A/B test: ThreatEstimator active vs disabled.
Runs 750 episodes each at L5, L6, L7 and compares mean/max lines.
"""
import sys, time
sys.path.insert(0, '.')

from throng4.runners.fast_loop import FastLoop

results = {}

for level in [5, 6, 7]:
    for condition in ['active', 'disabled']:
        fl = FastLoop(level=level)

        if condition == 'disabled':
            # Disable the amygdala by nulling the threat estimator
            fl._threat = None

        t0 = time.time()
        summary = fl.run(n_episodes=750, verbose=False)
        elapsed = time.time() - t0

        key = f"L{level}_{condition}"
        results[key] = {
            'mean':    summary['mean_lines'],
            'final':   summary['final_mean_lines'],
            'max':     summary['max_lines'],
            'elapsed': round(elapsed, 1),
        }
        print(
            f"L{level} {condition:<8}: "
            f"mean={summary['mean_lines']:.1f}  "
            f"final={summary['final_mean_lines']:.1f}  "
            f"max={summary['max_lines']}  "
            f"({elapsed:.0f}s)"
        )

print()
print("=== A/B Summary ===")
for level in [5, 6, 7]:
    a = results[f"L{level}_active"]
    d = results[f"L{level}_disabled"]
    delta_mean  = a['mean']  - d['mean']
    delta_final = a['final'] - d['final']
    delta_max   = a['max']   - d['max']
    sign = '+' if delta_mean >= 0 else ''
    print(
        f"L{level}: amygdala {sign}{delta_mean:.1f} mean  "
        f"{'+' if delta_final>=0 else ''}{delta_final:.1f} final  "
        f"{'+' if delta_max>=0 else ''}{delta_max} max"
    )
