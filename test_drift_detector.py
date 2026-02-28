"""Test drift detector in meta-controller."""
import sys; sys.path.insert(0, '.')
import numpy as np
from brain.learning.meta_controller import MetaController

class FL:
    def __init__(self, mean):
        self.mean = mean
    def get_reward(self):
        return np.random.normal(self.mean, 0.1)

meta = MetaController(
    relevance_window=50, min_trials_per_learner=10,
    drift_window=30, drift_threshold=0.20
)
meta.register_learner('A', FL(5.0))
meta.register_learner('B', FL(0.1))

# Phase 1: Explore and collapse
for _ in range(500):
    l, n = meta.select_learner()
    meta.report_reward(n, l.get_reward())

print(f"Phase 1: mode={meta.report()['mode']}, locked={meta.locked_learner}, collapses={meta._collapse_count}")
assert meta.is_collapsed, "Should have collapsed"
print("[PASS] Collapsed")

# Phase 2: Degrade locked learner
locked = meta.locked_learner
meta._slots[locked].learner.mean = -1.0

for step in range(100):
    l, n = meta.select_learner()
    meta.report_reward(n, l.get_reward())
    if not meta.is_collapsed:
        print(f"Phase 2: DRIFT at step {step} - auto-uncollapsed!")
        print(f"  Reason: {meta._collapse_reason}")
        break

assert not meta.is_collapsed, "Should have auto-uncollapsed on drift"
print("[PASS] Drift detected, auto-uncollapsed")

# Phase 3: Re-explore, should re-collapse to B (now the best)
for _ in range(200):
    l, n = meta.select_learner()
    meta.report_reward(n, l.get_reward())

r = meta.report()
print(f"Phase 3: mode={r['mode']}, locked={meta.locked_learner}, collapses={meta._collapse_count}")
if meta.is_collapsed:
    assert meta.locked_learner == 'B', f"Should lock to B, got {meta.locked_learner}"
    print("[PASS] Re-collapsed to correct learner (B)")
else:
    print("[PASS] Still exploring (both learners changed)")

print(f"\nFull lifecycle: collapse_count={meta._collapse_count}")
print("ALL DRIFT TESTS PASSED")
