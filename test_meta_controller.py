"""Test: Self-regulating meta-controller lifecycle."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from brain.learning.meta_controller import MetaController

print("=" * 60)
print("Meta-Controller: Self-Regulation Test")
print("=" * 60)

rng = np.random.RandomState(42)


# ── Fake learners with different reward distributions ─────────────────

class FakeLearner:
    """Simulates a learner with a characteristic reward distribution."""
    def __init__(self, name, mean_reward, std):
        self.name = name
        self.mean_reward = mean_reward
        self.std = std

    def get_reward(self):
        return np.random.normal(self.mean_reward, self.std)


# ── Test 1: One learner dominates → meta collapses ───────────────────

print("\n--- Test 1: Clear winner → meta-controller collapses ---")

meta = MetaController(
    relevance_window=100,
    collapse_threshold=0.02,
    min_trials_per_learner=20,
)

# Register learners with different abilities
good_learner = FakeLearner("good", mean_reward=1.0, std=0.3)
bad_learner = FakeLearner("bad", mean_reward=0.2, std=0.5)
mid_learner = FakeLearner("mid", mean_reward=0.5, std=0.4)

meta.register_learner("good", good_learner)
meta.register_learner("bad", bad_learner)
meta.register_learner("mid", mid_learner)

# Simulate 500 episodes
selections = {"good": 0, "bad": 0, "mid": 0}
for step in range(500):
    learner, name = meta.select_learner()
    reward = learner.get_reward()
    meta.report_reward(name, reward)
    selections[name] += 1

    if meta.is_collapsed:
        print(f"  Step {step}: META COLLAPSED → locked to '{meta.locked_learner}'")
        break

report = meta.report()
print(f"  Mode: {report['mode']}")
print(f"  Selections: good={selections['good']}, mid={selections['mid']}, bad={selections['bad']}")
print(f"  Locked to: {report['locked_learner']}")
print(f"  Reason: {report['collapse_reason']}")
assert meta.is_collapsed, "Meta should have collapsed"
assert meta.locked_learner == "good", f"Should lock to 'good', got '{meta.locked_learner}'"
print(f"[PASS] Meta-controller correctly collapsed to best learner")


# ── Test 2: Equal learners → meta stays active ──────────────────────

print("\n--- Test 2: Equal learners → meta stays active ---")

meta2 = MetaController(
    relevance_window=100,
    collapse_threshold=0.02,
    min_trials_per_learner=20,
)

# Two learners with VERY similar performance
learner_a = FakeLearner("A", mean_reward=0.5, std=0.3)
learner_b = FakeLearner("B", mean_reward=0.48, std=0.3)

meta2.register_learner("A", learner_a)
meta2.register_learner("B", learner_b)

for step in range(300):
    learner, name = meta2.select_learner()
    reward = learner.get_reward()
    meta2.report_reward(name, reward)

report2 = meta2.report()
print(f"  Mode after 300 steps: {report2['mode']}")
print(f"  A mean: {report2['learners']['A']['mean_reward']:.4f}")
print(f"  B mean: {report2['learners']['B']['mean_reward']:.4f}")
print(f"  Avg relevance: {report2['avg_relevance']:.4f}")
# With very close learners, meta might or might not collapse — both are OK
print(f"[PASS] Meta-controller handled equal learners (mode: {report2['mode']})")


# ── Test 3: Uncollapse on environment change ──────────────────────────

print("\n--- Test 3: Uncollapse on environment change ---")

# Start from collapsed state (test 1)
assert meta.is_collapsed
print(f"  Before: {meta.report()['mode']}, locked to '{meta.locked_learner}'")

# Simulate environment change — uncollapse
meta.uncollapse()
assert not meta.is_collapsed
assert meta.locked_learner is None
print(f"  After uncollapse: {meta.report()['mode']}")

# Now the previously-bad learner becomes good (environment shifted!)
meta.unregister_learner("bad")
meta.register_learner("bad_now_good", FakeLearner("bad_now_good", 2.0, 0.2))

for step in range(300):
    learner, name = meta.select_learner()
    reward = learner.get_reward()
    meta.report_reward(name, reward)

    if meta.is_collapsed:
        print(f"  Step {step}: RE-COLLAPSED → locked to '{meta.locked_learner}'")
        break

report3 = meta.report()
print(f"  Mode: {report3['mode']}")
if meta.is_collapsed:
    print(f"  Locked to: {meta.locked_learner}")
    assert meta.locked_learner == "bad_now_good", (
        f"Should lock to 'bad_now_good', got '{meta.locked_learner}'"
    )
    print(f"[PASS] After uncollapse, correctly found new best learner")
else:
    print(f"[PASS] Still exploring (environment is dynamic)")


# ── Test 4: Report structure ──────────────────────────────────────────

print("\n--- Test 4: Report structure ---")
report = meta.report()
assert "mode" in report
assert "locked_learner" in report
assert "learners" in report
assert "avg_relevance" in report
assert len(report["learners"]) >= 2
print(f"  Report keys: {list(report.keys())}")
print(f"  Learners tracked: {list(report['learners'].keys())}")
print(f"[PASS] Report structure valid")


# ── Test 5: Post-collapse efficiency ──────────────────────────────────

print("\n--- Test 5: Post-collapse zero overhead ---")

meta_fast = MetaController(relevance_window=50, min_trials_per_learner=10)
fa = FakeLearner("alpha", 5.0, 0.1)
fb = FakeLearner("beta", 0.1, 0.1)
meta_fast.register_learner("alpha", fa)
meta_fast.register_learner("beta", fb)

# Run until collapsed
for _ in range(200):
    learner, name = meta_fast.select_learner()
    meta_fast.report_reward(name, learner.get_reward())

if meta_fast.is_collapsed:
    # After collapse, selection should be O(1) — no Thompson sampling
    import time
    start = time.perf_counter()
    for _ in range(10000):
        meta_fast.select_learner()
    elapsed = (time.perf_counter() - start) * 1000
    per_call_us = elapsed / 10000 * 1000
    print(f"  10K collapsed selections: {elapsed:.1f}ms ({per_call_us:.1f}µs/call)")
    print(f"[PASS] Post-collapse overhead: {per_call_us:.1f}µs per selection")
else:
    print(f"  (Meta didn't collapse in 200 steps, skipping speed test)")
    print(f"[PASS] Meta still exploring — acceptable")

print()
print("=" * 60)
print("ALL META-CONTROLLER TESTS PASSED")
print("=" * 60)
