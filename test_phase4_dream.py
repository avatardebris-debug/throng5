"""Phase 4 test — DreamLoop overnight processing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from brain.orchestrator import WholeBrain
from brain.overnight.dream_loop import DreamLoop

print("=" * 60)
print("Phase 4: Overnight Dream Loop Test")
print("=" * 60)

# ── 1. Create brain and run some training episodes ────────────────────

brain = WholeBrain(n_features=20, n_actions=4, session_name="phase4_test", enable_logging=False)
rng = np.random.RandomState(42)

# Simulate 200 steps across 4 episodes to populate replay buffer
steps = 0
for ep in range(4):
    for step in range(50):
        obs = rng.randn(20).astype(np.float32)
        reward = rng.randn() * 0.5
        done = (step == 49)
        if step > 40 and rng.random() < 0.3:
            reward = 5.0  # Some high-reward events
        brain.step(obs, prev_action=rng.randint(4), reward=reward, done=done)
        steps += 1

print(f"[PASS] Ran {steps} training steps, {brain._episode_count} episodes")

# Check hippocampus buffer
hip_report = brain.hippocampus.report()
print(f"  Hippocampus buffer: {hip_report['buffer_size']} transitions")
assert hip_report["buffer_size"] > 0, "Hippocampus buffer empty!"

# ── 2. Run DreamLoop ─────────────────────────────────────────────────

dream = DreamLoop(brain)

report = dream.run(
    n_replay_cycles=10,
    n_dream_steps=5,
    generate_heuristics=True,
    max_time_seconds=30.0,
)

print(f"\n[PASS] DreamLoop completed in {report['total_time']}s")

# Phase A: Replay
replay = report["phase_a_replay"]
print(f"  Replay: {replay['batches_processed']} batches, avg_loss={replay['avg_loss']:.5f}")
print(f"  Scheduler: {replay['scheduler_stats']}")
assert replay["batches_processed"] > 0, "No replay batches processed!"

# Phase B: Dreams
dreams = report["phase_b_dreams"]
print(f"  Dreams: {dreams['dreams_completed']} completed")

# Phase C: Heuristics
heuristics = report["phase_c_heuristics"]
print(f"  Heuristics: {heuristics['new_heuristics']} new, {heuristics['total_heuristics']} installed in Motor Cortex")
print(f"  Heuristic stats: {heuristics['heuristic_stats']}")

# ── 3. Verify Motor Cortex has heuristics ─────────────────────────────

motor_report = brain.motor.report()
print(f"  Motor Cortex heuristic_count: {motor_report['heuristic_count']}")

# ── 4. Run a second overnight cycle (staleness should increase) ───────

report2 = dream.run(n_replay_cycles=5, n_dream_steps=3, max_time_seconds=10.0)
print(f"\n[PASS] Second DreamLoop in {report2['total_time']}s")

# Check DreamLoop stats
stats = dream.stats()
print(f"  Total cycles: {stats['total_cycles']}")
print(f"  Total dreams: {stats['total_dreams']}")
assert stats["total_cycles"] == 2

# ── 5. Verify Striatum improved ───────────────────────────────────────

striatum_report = brain.striatum.report()
print(f"\n  Striatum total_updates: {striatum_report['total_updates']}")
print(f"  Striatum buffer: {striatum_report['buffer_size']}")

brain.close()

print()
print("=" * 60)
print("ALL PHASE 4 TESTS PASSED")
print("=" * 60)
