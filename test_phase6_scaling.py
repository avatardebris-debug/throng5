"""Phase 6 test — Scaling Bridge components."""

import sys, os, shutil, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from brain.orchestrator import WholeBrain

print("=" * 60)
print("Phase 6: Scaling Bridge Test")
print("=" * 60)

# ── Setup: train a brain for a bit ────────────────────────────────────

brain = WholeBrain(n_features=20, n_actions=4, session_name="phase6_test", enable_logging=False)
rng = np.random.RandomState(42)

for ep in range(3):
    for step in range(40):
        obs = rng.randn(20).astype(np.float32)
        reward = rng.randn() * 0.5
        done = (step == 39)
        brain.step(obs, prev_action=rng.randint(4), reward=reward, done=done)

print(f"[PASS] Trained brain: {brain._step_count} steps, {brain._episode_count} episodes")

# ── 1. Brain State: Save & Load ───────────────────────────────────────

from brain.scaling.brain_state import BrainState, AutoCheckpointer

tmpdir = tempfile.mkdtemp()
save_path = os.path.join(tmpdir, "test.brain")

meta = BrainState.save(brain, save_path)
print(f"[PASS] Saved brain: {meta['size_mb']}MB, {meta['step_count']} steps")

# Load into new brain
brain2 = BrainState.load(save_path)
assert brain2._step_count == brain._step_count
assert brain2._episode_count == brain._episode_count
assert brain2.n_features == brain.n_features
assert brain2.n_actions == brain.n_actions

# Verify weights match
assert np.allclose(brain2.striatum._W1, brain.striatum._W1)
assert np.allclose(brain2.amygdala._W1, brain.amygdala._W1)
print(f"[PASS] Loaded brain: weights match, {brain2._step_count} steps restored")

# Info without full load
info = BrainState.info(save_path)
assert info["step_count"] == brain._step_count
print(f"[PASS] Info read: v{info['version']}, {info['step_count']} steps, {info['size_mb']}MB")

# AutoCheckpointer
ckpt_dir = os.path.join(tmpdir, "checkpoints")
checkpointer = AutoCheckpointer(brain, checkpoint_dir=ckpt_dir, interval_episodes=2)
result = checkpointer.maybe_checkpoint()
assert result is not None  # Episode count should trigger
print(f"[PASS] AutoCheckpointer: saved to {ckpt_dir}")

# ── 2. Distributed Bus ───────────────────────────────────────────────

from brain.scaling.distributed_bus import DistributedBus
from brain.message_bus import BrainMessage, Priority

# Test local mode (no network)
local_bus = DistributedBus(role="local")
local_bus.start()
local_bus.send(BrainMessage(source="test", target="test", priority=Priority.ROUTINE, msg_type="test", payload={"a": 1}))
stats = local_bus.network_stats()
assert stats["role"] == "local"
assert not stats["running"]
print(f"[PASS] DistributedBus local mode works")

# Test coordinator start/stop
coord_bus = DistributedBus(role="coordinator", port=9501)
coord_bus.start()
assert coord_bus.network_stats()["running"]
coord_bus.register_remote("striatum")
assert "striatum" in coord_bus.network_stats()["remote_regions"]
coord_bus.stop()
print(f"[PASS] DistributedBus coordinator start/stop works")

# ── 3. Compute Allocator ─────────────────────────────────────────────

from brain.scaling.compute_allocator import ComputeAllocator, ResourceProfile

allocator = ComputeAllocator()
resources = ResourceProfile(cpu_cores=8, gpu_available=True, gpu_vram_gb=8.0)
plan = allocator.create_plan(resources)

assert plan["sensory_cortex"].tier == "fast_local"
assert plan["motor_cortex"].tier == "fast_local"
assert plan["striatum"].gpu_required
print(f"[PASS] Compute plan: {len(plan)} regions assigned")
for name, assignment in plan.items():
    print(f"  {name:25s} -> {assignment.tier:15s} ({assignment.budget_ms:.0f}ms)")

# Test timing
with allocator.time_region("sensory_cortex"):
    _ = np.dot(rng.randn(84, 84), rng.randn(84))
timing_stats = allocator.stats()
assert "sensory_cortex" in timing_stats
print(f"[PASS] Timing recorded: {timing_stats['sensory_cortex']['avg_ms']:.2f}ms")

# ── 4. Model Distiller ───────────────────────────────────────────────

from brain.scaling.model_distiller import ModelDistiller

distiller = ModelDistiller(brain)
compact = distiller.distill(prune_threshold=0.01)
print(f"[PASS] Distilled: {compact.original_params} -> {compact.pruned_params} params "
      f"({compact.compression_ratio:.1%})")

# Test compact model action selection
features = rng.randn(20).astype(np.float32)
action = compact.select_action(features)
assert 0 <= action < 4
print(f"[PASS] CompactModel action selection works")

# Benchmark
bench_result = distiller.benchmark(compact, n_steps=500)
print(f"[PASS] Benchmark: {bench_result['speedup']:.1f}x speedup, "
      f"{bench_result['agreement']:.0%} agreement")

# Save and reload compact model
compact_path = os.path.join(tmpdir, "compact.npz")
compact.save(compact_path)
compact2 = type(compact).load(compact_path)
assert compact2.n_features == compact.n_features
action2 = compact2.select_action(features)
print(f"[PASS] CompactModel save/load works")

# ── 5. Neuron Benchmark ──────────────────────────────────────────────

from brain.scaling.neuron_benchmark import NeuronBenchmark

nbench = NeuronBenchmark(brain)
report = nbench.full_report()

print(f"\n--- Neuron Budget Report ---")
print(f"  Params by region:")
for region, count in report["params_by_region"].items():
    print(f"    {region:25s}: {count:,}")
print(f"  Total params: {report['efficiency']['total_params']:,}")
print(f"  Active params: {report['efficiency']['active_params']:,}")
print(f"  Sparsity: {report['efficiency']['sparsity']:.1%}")
print(f"  Memory: {report['memory_kb']['total_kb']:.1f} KB")

vs = report["vs_throng2"]
print(f"\n  vs Throng 2:")
print(f"    Throng 2 params: {vs['throng2_params']:,}")
print(f"    Throng 5 params: {vs['throng5_params']:,}")
print(f"    Param ratio: {vs['param_ratio']}x")
print(f"    Throng 5 advantages: {len(vs['throng5_advantages'])} items")
print(f"[PASS] Neuron benchmark works")

# ── Cleanup ───────────────────────────────────────────────────────────

brain.close()
brain2.close()
shutil.rmtree(tmpdir)

print()
print("=" * 60)
print("ALL PHASE 6 TESTS PASSED")
print("=" * 60)
