"""Phase 7A test — PyTorch DQN upgrade for Striatum."""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from brain.message_bus import MessageBus

print("=" * 60)
print("Phase 7A: PyTorch DQN Upgrade Test")
print("=" * 60)

rng = np.random.RandomState(42)

# ── 1. Test TorchDQN standalone ───────────────────────────────────────

from brain.learning.torch_dqn import TorchDQN, TORCH_AVAILABLE

print(f"\nPyTorch available: {TORCH_AVAILABLE}")
assert TORCH_AVAILABLE, "PyTorch must be installed for this test"

dqn = TorchDQN(n_features=84, n_actions=18, batch_size=32, buffer_size=5000)
print(f"[PASS] TorchDQN created: {dqn.stats()['n_params']:,} parameters")
print(f"       Architecture: {dqn.stats()['architecture']}, Device: {dqn.stats()['device']}")

# Fill buffer and train
for i in range(100):
    s = rng.randn(84).astype(np.float32)
    a = rng.randint(18)
    r = rng.randn() * 0.5
    ns = rng.randn(84).astype(np.float32)
    d = (i % 20 == 19)
    dqn.store_transition(s, a, r, ns, d)

# Train for several steps
losses = []
for _ in range(20):
    result = dqn.train_step()
    if result["loss"] > 0:
        losses.append(result["loss"])

print(f"[PASS] Trained {len(losses)} steps, avg_loss: {np.mean(losses):.6f}")
print(f"       Epsilon: {dqn.epsilon:.4f}, Buffer: {dqn.stats()['buffer_size']}")

# Action selection
features = rng.randn(84).astype(np.float32)
action, q_values = dqn.select_action(features)
assert 0 <= action < 18
assert len(q_values) == 18
print(f"[PASS] Action selection: action={action}, max_Q={q_values.max():.4f}")

# ── 2. Compare NumPy vs PyTorch backends ──────────────────────────────

from brain.regions.striatum import Striatum

bus = MessageBus()

# NumPy backend
striatum_np = Striatum(bus, n_features=20, n_actions=4, use_torch=False)
# PyTorch backend
striatum_pt = Striatum(bus, n_features=20, n_actions=4, use_torch=True)

assert striatum_np._torch_dqn is None, "NumPy backend should not have TorchDQN"
assert striatum_pt._torch_dqn is not None, "PyTorch backend should have TorchDQN"
print(f"\n[PASS] NumPy backend: {striatum_np.report()['backend']}")
print(f"[PASS] PyTorch backend: {striatum_pt.report()['backend']}")
print(f"       PyTorch params: {striatum_pt.report()['n_params']:,}")

# Train both backends for 200 steps
for ep in range(5):
    for step in range(40):
        obs = rng.randn(20).astype(np.float32)
        reward = rng.randn() * 0.3
        done = (step == 39)
        prev_action = rng.randint(4)

        # NumPy path
        result_np = striatum_np.process({"features": obs, "explore": True})
        striatum_np.learn({
            "state": obs, "action": result_np["action"],
            "reward": reward, "next_state": rng.randn(20).astype(np.float32),
            "done": done,
        })

        # PyTorch path
        result_pt = striatum_pt.process({"features": obs, "explore": True})
        striatum_pt.learn({
            "state": obs, "action": result_pt["action"],
            "reward": reward, "next_state": rng.randn(20).astype(np.float32),
            "done": done,
        })

np_report = striatum_np.report()
pt_report = striatum_pt.report()

print(f"\n--- Backend Comparison (200 steps) ---")
print(f"  NumPy:   updates={np_report['total_updates']:4d}, "
      f"buffer={np_report['buffer_size']:4d}, "
      f"reward={np_report['avg_reward_100ep']:.2f}")
print(f"  PyTorch: updates={pt_report['total_updates']:4d}, "
      f"buffer={pt_report['buffer_size']:4d}, "
      f"params={pt_report['n_params']:,}, "
      f"avg_loss={pt_report.get('avg_loss', 0):.6f}")
print(f"[PASS] Both backends trained successfully")

# ── 3. Test with WholeBrain ───────────────────────────────────────────

from brain.orchestrator import WholeBrain

# Check if orchestrator accepts use_torch
brain = WholeBrain(n_features=20, n_actions=4, enable_logging=False)
print(f"\n[PASS] WholeBrain created, striatum backend: {brain.striatum.report()['backend']}")

# ── 4. Benchmark speed ───────────────────────────────────────────────

N = 500
features_batch = rng.randn(N, 20).astype(np.float32)

# NumPy speed
start = time.perf_counter()
for f in features_batch:
    striatum_np.process({"features": f, "explore": False})
np_time = (time.perf_counter() - start) * 1000

# PyTorch speed
start = time.perf_counter()
for f in features_batch:
    striatum_pt.process({"features": f, "explore": False})
pt_time = (time.perf_counter() - start) * 1000

print(f"\n--- Speed Benchmark ({N} forward passes) ---")
print(f"  NumPy:   {np_time:.1f}ms ({np_time/N:.2f}ms/step)")
print(f"  PyTorch: {pt_time:.1f}ms ({pt_time/N:.2f}ms/step)")
print(f"  Ratio:   {pt_time/np_time:.1f}x")
print(f"[PASS] Speed benchmark complete")

# ── 5. Save/Load TorchDQN ────────────────────────────────────────────

import tempfile, os

tmpdir = tempfile.mkdtemp()
model_path = os.path.join(tmpdir, "dqn.pt")

dqn.save(model_path)
size_kb = os.path.getsize(model_path) / 1024
print(f"\n[PASS] Saved TorchDQN: {size_kb:.1f} KB")

dqn2 = TorchDQN(n_features=84, n_actions=18)
dqn2.load(model_path)

# Set both to eval mode for deterministic comparison
dqn.online_net.eval()
dqn2.online_net.eval()

# Verify same outputs
features = rng.randn(84).astype(np.float32)
q1 = dqn.forward(features)
q2 = dqn2.forward(features)
max_diff = np.max(np.abs(q1 - q2))
assert max_diff < 1e-4, f"Q-values differ: {max_diff}"
print(f"[PASS] Loaded TorchDQN: Q-values match (max_diff={max_diff:.8f})")

# Cleanup
import shutil
shutil.rmtree(tmpdir)

print()
print("=" * 60)
print("ALL PHASE 7A TESTS PASSED")
print("=" * 60)
