"""Phase 3 test — RL Registry and Learner Selector."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.learning.rl_registry import RLRegistry, AlgorithmType, ActionSpaceType
from brain.learning.learner_selector import LearnerSelector
import numpy as np

print("=" * 60)
print("Phase 3: RLZoo Integration Test")
print("=" * 60)

# ── 1. Registry ───────────────────────────────────────────────────────

registry = RLRegistry()
all_algos = registry.list_all()
print(f"[PASS] Registry has {len(all_algos)} algorithms registered")
assert len(all_algos) >= 18, f"Expected >= 18 algorithms, got {len(all_algos)}"

# Check types
value_based = registry.list_by_type(AlgorithmType.VALUE_BASED)
actor_critic = registry.list_by_type(AlgorithmType.ACTOR_CRITIC)
print(f"  Value-based: {len(value_based)}")
print(f"  Actor-critic: {len(actor_critic)}")
assert len(value_based) >= 8
assert len(actor_critic) >= 5

# Check action space filtering
discrete = registry.list_for_action_space(ActionSpaceType.DISCRETE)
continuous = registry.list_for_action_space(ActionSpaceType.CONTINUOUS)
print(f"  Discrete-compatible: {len(discrete)}")
print(f"  Continuous-compatible: {len(continuous)}")

# Create builtin DQN
dqn = registry.create("dqn_builtin", n_features=84, n_actions=18)
assert dqn is not None
q = dqn.get_q_values(np.random.randn(84))
assert q.shape == (18,), f"Expected (18,) Q-values, got {q.shape}"
action = dqn.select_action(np.random.randn(84), epsilon=0.1)
assert 0 <= action < 18
print(f"[PASS] BuiltinDQN created and working (action={action})")

# Train for a few steps
for i in range(50):
    s = np.random.randn(84).astype(np.float32)
    ns = np.random.randn(84).astype(np.float32)
    metrics = dqn.update(s, np.random.randint(18), np.random.randn(), ns, False)
assert metrics.get("loss", -1) >= 0
print(f"[PASS] BuiltinDQN training works (loss={metrics['loss']:.4f})")

# Save/load weights
weights = dqn.save_weights()
dqn2 = registry.create("dqn_builtin", n_features=84, n_actions=18)
dqn2.load_weights(weights)
q1 = dqn.get_q_values(np.ones(84))
q2 = dqn2.get_q_values(np.ones(84))
assert np.allclose(q1, q2), "Weights not transferred correctly"
print(f"[PASS] Weight save/load works")

# Print registry summary
print()
print(registry.summary())
print()

# ── 2. Learner Selector ──────────────────────────────────────────────

selector = LearnerSelector(registry)

# Test: Atari discrete, no GPU, sparse reward
rec = selector.recommend({
    "action_space": "discrete",
    "n_actions": 18,
    "obs_dim": 84,
    "reward_sparsity": "sparse",
    "has_gpu": False,
    "compute_budget": "low",
})
print(f"[PASS] Sparse reward + no GPU + low budget -> {rec.name}")
print(f"  Reason: {rec.reason}")
print(f"  Alternatives: {rec.alternatives}")
assert rec.name == "dqn_builtin", f"Expected dqn_builtin for no-GPU, got {rec.name}"

# Test: Atari with GPU, dense reward
rec2 = selector.recommend({
    "action_space": "discrete",
    "n_actions": 18,
    "obs_dim": 84,
    "reward_sparsity": "dense",
    "has_gpu": True,
    "compute_budget": "high",
})
print(f"[PASS] Dense reward + GPU + high budget -> {rec2.name}")
print(f"  Reason: {rec2.reason}")

# Test: Continuous control (MuJoCo-style)
rec3 = selector.recommend({
    "action_space": "continuous",
    "n_actions": 6,
    "obs_dim": 17,
    "reward_sparsity": "dense",
    "has_gpu": True,
    "compute_budget": "medium",
})
print(f"[PASS] Continuous + GPU -> {rec3.name}")
assert rec3.name in ("sac", "ppo"), f"Expected SAC or PPO for continuous, got {rec3.name}"

# Test: POMDP  
rec4 = selector.recommend({
    "action_space": "discrete",
    "n_actions": 18,
    "obs_dim": 84,
    "reward_sparsity": "moderate",
    "has_gpu": True,
    "is_pomdp": True,
})
print(f"[PASS] POMDP -> {rec4.name}")
# Should prefer recurrent architecture
assert rec4.name in ("drqn", "r2d2"), f"Expected DRQN or R2D2 for POMDP, got {rec4.name}"

# Test: Sparse reward + GPU (should prefer curiosity/exploration algos)
rec5 = selector.recommend({
    "action_space": "discrete",
    "n_actions": 18,
    "obs_dim": 84,
    "reward_sparsity": "sparse",
    "has_gpu": True,
    "compute_budget": "high",
})
print(f"[PASS] Sparse + GPU + high budget -> {rec5.name}")

# Test: create_recommended API
learner, rec6 = selector.create_recommended({
    "action_space": "discrete",
    "n_actions": 4,
    "obs_dim": 20,
    "reward_sparsity": "dense",
    "has_gpu": False,
})
assert learner is not None, "create_recommended returned None learner"
action = learner.select_action(np.random.randn(20), epsilon=0.5)
print(f"[PASS] create_recommended works (action={action}, name={rec6.name})")

print()
print("=" * 60)
print("ALL PHASE 3 TESTS PASSED")
print("=" * 60)
