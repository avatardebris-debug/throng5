"""
Test Enhanced Executive with Adaptive Routing

Validates that region gating improves efficiency while maintaining performance.
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.regions.cortex import CortexRegion
from throng35.regions.hippocampus import HippocampusRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.learning.hebbian import HebbianConfig
from throng35.learning.stdp import STDPConfig
from throng3.environments import GridWorldAdapter

print("="*70)
print("Enhanced Executive Test: Adaptive Routing + Region Gating")
print("="*70)
print("Goal: Maintain ≥95% success with ≥20% compute reduction")
print()

# ============================================================
# TEST 1: Baseline (No Gating)
# ============================================================
print("="*70)
print("TEST 1: Baseline (No Gating)")
print("="*70)

env = GridWorldAdapter()

striatum = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

cortex = CortexRegion(
    n_neurons=100,
    n_features=10,
    config=HebbianConfig(learning_rate=0.01)
)

hippocampus = HippocampusRegion(
    n_neurons=50,
    sequence_length=10,
    config=STDPConfig()
)

# Executive WITHOUT gating
executive_baseline = ExecutiveController(
    regions={
        'striatum': striatum,
        'cortex': cortex,
        'hippocampus': hippocampus
    },
    enable_gating=False  # Baseline: no gating
)

print("Running baseline (no gating)...")

# Curriculum
optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]
for demo_ep in range(5):
    obs = env.reset()
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    for action in optimal_actions:
        executive_baseline.step(obs, prev_reward, False, activations)
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        activations += np.random.randn(100) * 0.05
    
    executive_baseline.step(obs, prev_reward, True, activations)
    executive_baseline.reset()

# Exploration
baseline_successes = 0
for episode in range(50):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive_baseline.step(obs, prev_reward, False, activations)
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive_baseline.step(obs, prev_reward, True, activations)
    executive_baseline.reset()
    
    if info['pos'] == info['goal']:
        baseline_successes += 1

baseline_stats = executive_baseline.get_stats()
baseline_compute = (
    baseline_stats['striatum']['resource_usage']['compute_ms'] +
    baseline_stats['cortex']['resource_usage']['compute_ms'] +
    baseline_stats['hippocampus']['resource_usage']['compute_ms']
)

print(f"Baseline Results:")
print(f"  Success: {baseline_successes}/50 ({baseline_successes/50:.1%})")
print(f"  Total compute: {baseline_compute:.2f}ms/step")
print()

# ============================================================
# TEST 2: Enhanced Executive (With Gating)
# ============================================================
print("="*70)
print("TEST 2: Enhanced Executive (With Adaptive Gating)")
print("="*70)

# Create fresh regions
striatum2 = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

cortex2 = CortexRegion(
    n_neurons=100,
    n_features=10,
    config=HebbianConfig(learning_rate=0.01)
)

hippocampus2 = HippocampusRegion(
    n_neurons=50,
    sequence_length=10,
    config=STDPConfig()
)

# Executive WITH gating
executive_gated = ExecutiveController(
    regions={
        'striatum': striatum2,
        'cortex': cortex2,
        'hippocampus': hippocampus2
    },
    enable_gating=True,  # Enhanced: adaptive gating
    router_config={
        'cortex_td_threshold': 0.05,
        'hippocampus_sequence_threshold': 3,
        'warmup_steps': 100
    }
)

print("Running with adaptive gating...")

# Curriculum
for demo_ep in range(5):
    obs = env.reset()
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    for action in optimal_actions:
        executive_gated.step(obs, prev_reward, False, activations)
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        activations += np.random.randn(100) * 0.05
    
    executive_gated.step(obs, prev_reward, True, activations)
    executive_gated.reset()

# Exploration
gated_successes = 0
for episode in range(50):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive_gated.step(obs, prev_reward, False, activations)
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive_gated.step(obs, prev_reward, True, activations)
    executive_gated.reset()
    
    if info['pos'] == info['goal']:
        gated_successes += 1

gated_stats = executive_gated.get_stats()
gated_compute = (
    gated_stats['striatum']['resource_usage']['compute_ms'] +
    gated_stats['cortex']['resource_usage']['compute_ms'] +
    gated_stats['hippocampus']['resource_usage']['compute_ms']
)

print(f"Gated Results:")
print(f"  Success: {gated_successes}/50 ({gated_successes/50:.1%})")
print(f"  Total compute: {gated_compute:.2f}ms/step")
print()

# ============================================================
# COMPARISON
# ============================================================
print("="*70)
print("RESULTS: Baseline vs Enhanced Executive")
print("="*70)
print()

print(f"Performance:")
print(f"  Baseline (no gating):  {baseline_successes}/50 ({baseline_successes/50:.1%})")
print(f"  Enhanced (gating):     {gated_successes}/50 ({gated_successes/50:.1%})")
print(f"  Change:                {gated_successes - baseline_successes:+d} ({(gated_successes - baseline_successes)/50:+.1%})")
print()

compute_reduction = (baseline_compute - gated_compute) / baseline_compute * 100
print(f"Efficiency:")
print(f"  Baseline compute:  {baseline_compute:.2f}ms/step")
print(f"  Enhanced compute:  {gated_compute:.2f}ms/step")
print(f"  Reduction:         {compute_reduction:.1f}%")
print()

if 'router' in gated_stats:
    router_stats = gated_stats['router']
    print(f"Gating Statistics:")
    print(f"  Cortex activation:      {router_stats['activation_rates']['cortex']:.1%}")
    print(f"  Hippocampus activation: {router_stats['activation_rates']['hippocampus']:.1%}")
    print(f"  Cortex gating rate:     {router_stats['cortex_gating_rate']:.1%}")
    print(f"  Hippocampus gating:     {router_stats['hippocampus_gating_rate']:.1%}")

print()
print("="*70)
print("VALIDATION")
print("="*70)
print()

performance_maintained = gated_successes >= 48  # ≥95%
efficiency_improved = compute_reduction >= 20

if performance_maintained and efficiency_improved:
    print("✓ SUCCESS! Enhanced Executive validated!")
    print(f"  Performance: {gated_successes/50:.1%} (≥95% target)")
    print(f"  Efficiency: {compute_reduction:.1f}% reduction (≥20% target)")
    print("  Ready for Phase D: Optimizations")
elif performance_maintained:
    print("⚠ PARTIAL SUCCESS")
    print(f"  Performance maintained: {gated_successes/50:.1%}")
    print(f"  Efficiency gain: {compute_reduction:.1f}% (target: ≥20%)")
    print("  May need more aggressive gating")
else:
    print("✗ NEEDS WORK")
    print(f"  Performance degraded: {gated_successes/50:.1%} (target: ≥95%)")
    print("  Gating too aggressive")

print("="*70)
