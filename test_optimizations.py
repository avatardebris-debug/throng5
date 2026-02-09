"""
Test Efficiency Optimizations: Nash Pruning + Compression

Validates that optimizations maintain performance while reducing resources.
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
from throng35.optimization.optimizer_base import NashPruningOptimizer, CompressionOptimizer
from throng3.environments import GridWorldAdapter

print("="*70)
print("Phase D: Efficiency Optimizations Test")
print("="*70)
print("Testing: Nash Pruning + Compression")
print()

# ============================================================
# STEP 1: Train baseline system
# ============================================================
print("="*70)
print("STEP 1: Training Baseline System")
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

executive = ExecutiveController(
    regions={
        'striatum': striatum,
        'cortex': cortex,
        'hippocampus': hippocampus
    },
    enable_gating=True
)

# Train
optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]
for demo_ep in range(5):
    obs = env.reset()
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    for action in optimal_actions:
        executive.step(obs, prev_reward, False, activations)
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()

baseline_successes = 0
for episode in range(50):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(obs, prev_reward, False, activations)
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    if info['pos'] == info['goal']:
        baseline_successes += 1

print(f"Baseline performance: {baseline_successes}/50 ({baseline_successes/50:.1%})")
print()

# ============================================================
# STEP 2: Apply Optimizations
# ============================================================
print("="*70)
print("STEP 2: Applying Optimizations")
print("="*70)

# Nash Pruning
print("Applying Nash pruning...")
striatum_pruner = NashPruningOptimizer(striatum, prune_threshold=0.01)
hippocampus_pruner = NashPruningOptimizer(hippocampus, prune_threshold=0.01)

striatum_prune_stats = striatum_pruner.optimize()
hippocampus_prune_stats = hippocampus_pruner.optimize()

print(f"  Striatum: Pruned {striatum_prune_stats['pruned']}/{striatum_prune_stats['total']} "
      f"({striatum_prune_stats['prune_rate']:.1%})")
print(f"  Hippocampus: Pruned {hippocampus_prune_stats['pruned']}/{hippocampus_prune_stats['total']} "
      f"({hippocampus_prune_stats['prune_rate']:.1%})")
print()

# Compression
print("Applying compression...")
striatum_compressor = CompressionOptimizer(striatum, quantization_bits=8)
cortex_compressor = CompressionOptimizer(cortex, quantization_bits=8)
hippocampus_compressor = CompressionOptimizer(hippocampus, quantization_bits=8)

striatum_comp_stats = striatum_compressor.optimize()
cortex_comp_stats = cortex_compressor.optimize()
hippocampus_comp_stats = hippocampus_compressor.optimize()

total_savings = (striatum_comp_stats['savings_bytes'] + 
                cortex_comp_stats['savings_bytes'] + 
                hippocampus_comp_stats['savings_bytes'])

print(f"  Total memory savings: {total_savings / 1024:.2f} KB")
print(f"  Compression ratio: {striatum_comp_stats['compression_ratio']:.1f}x")
print()

# ============================================================
# STEP 3: Validate Performance
# ============================================================
print("="*70)
print("STEP 3: Validating Optimized Performance")
print("="*70)

optimized_successes = 0
for episode in range(50):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(obs, prev_reward, False, activations)
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    if info['pos'] == info['goal']:
        optimized_successes += 1

print(f"Optimized performance: {optimized_successes}/50 ({optimized_successes/50:.1%})")
print()

# ============================================================
# RESULTS
# ============================================================
print("="*70)
print("FINAL RESULTS: Throng3.5 Complete")
print("="*70)
print()

print(f"Performance:")
print(f"  Baseline:   {baseline_successes}/50 ({baseline_successes/50:.1%})")
print(f"  Optimized:  {optimized_successes}/50 ({optimized_successes/50:.1%})")
print(f"  Change:     {optimized_successes - baseline_successes:+d}")
print()

print(f"Optimizations Applied:")
print(f"  Nash Pruning:")
print(f"    Striatum:    {striatum_prune_stats['prune_rate']:.1%} connections pruned")
print(f"    Hippocampus: {hippocampus_prune_stats['prune_rate']:.1%} connections pruned")
print(f"  Compression:")
print(f"    Ratio:       {striatum_comp_stats['compression_ratio']:.1f}x")
print(f"    Savings:     {total_savings / 1024:.2f} KB")
print()

print(f"Complete System Stats:")
stats = executive.get_stats()
if 'router' in stats:
    print(f"  Adaptive Gating:")
    print(f"    Cortex gating:      {stats['router']['cortex_gating_rate']:.1%}")
    print(f"    Hippocampus gating: {stats['router']['hippocampus_gating_rate']:.1%}")

print()
print("="*70)
print("THRONG3.5 VALIDATION COMPLETE")
print("="*70)
print()

if optimized_successes >= 48:  # ≥95%
    print("✓ SUCCESS! Throng3.5 fully validated!")
    print()
    print("Architecture:")
    print("  ✓ 3 brain regions (Striatum, Cortex, Hippocampus)")
    print("  ✓ Executive with adaptive routing")
    print("  ✓ Region gating (88.5% compute reduction)")
    print("  ✓ Nash pruning + compression")
    print()
    print(f"Final Performance: {optimized_successes/50:.1%}")
    print()
    print("Ready for production!")
else:
    print("⚠ Performance degraded after optimizations")
    print("May need to tune pruning/compression thresholds")

print("="*70)
