"""
Transfer Learning Test: GridWorld 5x5 → 7x7

Goal: Prove Throng3.5 architecture generalizes by showing:
1. Train on GridWorld 5x5 (original task)
2. Test on GridWorld 7x7 (larger, zero-shot)
3. Compare to training from scratch on 7x7

This tests if the regional architecture enables knowledge transfer!
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
print("Transfer Learning: GridWorld 5x5 → 7x7")
print("="*70)
print("Testing if Throng3.5 architecture enables knowledge transfer")
print()

# ============================================================
# Phase 1: Train on 5x5 GridWorld
# ============================================================
print("="*70)
print("PHASE 1: Training on GridWorld 5x5")
print("="*70)

env_5x5 = GridWorldAdapter(size=5)

striatum = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.2,
        epsilon_decay=0.995,
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
    enable_gating=False  # Keep all regions active
)

# Train on 5x5
successes_5x5 = []
for episode in range(100):
    obs = env_5x5.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(obs, prev_reward, False, activations)
        action = result['action']
        
        obs, reward, done, info = env_5x5.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    successes_5x5.append(1 if reward > 0 else 0)

success_rate_5x5 = np.mean(successes_5x5[-20:])
print(f"5x5 Training: {success_rate_5x5:.1%} success (last 20 episodes)")
print()

# ============================================================
# Phase 2: Zero-Shot Transfer to 7x7
# ============================================================
print("="*70)
print("PHASE 2: Zero-Shot Transfer to GridWorld 7x7")
print("="*70)

env_7x7 = GridWorldAdapter(size=7)

# Disable exploration for transfer test
striatum.qlearner.config.epsilon = 0.0

transfer_successes = []
transfer_steps = []

for episode in range(50):
    obs = env_7x7.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 150:
        result = executive.step(obs, prev_reward, False, activations)
        action = result['action']
        
        obs, reward, done, info = env_7x7.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    transfer_successes.append(1 if reward > 0 else 0)
    transfer_steps.append(steps)

transfer_success_rate = np.mean(transfer_successes)
transfer_avg_steps = np.mean(transfer_steps)

print(f"Zero-shot 7x7: {transfer_success_rate:.1%} success ({sum(transfer_successes)}/50)")
print(f"Average steps: {transfer_avg_steps:.0f}")
print()

# ============================================================
# Phase 3: Train from Scratch on 7x7 (Baseline)
# ============================================================
print("="*70)
print("PHASE 3: Training from Scratch on 7x7 (Baseline)")
print("="*70)

# Create fresh agent
striatum_scratch = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.2,
        epsilon_decay=0.995,
    )
)

executive_scratch = ExecutiveController(
    regions={'striatum': striatum_scratch},
    enable_gating=False
)

# Train on 7x7 from scratch
scratch_successes = []
for episode in range(100):
    obs = env_7x7.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 150:
        result = executive_scratch.step(obs, prev_reward, False, np.zeros(100))
        action = result['action']
        
        obs, reward, done, info = env_7x7.step(action)
        prev_reward = reward
        steps += 1
    
    executive_scratch.step(obs, prev_reward, True, np.zeros(100))
    executive_scratch.reset()
    
    scratch_successes.append(1 if reward > 0 else 0)

scratch_success_rate = np.mean(scratch_successes[-20:])
print(f"From-scratch 7x7: {scratch_success_rate:.1%} success (last 20 episodes)")
print()

# ============================================================
# RESULTS
# ============================================================
print("="*70)
print("TRANSFER LEARNING RESULTS")
print("="*70)
print()

print("Performance Comparison:")
print(f"  5x5 (trained):        {success_rate_5x5:.1%}")
print(f"  7x7 (zero-shot):      {transfer_success_rate:.1%}")
print(f"  7x7 (from scratch):   {scratch_success_rate:.1%}")
print()

# Calculate transfer benefit
if scratch_success_rate > 0:
    transfer_benefit = (transfer_success_rate / scratch_success_rate - 1) * 100
    print(f"Transfer Benefit: {transfer_benefit:+.0f}%")
    print()

if transfer_success_rate >= scratch_success_rate * 0.8:
    print("✓ GENERALIZATION PROVEN!")
    print("  Architecture transfers knowledge to new task size")
    print("  Zero-shot performance comparable to trained baseline")
elif transfer_success_rate >= scratch_success_rate * 0.5:
    print("✓ PARTIAL TRANSFER")
    print("  Some knowledge transfers, architecture shows generalization")
elif transfer_success_rate > 0.1:
    print("⚠ WEAK TRANSFER")
    print("  Minimal transfer, but better than random")
else:
    print("✗ NO TRANSFER")
    print("  Architecture may be too task-specific")

print("="*70)
