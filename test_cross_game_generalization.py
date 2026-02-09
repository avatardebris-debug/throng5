"""
Cross-Game Generalization Test

Tests if Throng3.5 generalizes across different game mechanics:
1. Train on standard GridWorld
2. Zero-shot test on:
   - GridWorld with obstacles
   - Stochastic GridWorld
   - Sparse reward GridWorld

Proves architecture handles different game mechanics!
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
from throng3.environments.gridworld_variants import (
    GridWorldWithObstacles,
    StochasticGridWorld,
    SparseRewardGridWorld
)

print("="*70)
print("CROSS-GAME GENERALIZATION TEST")
print("="*70)
print("Testing Throng3.5 across different game mechanics")
print()

# ============================================================
# Phase 1: Train on Standard GridWorld
# ============================================================
print("="*70)
print("PHASE 1: Training on Standard GridWorld")
print("="*70)

env_standard = GridWorldAdapter(size=5)

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
    enable_gating=False
)

# Train
successes = []
for episode in range(100):
    obs = env_standard.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(obs, prev_reward, False, activations)
        action = result['action']
        
        obs, reward, done, info = env_standard.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    successes.append(1 if reward > 0 else 0)

standard_success = np.mean(successes[-20:])
print(f"Standard GridWorld: {standard_success:.1%} success")
print()

# ============================================================
# Phase 2: Zero-Shot Transfer to Game Variants
# ============================================================

# Disable exploration for transfer tests
striatum.qlearner.config.epsilon = 0.0

game_variants = [
    ("Obstacles", GridWorldWithObstacles(size=5)),
    ("Stochastic (30%)", StochasticGridWorld(size=5, stochastic_prob=0.3)),
    ("Sparse Rewards", SparseRewardGridWorld(size=5)),
]

results = []

for game_name, env in game_variants:
    print("="*70)
    print(f"ZERO-SHOT: {game_name}")
    print("="*70)
    
    successes = []
    steps_list = []
    
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
        
        successes.append(1 if reward > 0 else 0)
        steps_list.append(steps)
    
    success_rate = np.mean(successes)
    avg_steps = np.mean(steps_list)
    
    results.append({
        'name': game_name,
        'success': success_rate,
        'steps': avg_steps
    })
    
    print(f"Success: {success_rate:.1%} ({sum(successes)}/50)")
    print(f"Avg steps: {avg_steps:.0f}")
    print()

# ============================================================
# RESULTS
# ============================================================
print("="*70)
print("CROSS-GAME GENERALIZATION RESULTS")
print("="*70)
print()

print("Performance Across Games:")
print(f"  Standard GridWorld (trained):  {standard_success:.1%}")
for result in results:
    print(f"  {result['name']:25s}  {result['success']:5.1%} ({result['steps']:4.0f} steps)")
print()

# Calculate average transfer performance
avg_transfer = np.mean([r['success'] for r in results])
print(f"Average Zero-Shot Transfer: {avg_transfer:.1%}")
print()

if avg_transfer >= 0.70:
    print("✓ EXCELLENT CROSS-GAME GENERALIZATION!")
    print("  Architecture handles different game mechanics")
    print("  Transfers to obstacles, stochasticity, and sparse rewards")
elif avg_transfer >= 0.50:
    print("✓ GOOD CROSS-GAME TRANSFER")
    print("  Architecture shows generalization across game types")
elif avg_transfer >= 0.30:
    print("✓ PARTIAL TRANSFER")
    print("  Some generalization, architecture adapts to new mechanics")
else:
    print("⚠ LIMITED TRANSFER")
    print("  Architecture may need game-specific tuning")

print()
print("Key Insight:")
if avg_transfer >= 0.50:
    print("  Regional architecture enables zero-shot transfer")
    print("  to different game mechanics without retraining!")
else:
    print("  Transfer varies by game type")
    print("  Some mechanics may need additional training")

print("="*70)
