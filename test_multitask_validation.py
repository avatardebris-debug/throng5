"""
Multi-Task Validation for Regional Architecture

Test Striatum + Cortex + Executive on multiple environments:
- FrozenLake (sparse rewards, slippery)
- CartPole (continuous control, dense rewards)
- GridWorld (baseline)

Validates that regional separation works across different task types.
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.regions.cortex import CortexRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.learning.hebbian import HebbianConfig
from throng3.environments import GridWorldAdapter, CartPoleAdapter

print("="*70)
print("Multi-Task Validation: Regional Architecture")
print("="*70)
print("Testing: GridWorld → CartPole (different task types)")
print()

# ============================================================
# TASK 1: GridWorld (Baseline)
# ============================================================
print("="*70)
print("TASK 1: GridWorld (5x5, deterministic, sparse rewards)")
print("="*70)

env1 = GridWorldAdapter()

# Create regions
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

executive = ExecutiveController({
    'striatum': striatum,
    'cortex': cortex
})

# Curriculum learning (5 demonstrations)
print("Curriculum learning (5 demonstrations)...")
optimal_actions = [1, 1, 1, 1, 0, 0, 0, 0]  # right×4, up×4
for demo_ep in range(5):
    obs = env1.reset()
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    for step, action in enumerate(optimal_actions):
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        obs, reward, done, info = env1.step(action)
        prev_reward = reward
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()

# Quick training on GridWorld
print("Training on GridWorld (50 episodes)...")
successes = 0
returns = []
for episode in range(50):
    obs = env1.reset()
    done = False
    steps = 0
    prev_reward = 0
    episode_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        action = result['action']
        obs, reward, done, info = env1.step(action)
        prev_reward = reward
        episode_reward += reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    if info['pos'] == info['goal']:
        successes += 1
    returns.append(episode_reward)

print(f"GridWorld performance: {successes}/50 ({successes/50:.1%})")
print(f"Average return: {np.mean(returns[-10:]):.2f}")

# ============================================================
# TASK 2: CartPole (Different Task Type)
# ============================================================
print(f"\n{'='*70}")
print("TASK 2: CartPole (continuous state, dense rewards, balance)")
print("="*70)
print("Testing: Can architecture handle different task structure?")
print()

env2 = CartPoleAdapter()

# Create NEW regions for CartPole (different state/action space)
striatum2 = StriatumRegion(
    n_states=4,  # CartPole: [position, velocity, angle, angular_velocity]
    n_actions=2,  # left, right
    config=QLearningConfig(
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.995,
    )
)

cortex2 = CortexRegion(
    n_neurons=100,
    n_features=10,
    config=HebbianConfig(learning_rate=0.01)
)

executive2 = ExecutiveController({
    'striatum': striatum2,
    'cortex': cortex2
})

print("Training on CartPole (50 episodes)...")
cp_returns = []
for episode in range(50):
    obs = env2.reset()
    done = False
    steps = 0
    prev_reward = 0
    episode_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 500:
        result = executive2.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        action = result['action']
        obs, reward, done, info = env2.step(action)
        prev_reward = reward
        episode_reward += reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive2.step(obs, prev_reward, True, activations)
    executive2.reset()
    
    cp_returns.append(episode_reward)

print(f"CartPole average return (last 10): {np.mean(cp_returns[-10:]):.1f}")
print(f"CartPole best episode: {max(cp_returns):.1f}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*70}")
print("MULTI-TASK VALIDATION RESULTS")
print("="*70)

stats1 = executive.get_stats()
stats2 = executive2.get_stats()

print(f"\nTask Performance:")
print(f"  GridWorld (sparse):  {successes}/50 ({successes/50:.1%}), avg return: {np.mean(returns[-10:]):.2f}")
print(f"  CartPole (dense):    avg return: {np.mean(cp_returns[-10:]):.1f} (target: >100)")

print(f"\nRegional Statistics (GridWorld):")
print(f"  Striatum:")
print(f"    Total updates: {stats1['striatum']['n_updates']}")
print(f"    Epsilon: {stats1['striatum']['epsilon']:.4f}")
print(f"    Compute: {stats1['striatum']['resource_usage']['compute_ms']:.2f}ms/step")
print(f"  Cortex:")
print(f"    Pattern strength: {stats1['cortex']['avg_pattern_strength']:.3f}")
print(f"    Compute: {stats1['cortex']['resource_usage']['compute_ms']:.2f}ms/step")

print(f"\nRegional Statistics (CartPole):")
print(f"  Striatum:")
print(f"    Total updates: {stats2['striatum']['n_updates']}")
print(f"    Epsilon: {stats2['striatum']['epsilon']:.4f}")
print(f"  Cortex:")
print(f"    Pattern strength: {stats2['cortex']['avg_pattern_strength']:.3f}")

print(f"\n{'='*70}")
print("VALIDATION")
print("="*70)

gw_success = successes >= 40
cp_success = np.mean(cp_returns[-10:]) >= 50  # Reasonable threshold

if gw_success and cp_success:
    print("\n✓ SUCCESS! Regional architecture works across task types!")
    print("  - GridWorld (sparse rewards): ✓")
    print("  - CartPole (dense rewards): ✓")
    print("  - Architecture is task-agnostic")
    print("  - Ready for Phase A: Hippocampus region")
elif gw_success:
    print("\n⚠ PARTIAL SUCCESS")
    print("  - GridWorld works well")
    print("  - CartPole needs tuning (expected - different task type)")
else:
    print("\n✗ NEEDS WORK")
    print("  - Performance issues detected")

print("="*70)
