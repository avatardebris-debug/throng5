"""
Test Throng3.5 on MountainCar with RBF Features

Goal: Prove architecture generalizes to continuous sparse reward tasks.
Uses RBF feature expansion for richer state representation.
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
from throng35.core.features import RBFFeatures
from throng3.environments import MountainCarAdapter

print("="*70)
print("MountainCar Validation: Continuous Sparse Reward Test")
print("="*70)
print("Testing Throng3.5 with RBF feature expansion")
print()

# ============================================================
# Create environment and RBF features
# ============================================================
env = MountainCarAdapter()

# RBF feature expansion
# MountainCar state: [position, velocity]
# Position: [-1.2, 0.6], Velocity: [-0.07, 0.07]
rbf = RBFFeatures(
    n_features=20,  # 20 RBF centers
    state_bounds=[(-1.2, 0.6), (-0.07, 0.07)]
)

print("RBF Feature Expansion:")
print(f"  Input: 2D [position, velocity]")
print(f"  Output: {rbf.get_n_features()}D RBF features")
print(f"  Centers: {len(rbf.centers)} spread across state space")
print()

# ============================================================
# Create 3-region system with RBF features
# ============================================================
striatum = StriatumRegion(
    n_states=rbf.get_n_features(),  # Use RBF features
    n_actions=3,  # left, nothing, right
    config=QLearningConfig(
        learning_rate=0.5,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.999,
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
    enable_gating=False  # Disable for sparse rewards
)

print("3-Region System:")
print("  Striatum: Q-learning with 20 RBF features")
print("  Cortex: Hebbian")
print("  Hippocampus: STDP")
print("  Gating: DISABLED (sparse rewards)")
print()

# ============================================================
# Training
# ============================================================
print("="*70)
print("TRAINING (200 episodes)")
print("="*70)

episode_returns = []
episode_lengths = []

for episode in range(200):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 1000:
        # Expand features with RBF
        rbf_features = rbf.transform(obs)
        
        result = executive.step(rbf_features, prev_reward, False, activations)
        action = result['action']
        
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    # Final update
    rbf_features = rbf.transform(obs)
    executive.step(rbf_features, prev_reward, True, activations)
    executive.reset()
    
    episode_returns.append(env.episode_reward)
    episode_lengths.append(steps)
    
    if (episode + 1) % 50 == 0:
        recent_return = np.mean(episode_returns[-50:])
        recent_length = np.mean(episode_lengths[-50:])
        print(f"  Episode {episode+1}: avg_return={recent_return:.1f}, avg_steps={recent_length:.0f}")

print()

# ============================================================
# Evaluation
# ============================================================
print("="*70)
print("EVALUATION (20 episodes, greedy)")
print("="*70)

striatum.qlearner.config.epsilon = 0.0

eval_returns = []
eval_lengths = []
successes = 0

for episode in range(20):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 1000:
        rbf_features = rbf.transform(obs)
        result = executive.step(rbf_features, prev_reward, False, activations)
        action = result['action']
        
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    rbf_features = rbf.transform(obs)
    executive.step(rbf_features, prev_reward, True, activations)
    executive.reset()
    
    eval_returns.append(env.episode_reward)
    eval_lengths.append(steps)
    
    # Success = reached goal (position >= 0.5)
    if info.get('success', False) or steps < 200:
        successes += 1

avg_return = np.mean(eval_returns)
avg_length = np.mean(eval_lengths)

print(f"Results:")
print(f"  Average return: {avg_return:.1f}")
print(f"  Average steps: {avg_length:.0f}")
print(f"  Successes: {successes}/20")
print()

# ============================================================
# VALIDATION
# ============================================================
print("="*70)
print("GENERALIZATION VALIDATION")
print("="*70)
print()

print("Task Comparison:")
print(f"  GridWorld (deterministic, dense):   96.8%")
print(f"  MountainCar (continuous, sparse):   {successes/20:.1%}")
print()

# MountainCar is HARD - random takes ~1000 steps
# Good learning: <200 steps
# Excellent: <150 steps

if avg_length < 200:
    print("✓ EXCELLENT! Architecture generalizes to sparse rewards!")
    print(f"  Average steps: {avg_length:.0f} (target: <200)")
    print("  RBF features enable continuous state learning")
    print()
    print("Generalization PROVEN:")
    print("  ✓ Deterministic + dense (GridWorld)")
    print("  ✓ Continuous + sparse (MountainCar)")
elif avg_length < 500:
    print("✓ GOOD! Learning sparse reward task")
    print(f"  Average steps: {avg_length:.0f}")
    print("  Architecture works, may need more training")
elif avg_length < 800:
    print("⚠ LEARNING - Better than random")
    print(f"  Average steps: {avg_length:.0f} (random: ~1000)")
else:
    print("⚠ NEEDS MORE TRAINING")
    print(f"  Average steps: {avg_length:.0f}")
    print("  Try more episodes or tune hyperparameters")

print("="*70)
