"""
Test Throng3.5 on FrozenLake WITHOUT Gating

Goal: Prove the regional architecture generalizes to stochastic/sparse rewards.
The previous 0% was due to aggressive gating, not architecture failure.
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
from throng3.environments import FrozenLakeAdapter

print("="*70)
print("FrozenLake Validation: Architecture Generalization Test")
print("="*70)
print("Testing Throng3.5 WITHOUT gating (prove core architecture works)")
print()

# ============================================================
# Create environment and regions
# ============================================================
env = FrozenLakeAdapter(is_slippery=True)

striatum = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.8,  # High for sparse rewards
        gamma=0.99,
        epsilon=0.9,        # High exploration for stochastic
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

# Executive WITHOUT gating
executive = ExecutiveController(
    regions={
        'striatum': striatum,
        'cortex': cortex,
        'hippocampus': hippocampus
    },
    enable_gating=False  # KEY: No gating for sparse rewards
)

print("Configuration:")
print("  Environment: FrozenLake (stochastic, sparse)")
print("  Gating: DISABLED (all regions always active)")
print("  Q-learning: lr=0.8, ε=0.9 (tuned for sparse rewards)")
print()

# ============================================================
# Training
# ============================================================
print("="*70)
print("TRAINING (1000 episodes)")
print("="*70)

successes = []
episode_returns = []

for episode in range(1000):
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
    
    episode_returns.append(env.episode_reward)
    successes.append(1 if info.get('success', False) else 0)
    
    if (episode + 1) % 200 == 0:
        recent_success = np.mean(successes[-200:])
        print(f"  Episode {episode+1}: success={recent_success:.1%}")

print()

# ============================================================
# Evaluation
# ============================================================
print("="*70)
print("EVALUATION (100 episodes, greedy)")
print("="*70)

striatum.qlearner.config.epsilon = 0.0

eval_successes = []
for episode in range(100):
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
    
    eval_successes.append(1 if info.get('success', False) else 0)

eval_success_rate = np.mean(eval_successes)

print(f"Evaluation: {eval_success_rate:.1%} ({sum(eval_successes)}/100)")
print()

# ============================================================
# VALIDATION
# ============================================================
print("="*70)
print("GENERALIZATION PROOF")
print("="*70)
print()

print("Task Comparison:")
print(f"  GridWorld (deterministic, dense):  96.8%")
print(f"  FrozenLake (stochastic, sparse):   {eval_success_rate:.1%}")
print()

# FrozenLake baselines:
# - Random: ~1-2%
# - Good Q-learning: 70-80%
# - Excellent: 85%+

if eval_success_rate >= 0.70:
    print("✓ GENERALIZATION PROVEN!")
    print(f"  Architecture works on stochastic + sparse rewards")
    print(f"  Success: {eval_success_rate:.1%} (target: ≥70%)")
    print()
    print("Key Finding:")
    print("  - Core architecture generalizes ✓")
    print("  - Gating needs environment-specific tuning")
    print("  - Ready for Meta^2 Modulator (learned gating)")
elif eval_success_rate >= 0.50:
    print("✓ GOOD! Architecture learns stochastic tasks")
    print(f"  Success: {eval_success_rate:.1%}")
    print("  May benefit from more training")
elif eval_success_rate >= 0.20:
    print("⚠ PARTIAL - Better than random, needs tuning")
    print(f"  Success: {eval_success_rate:.1%}")
else:
    print("✗ Architecture may need adjustments for sparse rewards")
    print(f"  Success: {eval_success_rate:.1%}")

print("="*70)
