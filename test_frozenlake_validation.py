"""
Test Throng3.5 on FrozenLake (Stochastic Environment)

Validates that the regional architecture handles:
- Stochastic dynamics (slippery ice)
- Sparse rewards (only at goal)
- Different state space (4x4 vs 5x5)
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
print("Multi-Task Validation: FrozenLake (Stochastic)")
print("="*70)
print("Testing Throng3.5 on stochastic environment with sparse rewards")
print()

# ============================================================
# Create FrozenLake environment
# ============================================================
env = FrozenLakeAdapter(is_slippery=True)

print("Environment: FrozenLake-v1")
print("  Grid: 4x4 with holes")
print("  Stochastic: 33% chance of sideways movement")
print("  Reward: +1 at goal only (sparse)")
print()

# ============================================================
# Create 3-region system
# ============================================================
striatum = StriatumRegion(
    n_states=2,  # Normalized (x, y)
    n_actions=4,  # left, down, right, up
    config=QLearningConfig(
        learning_rate=0.5,  # Higher for sparse rewards
        gamma=0.99,         # Long-term planning
        epsilon=0.5,        # More exploration for stochastic
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
    enable_gating=True,
    router_config={
        'cortex_td_threshold': 0.1,  # Higher for sparse rewards
        'hippocampus_sequence_threshold': 3,
        'warmup_steps': 200  # More warmup for stochastic
    }
)

print("3-Region System Created:")
print("  Striatum: Q-learning (lr=0.5, ε=0.5)")
print("  Cortex: Hebbian")
print("  Hippocampus: STDP")
print("  Executive: Adaptive gating enabled")
print()

# ============================================================
# Training
# ============================================================
print("="*70)
print("TRAINING (500 episodes)")
print("="*70)

successes = []
episode_returns = []

for episode in range(500):
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
    
    # Final update
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    episode_returns.append(env.episode_reward)
    successes.append(1 if info.get('success', False) else 0)
    
    # Progress updates
    if (episode + 1) % 100 == 0:
        recent_success = np.mean(successes[-100:])
        recent_return = np.mean(episode_returns[-100:])
        print(f"  Episode {episode+1}: success={recent_success:.1%}, avg_return={recent_return:.3f}")

print()

# ============================================================
# Evaluation
# ============================================================
print("="*70)
print("EVALUATION (100 episodes, greedy)")
print("="*70)

# Disable exploration for evaluation
striatum.qlearner.config.epsilon = 0.0

eval_successes = []
eval_returns = []

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
    
    eval_returns.append(env.episode_reward)
    eval_successes.append(1 if info.get('success', False) else 0)

eval_success_rate = np.mean(eval_successes)
eval_avg_return = np.mean(eval_returns)

print(f"Evaluation Results:")
print(f"  Success rate: {eval_success_rate:.1%} ({sum(eval_successes)}/100)")
print(f"  Average return: {eval_avg_return:.3f}")
print()

# ============================================================
# Statistics
# ============================================================
stats = executive.get_stats()

print("="*70)
print("SYSTEM STATISTICS")
print("="*70)
print()

print("Regional Performance:")
print(f"  Striatum:")
print(f"    Epsilon: {stats['striatum']['epsilon']:.4f}")
print(f"    Updates: {stats['striatum']['n_updates']}")
print(f"    Compute: {stats['striatum']['resource_usage']['compute_ms']:.2f}ms/step")
print(f"  Cortex:")
print(f"    Pattern strength: {stats['cortex']['avg_pattern_strength']:.3f}")
print(f"    Compute: {stats['cortex']['resource_usage']['compute_ms']:.2f}ms/step")
print(f"  Hippocampus:")
print(f"    Episodes stored: {stats['hippocampus'].get('n_episodes', 'N/A')}")
print(f"    Compute: {stats['hippocampus']['resource_usage']['compute_ms']:.2f}ms/step")
print()

if 'router' in stats:
    print("Adaptive Gating:")
    print(f"  Cortex activation: {stats['router']['activation_rates']['cortex']:.1%}")
    print(f"  Hippocampus activation: {stats['router']['activation_rates']['hippocampus']:.1%}")
    print(f"  Cortex gating: {stats['router']['cortex_gating_rate']:.1%}")
    print(f"  Hippocampus gating: {stats['router']['hippocampus_gating_rate']:.1%}")
print()

# ============================================================
# VALIDATION
# ============================================================
print("="*70)
print("VALIDATION")
print("="*70)
print()

print("Environment Comparison:")
print(f"  GridWorld (deterministic):  96.8% success")
print(f"  FrozenLake (stochastic):    {eval_success_rate:.1%} success")
print()

# FrozenLake baseline (random policy) is ~1-2%
# Good Q-learning typically gets 70-80% on slippery FrozenLake
if eval_success_rate >= 0.70:
    print("✓ EXCELLENT! Throng3.5 handles stochastic environments!")
    print(f"  Success rate: {eval_success_rate:.1%}")
    print("  Architecture generalizes to different dynamics")
elif eval_success_rate >= 0.50:
    print("✓ GOOD! Reasonable performance on stochastic task")
    print(f"  Success rate: {eval_success_rate:.1%}")
    print("  May benefit from more training or tuning")
elif eval_success_rate >= 0.20:
    print("⚠ LEARNING! Better than random, needs more training")
    print(f"  Success rate: {eval_success_rate:.1%}")
    print("  Stochastic environments are challenging")
else:
    print("✗ NEEDS WORK - performance near random")
    print(f"  Success rate: {eval_success_rate:.1%}")
    print("  May need architecture adjustments for stochastic tasks")

print("="*70)
