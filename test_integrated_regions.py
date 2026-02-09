"""
Test Integrated Regional Architecture

Validate that Striatum + Cortex + Executive work together
and maintain the 100% success rate.
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
from throng3.environments import GridWorldAdapter

print("="*60)
print("Integrated Regional Architecture Test")
print("="*60)
print("Striatum (Q-learning) + Cortex (Hebbian) + Executive")
print("Target: Maintain 100% success with regional coordination")
print()

# Create environment
env = GridWorldAdapter()

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

# Create Executive controller
executive = ExecutiveController({
    'striatum': striatum,
    'cortex': cortex
})

print("Regional architecture created:")
print("  Striatum: Q-learning (2 states, 4 actions)")
print("  Cortex: Hebbian (100 neurons, 10 features)")
print("  Executive: Coordinating both regions")
print()

# ============================================================
# PHASE 1: CURRICULUM (5 demonstrations)
# ============================================================
print("="*60)
print("PHASE 1: Curriculum Learning (5 demonstrations)")
print("="*60)

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]  # right×4, down×4

for demo_ep in range(5):
    obs = env.reset()
    episode_reward = 0
    prev_reward = 0
    
    # Generate dummy neuron activations (in real system, from Meta^0)
    activations = np.random.randn(100) * 0.1
    
    for step, action in enumerate(optimal_actions):
        # Executive coordinates regions
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        # Take optimal action (override for demo)
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        episode_reward += reward
        
        # Update activations (simulate learning)
        activations += np.random.randn(100) * 0.05
        
        if done:
            executive.step(
                raw_observation=obs,
                reward=prev_reward,
                done=True,
                activations=activations
            )
            break
    
    executive.reset()
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}")

stats = executive.get_stats()
print(f"\nAfter demonstrations:")
print(f"  Striatum updates: {stats['striatum']['n_updates']}")
print(f"  Cortex pattern strength: {stats['cortex']['avg_pattern_strength']:.3f}")

# ============================================================
# PHASE 2: EXPLORATION & LEARNING (95 episodes)
# ============================================================
print(f"\n{'='*60}")
print("PHASE 2: Exploration & Learning (95 episodes)")
print("="*60)

episode_returns = []
successes = 0

for episode in range(95):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    prev_reward = 0
    
    # Initialize activations
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        # Executive coordinates regions
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        action = result['action']
        
        # Execute action
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        episode_reward += reward
        steps += 1
        
        # Update activations (simulate learning)
        activations += np.random.randn(100) * 0.05
    
    # Final update
    executive.step(
        raw_observation=obs,
        reward=prev_reward,
        done=True,
        activations=activations
    )
    
    executive.reset()
    
    if info['pos'] == info['goal']:
        successes += 1
    
    episode_returns.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        recent = episode_returns[-10:]
        success_rate = sum(1 for r in recent if r > 0) / 10
        print(f"  Episode {episode+1}: return={episode_reward:.3f}, "
              f"avg={np.mean(recent):.3f}, success={success_rate:.0%}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

early = np.mean(episode_returns[:10])
late = np.mean(episode_returns[-10:])
improvement = late - early

print(f"Early episodes (1-10):  {early:.3f}")
print(f"Late episodes (86-95):  {late:.3f}")
print(f"Improvement:            {improvement:+.3f}")
print(f"Total successes:        {successes}/95 ({successes/95:.1%})")

stats = executive.get_stats()
print(f"\nRegional statistics:")
print(f"  Striatum:")
print(f"    Epsilon: {stats['striatum']['epsilon']:.4f}")
print(f"    Updates: {stats['striatum']['n_updates']}")
print(f"  Cortex:")
print(f"    Avg pattern strength: {stats['cortex']['avg_pattern_strength']:.3f}")
print(f"    Feature weights norm: {stats['cortex']['feature_weights_norm']:.3f}")

# ============================================================
# VALIDATION
# ============================================================
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

print(f"\nBaseline (Striatum alone):     98.9%")
print(f"Integrated (Striatum+Cortex):  {successes/95:.1%}")

if successes / 95 >= 0.95:
    print(f"\n✓ SUCCESS! Regional integration works!")
    print(f"  Striatum + Cortex + Executive validated.")
    print(f"  Ready for more complex tasks.")
elif successes / 95 >= 0.8:
    print(f"\n⚠ GOOD - minor tuning may help")
else:
    print(f"\n✗ DEGRADED - integration causing issues")
    print(f"  Need to debug regional coordination")

print("="*60)
