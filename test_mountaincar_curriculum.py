"""
MountainCar with Improved Curriculum (Randomized Starts)

Key fix: Use RANDOM start positions within expanding ranges
- Phase 1: Start in [0.3, 0.5] - near goal
- Phase 2: Start in [0.0, 0.5] - expanding left
- Phase 3: Start in [-0.5, 0.5] - more expansion
- Phase 4: Start in [-1.2, 0.5] - full range

This prevents overfitting to specific start positions!
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
import gymnasium as gym
from throng35.regions.striatum import StriatumRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.core.features import RBFFeatures

print("="*70)
print("MountainCar: Curriculum with Randomized Starts")
print("="*70)
print()

rbf = RBFFeatures(
    n_centers_per_dim=10,
    state_bounds=[(-1.2, 0.6), (-0.07, 0.07)],
    normalize=True,
    add_bias=True
)

striatum = StriatumRegion(
    n_states=rbf.get_n_features(),
    n_actions=3,
    config=QLearningConfig(
        learning_rate=0.5,
        gamma=0.99,
        epsilon=0.3,  # Higher exploration
        epsilon_decay=0.997,
    )
)

executive = ExecutiveController(
    regions={'striatum': striatum},
    enable_gating=False
)

# Improved curriculum with random starts
curriculum = [
    {"name": "Phase 1: Near Goal", "start_range": (0.3, 0.5), "episodes": 100},
    {"name": "Phase 2: Expanding Left", "start_range": (0.0, 0.5), "episodes": 150},
    {"name": "Phase 3: Bottom Valley", "start_range": (-0.5, 0.5), "episodes": 200},
    {"name": "Phase 4: Full Range", "start_range": (-1.2, 0.5), "episodes": 250},
]

env = gym.make('MountainCar-v0')

print("Improved Curriculum (Randomized Starts):")
for i, phase in enumerate(curriculum, 1):
    print(f"  {i}. {phase['name']}: start ∈ [{phase['start_range'][0]}, {phase['start_range'][1]}]")
print()

phase_results = []

for phase in curriculum:
    print("="*70)
    print(f"{phase['name']}")
    print("="*70)
    
    phase_lengths = []
    successes = 0
    
    for episode in range(phase['episodes']):
        obs, _ = env.reset()
        
        # Random start within curriculum range
        start_pos = np.random.uniform(phase['start_range'][0], phase['start_range'][1])
        env.unwrapped.state = np.array([start_pos, 0.0])
        obs = env.unwrapped.state
        
        done = False
        steps = 0
        prev_reward = 0
        
        while not done and steps < 200:
            features = rbf.transform(obs)
            result = executive.step(features, prev_reward, False, np.zeros(100))
            action = result['action']
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            prev_reward = reward
            steps += 1
        
        features = rbf.transform(obs)
        executive.step(features, prev_reward, True, np.zeros(100))
        executive.reset()
        
        phase_lengths.append(steps)
        if obs[0] >= 0.5:
            successes += 1
    
    avg_steps = np.mean(phase_lengths)
    success_rate = successes / phase['episodes']
    phase_results.append({
        'name': phase['name'],
        'avg_steps': avg_steps,
        'success_rate': success_rate
    })
    
    print(f"  Avg steps: {avg_steps:.0f}, Success: {success_rate:.1%} ({successes}/{phase['episodes']})")
    print()

# Final evaluation
print("="*70)
print("FINAL EVALUATION (Standard MountainCar, Greedy)")
print("="*70)

striatum.qlearner.config.epsilon = 0.0

eval_lengths = []
eval_successes = 0

for episode in range(100):
    obs, _ = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 200:
        features = rbf.transform(obs)
        result = executive.step(features, prev_reward, False, np.zeros(100))
        action = result['action']
        
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        prev_reward = reward
        steps += 1
    
    eval_lengths.append(steps)
    if obs[0] >= 0.5:
        eval_successes += 1

final_avg = np.mean(eval_lengths)
final_success = eval_successes / 100

print(f"Average steps: {final_avg:.0f}")
print(f"Success rate: {final_success:.1%} ({eval_successes}/100)")
print()

# Summary
print("="*70)
print("RESULTS")
print("="*70)
print()

print("Curriculum Progression:")
for result in phase_results:
    print(f"  {result['name']:25s} {result['avg_steps']:5.0f} steps  {result['success_rate']:5.1%}")
print()

print(f"Final Performance: {final_avg:.0f} steps, {final_success:.1%} success")
print()

if final_success >= 0.80:
    print("✓ EXCELLENT! Curriculum learning works!")
    print("  Throng3.5 + RBF + Curriculum = Sparse reward generalization ✓")
elif final_success >= 0.50:
    print("✓ GOOD! Curriculum enabled learning")
    print(f"  Success: {final_success:.1%}")
    print("  Architecture generalizes with proper training strategy")
elif final_success >= 0.20:
    print("✓ LEARNING! Better than baseline")
    print(f"  Success: {final_success:.1%}")
    print("  Curriculum helps, may need longer training")
else:
    print("⚠ Needs adjustment")
    print("  Try: longer phases, more gradual progression, or higher lr")

print("="*70)
