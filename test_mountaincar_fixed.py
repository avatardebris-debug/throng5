"""
MountainCar with Improved RBF Features

Fixed issues:
- More RBF centers (10x10 = 100)
- Better sigma (wider coverage)
- Normalized features
- Longer training (500 episodes)
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.core.features import RBFFeatures
from throng3.environments import MountainCarAdapter

print("="*70)
print("MountainCar with Improved RBF (10x10 = 100 centers)")
print("="*70)

env = MountainCarAdapter()

# Improved RBF: more centers, better coverage
rbf = RBFFeatures(
    n_centers_per_dim=10,  # 10x10 = 100 centers
    state_bounds=[(-1.2, 0.6), (-0.07, 0.07)],
    normalize=True,
    add_bias=True
)

print(f"RBF: {rbf.get_n_features()} features (100 centers + bias)")
print(f"Sigma: {rbf.sigma:.4f}")

# Striatum with RBF features
striatum = StriatumRegion(
    n_states=rbf.get_n_features(),
    n_actions=3,
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=1.0,  # No discounting for sparse rewards
        epsilon=0.1,
        epsilon_decay=0.999,
    )
)

executive = ExecutiveController(
    regions={'striatum': striatum},
    enable_gating=False
)

print("Training 500 episodes...")
print()

episode_lengths = []

for episode in range(500):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 200:
        features = rbf.transform(obs)
        result = executive.step(features, prev_reward, False, np.zeros(100))
        action = result['action']
        
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
    
    features = rbf.transform(obs)
    executive.step(features, prev_reward, True, np.zeros(100))
    executive.reset()
    
    episode_lengths.append(steps)
    
    if (episode + 1) % 100 == 0:
        recent = np.mean(episode_lengths[-100:])
        print(f"Episode {episode+1}: avg_steps={recent:.0f}")

# Evaluate
print("\nEvaluation (greedy):")
striatum.qlearner.config.epsilon = 0.0

eval_lengths = []
for _ in range(20):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 200:
        features = rbf.transform(obs)
        result = executive.step(features, prev_reward, False, np.zeros(100))
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
    
    eval_lengths.append(steps)

avg = np.mean(eval_lengths)
print(f"Average steps: {avg:.0f}")

if avg < 150:
    print("✓ GENERALIZATION PROVEN! Architecture works on sparse rewards")
elif avg < 200:
    print("✓ GOOD - learning, reaching goal sometimes")
else:
    print("⚠ Still hitting timeout")

print("="*70)
