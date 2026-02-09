"""
CartPole Test: Prove Architecture Works on Continuous Control

CartPole should work because:
- Continuous state (4D: cart_pos, cart_vel, pole_angle, pole_vel)
- Dense rewards (1 per step)
- Deterministic

If this works, proves architecture is sound - FrozenLake issue is sparse+discrete.
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.core.features import RBFFeatures
from throng3.environments import CartPoleAdapter

print("="*70)
print("CartPole Test: Continuous Control with RBF")
print("="*70)

env = CartPoleAdapter()

# RBF for CartPole's 4D state
# State: [cart_pos, cart_vel, pole_angle, pole_vel]
rbf = RBFFeatures(
    n_centers_per_dim=4,  # 4^4 = 256 centers
    state_bounds=[
        (-4.8, 4.8),    # cart position
        (-5.0, 5.0),    # cart velocity
        (-0.42, 0.42),  # pole angle
        (-5.0, 5.0)     # pole velocity
    ],
    normalize=True,
    add_bias=True
)

print(f"RBF: {rbf.get_n_features()} features (256 centers + bias)")
print()

striatum = StriatumRegion(
    n_states=rbf.get_n_features(),
    n_actions=2,  # left, right
    config=QLearningConfig(
        learning_rate=0.1,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.995,
    )
)

executive = ExecutiveController(
    regions={'striatum': striatum},
    enable_gating=False
)

# Training
print("Training (200 episodes)...")
episode_lengths = []

for episode in range(200):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 500:
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
    
    if (episode + 1) % 50 == 0:
        recent = np.mean(episode_lengths[-50:])
        print(f"Episode {episode+1}: avg_length={recent:.0f}")

print()
print("Evaluation (greedy, 20 episodes)...")
striatum.qlearner.config.epsilon = 0.0

eval_lengths = []
for _ in range(20):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    
    while not done and steps < 500:
        features = rbf.transform(obs)
        result = executive.step(features, prev_reward, False, np.zeros(100))
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        steps += 1
    
    eval_lengths.append(steps)

avg = np.mean(eval_lengths)
print(f"Average length: {avg:.0f}")
print()

# CartPole baseline: random ~20-30, good agent >200, excellent >400
if avg >= 200:
    print("✓ EXCELLENT! Architecture works on continuous control!")
    print(f"  CartPole: {avg:.0f} steps")
    print("  Conclusion: FrozenLake issue is sparse+discrete, not architecture")
elif avg >= 100:
    print("✓ GOOD! Learning continuous control")
    print(f"  Performance: {avg:.0f} steps")
elif avg >= 50:
    print("⚠ LEARNING - Better than random")
    print(f"  Performance: {avg:.0f} steps")
else:
    print("⚠ Needs more training or tuning")
    print(f"  Performance: {avg:.0f} steps")

print("="*70)
