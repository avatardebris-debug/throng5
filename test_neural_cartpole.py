"""
CartPole with Tuned Neural Q-Learning

Improvements:
- More training episodes (500)
- Higher initial epsilon (more exploration)
- Target network stabilization
- Better reward shaping
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.learning.neural_qlearning import NeuralQLearner, NeuralQLearningConfig
from throng3.environments import CartPoleAdapter

print("="*70)
print("CartPole with Tuned Neural Q-Learning")
print("="*70)

env = CartPoleAdapter()

qlearner = NeuralQLearner(
    n_states=4,
    n_actions=2,
    config=NeuralQLearningConfig(
        learning_rate=0.0005,  # Lower lr for stability
        gamma=0.99,
        epsilon=0.9,  # Higher initial exploration
        epsilon_decay=0.997,  # Slower decay
        epsilon_min=0.01,
        hidden_dims=(128, 128)  # Bigger network
    )
)

stats = qlearner.get_stats()
print(f"Network: {stats['n_params']} parameters (128x128)")
print()

# Training
print("Training (500 episodes)...")
episode_lengths = []
best_avg = 0

for episode in range(500):
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = qlearner.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        
        # Better reward shaping
        if done and steps < 499:
            reward = -1  # Smaller penalty
        else:
            reward = 0.1  # Small reward for staying upright
        
        qlearner.update(obs, action, reward, next_obs, done)
        
        obs = next_obs
        steps += 1
    
    qlearner.end_episode()
    episode_lengths.append(steps)
    
    if (episode + 1) % 100 == 0:
        recent = np.mean(episode_lengths[-100:])
        best_avg = max(best_avg, recent)
        stats = qlearner.get_stats()
        print(f"Episode {episode+1}: avg={recent:.0f}, best={best_avg:.0f}, ε={stats['epsilon']:.3f}")

print()
print("Evaluation (greedy, 50 episodes)...")
qlearner.epsilon = 0.0

eval_lengths = []
for _ in range(50):
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = qlearner.select_action(obs)
        obs, reward, done, info = env.step(action)
        steps += 1
    
    eval_lengths.append(steps)

avg = np.mean(eval_lengths)
median = np.median(eval_lengths)
max_length = max(eval_lengths)

print(f"Average: {avg:.0f} steps")
print(f"Median: {median:.0f} steps")
print(f"Max: {max_length} steps")
print()

if avg >= 300:
    print("✓ SUCCESS! Neural network solves CartPole!")
    print("  → Architecture validated with proper function approximation")
    print("  → Linear Q-learning was the bottleneck")
elif avg >= 150:
    print("✓ GOOD PROGRESS! Network is learning")
    print(f"  {avg:.0f} steps (vs 9 with linear, 20 random)")
    print("  → Proves non-linear approximation helps")
elif avg >= 50:
    print("✓ IMPROVEMENT! Better than linear")
    print(f"  {avg:.0f} steps vs 9 with linear")
else:
    print("⚠ Needs more work")
    
print("="*70)
