"""
Simple FrozenLake Test: Prove Q-Learning Works

Minimal test to validate Q-learning can solve FrozenLake.
Uses discrete state representation (16 states).
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
import gymnasium as gym

print("="*70)
print("Baseline: Can Q-Learning Solve FrozenLake?")
print("="*70)

# Simple Q-learning
env = gym.make('FrozenLake-v1', is_slippery=True)
Q = np.zeros((16, 4))  # 16 states, 4 actions

# Hyperparameters
lr = 0.8
gamma = 0.99
epsilon = 0.9
epsilon_decay = 0.995

successes = []

for episode in range(1000):
    state, _ = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Q-learning update
        Q[state, action] += lr * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        state = next_state
    
    epsilon *= epsilon_decay
    successes.append(1 if reward > 0 else 0)
    
    if (episode + 1) % 200 == 0:
        print(f"Episode {episode+1}: {np.mean(successes[-200:]):.1%}")

# Evaluate
epsilon = 0
eval_successes = []
for _ in range(100):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    eval_successes.append(1 if reward > 0 else 0)

print(f"\nBaseline Q-learning: {np.mean(eval_successes):.1%}")
print("="*70)
