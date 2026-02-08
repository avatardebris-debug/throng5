"""
Standalone Q-learning test (no Meta^N integration)

Test Q-learning directly on GridWorld to verify it works.
"""

import numpy as np
from throng3.environments import GridWorldAdapter
from throng3.learning.qlearning import QLearner, QLearningConfig

print("="*60)
print("Standalone Q-Learning Test")
print("="*60)
print("Testing Q-learning directly on GridWorld (no Meta^N)")
print()

env = GridWorldAdapter()

# Create Q-learner
# State = 2D position, Actions = 4 directions
qlearner = QLearner(
    n_states=2,  # [x, y]
    n_actions=4,  # [up, down, left, right]
    config=QLearningConfig(
        learning_rate=0.1,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

episode_returns = []
episode_lengths = []

print("Training for 100 episodes...")

for episode in range(100):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    prev_obs = None
    prev_action = None
    
    while not done and steps < 200:
        # Select action using Q-learning
        action = qlearner.select_action(obs, explore=True)
        
        # Take action
        next_obs, reward, done, info = env.step(action)
        
        # Q-learning update
        if prev_obs is not None and prev_action is not None:
            qlearner.update(prev_obs, prev_action, reward, obs, done)
        
        prev_obs = obs.copy()
        prev_action = action
        obs = next_obs
        
        episode_reward += reward
        steps += 1
    
    # Final update
    if prev_obs is not None and prev_action is not None:
        qlearner.update(prev_obs, prev_action, reward, obs, done=True)
    
    qlearner.reset_episode()
    
    episode_returns.append(episode_reward)
    episode_lengths.append(steps)
    
    if (episode + 1) % 10 == 0:
        recent_returns = episode_returns[-10:]
        recent_lengths = episode_lengths[-10:]
        print(f"  Episode {episode+1}: "
              f"return={episode_reward:.3f}, "
              f"avg_return={np.mean(recent_returns):.3f}, "
              f"avg_length={np.mean(recent_lengths):.1f}")

# Analysis
early_returns = np.mean(episode_returns[:10])
late_returns = np.mean(episode_returns[-10:])
improvement = late_returns - early_returns

print(f"\n{'='*60}")
print("Results:")
print(f"{'='*60}")
print(f"Early episodes (1-10):  {early_returns:.3f}")
print(f"Late episodes (91-100): {late_returns:.3f}")
print(f"Improvement:            {improvement:+.3f}")

stats = qlearner.get_stats()
print(f"\nQ-learning stats:")
print(f"  Epsilon: {stats['epsilon']:.4f}")
print(f"  Updates: {stats['n_updates']}")
print(f"  Avg TD error: {stats['avg_td_error']:.4f}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")
print(f"  Max Q weight: {stats['max_q_weight']:.6f}")

# Show Q-values for a few states
print(f"\nSample Q-values:")
for x in [0, 2, 4]:
    for y in [0, 2, 4]:
        state = np.array([x, y], dtype=np.float32)
        q_vals = qlearner.get_q_values(state)
        best_action = np.argmax(q_vals)
        actions = ['↑', '↓', '←', '→']
        print(f"  State [{x},{y}]: Q={q_vals}, best={actions[best_action]}")

if improvement > 0.1:
    print(f"\n✓ SUCCESS! Q-learning works!")
elif improvement > 0.01:
    print(f"\n⚠ Slight improvement, may need tuning.")
else:
    print(f"\n✗ No significant learning.")

print("="*60)
