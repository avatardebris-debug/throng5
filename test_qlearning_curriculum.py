"""
Q-learning with Curriculum Learning (Demonstration Bootstrap)

Start by showing the agent the optimal path a few times,
then let it explore and learn on its own.
"""

import numpy as np
from throng3.environments import GridWorldAdapter
from throng3.learning.qlearning import QLearner, QLearningConfig

print("="*60)
print("Q-Learning with Curriculum Learning")
print("="*60)
print("Phase 1: Demonstrate optimal path")
print("Phase 2: Learn with exploration")
print()

env = GridWorldAdapter()

# Create Q-learner
qlearner = QLearner(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.3,  # Higher LR for faster learning from demos
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

# ============================================================
# PHASE 1: DEMONSTRATION (Curriculum Learning "Cheat")
# ============================================================
print("Phase 1: Demonstrating optimal path (5 episodes)...")
print("  Optimal: right→right→right→right→down→down→down→down")
print()

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]  # right×4, down×4

for demo_ep in range(5):
    obs = env.reset()
    episode_reward = 0
    
    prev_obs = None
    prev_action = None
    
    for step, action in enumerate(optimal_actions):
        # Take optimal action
        next_obs, reward, done, info = env.step(action)
        
        # Q-learning update
        if prev_obs is not None:
            qlearner.update(prev_obs, prev_action, reward, obs, done)
        
        prev_obs = obs.copy()
        prev_action = action
        obs = next_obs
        episode_reward += reward
        
        if done:
            break
    
    # Final update
    if prev_obs is not None:
        qlearner.update(prev_obs, prev_action, reward, obs, done=True)
    
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}, steps={step+1}")

print(f"\nAfter demonstrations:")
stats = qlearner.get_stats()
print(f"  Updates: {stats['n_updates']}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# Show learned Q-values along optimal path
print(f"\nQ-values along optimal path:")
for x in range(5):
    state = np.array([x, 0], dtype=np.float32)
    q_vals = qlearner.get_q_values(state)
    actions = ['↑', '↓', '←', '→']
    best = np.argmax(q_vals)
    print(f"  [{x},0]: best={actions[best]}, Q={q_vals[best]:.3f}")

# ============================================================
# PHASE 2: EXPLORATION & LEARNING
# ============================================================
print(f"\n{'='*60}")
print("Phase 2: Learning with exploration (95 episodes)...")
print()

episode_returns = []
episode_lengths = []
successes = 0

for episode in range(95):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    prev_obs = None
    prev_action = None
    
    while not done and steps < 100:
        # ε-greedy action selection
        action = qlearner.select_action(obs, explore=True)
        
        # Take action
        next_obs, reward, done, info = env.step(action)
        
        # Q-learning update
        if prev_obs is not None:
            qlearner.update(prev_obs, prev_action, reward, obs, done)
        
        prev_obs = obs.copy()
        prev_action = action
        obs = next_obs
        
        episode_reward += reward
        steps += 1
    
    # Final update
    if prev_obs is not None:
        qlearner.update(prev_obs, prev_action, reward, obs, done=True)
    
    qlearner.reset_episode()
    
    # Track success
    if info['pos'] == info['goal']:
        successes += 1
    
    episode_returns.append(episode_reward)
    episode_lengths.append(steps)
    
    if (episode + 1) % 10 == 0:
        recent_returns = episode_returns[-10:]
        recent_lengths = episode_lengths[-10:]
        recent_success = sum(1 for i in range(len(episode_returns)-10, len(episode_returns)) 
                            if episode_returns[i] > 0) / 10
        print(f"  Episode {episode+1}: "
              f"return={episode_reward:.3f}, "
              f"avg_return={np.mean(recent_returns):.3f}, "
              f"avg_length={np.mean(recent_lengths):.1f}, "
              f"success_rate={recent_success:.1%}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}")
print("Results:")
print(f"{'='*60}")

# Compare early vs late (excluding demos)
early_returns = np.mean(episode_returns[:10])
late_returns = np.mean(episode_returns[-10:])
improvement = late_returns - early_returns

print(f"Early episodes (1-10):   {early_returns:.3f}")
print(f"Late episodes (86-95):   {late_returns:.3f}")
print(f"Improvement:             {improvement:+.3f}")
print(f"Total successes:         {successes}/95 ({successes/95:.1%})")

stats = qlearner.get_stats()
print(f"\nQ-learning stats:")
print(f"  Epsilon: {stats['epsilon']:.4f}")
print(f"  Total updates: {stats['n_updates']}")
print(f"  Avg TD error: {stats['avg_td_error']:.4f}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# Show final Q-values
print(f"\nFinal Q-values (sample states):")
for y in range(0, 5, 2):
    for x in range(0, 5, 2):
        state = np.array([x, y], dtype=np.float32)
        q_vals = qlearner.get_q_values(state)
        best_action = np.argmax(q_vals)
        actions = ['↑', '↓', '←', '→']
        print(f"  [{x},{y}]: best={actions[best_action]}, Q={q_vals[best_action]:.3f}")

if improvement > 0.2:
    print(f"\n✓ SUCCESS! Curriculum learning works!")
elif improvement > 0.05:
    print(f"\n⚠ Moderate improvement. Curriculum helped but needs tuning.")
else:
    print(f"\n✗ Limited improvement. May need more demonstrations or different approach.")

print("="*60)
