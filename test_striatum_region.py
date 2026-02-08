"""
Test Striatum Region Standalone

Validate that Striatum region achieves 100% success on GridWorld
with curriculum learning (matching standalone Q-learning baseline).
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.learning.qlearning import QLearningConfig
from throng3.environments import GridWorldAdapter

print("="*60)
print("Striatum Region Standalone Test")
print("="*60)
print("Target: 100% success (matching Q-learning baseline)")
print()

# Create environment
env = GridWorldAdapter()

# Create Striatum region
striatum = StriatumRegion(
    n_states=2,  # [x, y] observations
    n_actions=4,  # up, down, left, right
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

print(f"Striatum region created:")
print(f"  State dim: 2 (raw observations)")
print(f"  Actions: 4")
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
    
    for step, action in enumerate(optimal_actions):
        # Striatum step (learns from previous step's reward)
        result = striatum.step({
            'raw_observation': obs,
            'reward': prev_reward,  # STEP reward
            'done': False
        })
        
        # Take optimal action (override Striatum's selection for demo)
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        episode_reward += reward
        
        if done:
            # Final update with last step's reward
            striatum.step({
                'raw_observation': obs,
                'reward': prev_reward,
                'done': True
            })
            break
    
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}")

stats = striatum.get_stats()
print(f"\nAfter demonstrations:")
print(f"  Q-learner updates: {stats['n_updates']}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

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
    prev_reward = 0  # Track previous step's reward
    
    while not done and steps < 100:
        # Striatum selects action (gets previous step's reward)
        result = striatum.step({
            'raw_observation': obs,
            'reward': prev_reward,  # STEP reward, not cumulative
            'done': False
        })
        
        action = result['action']
        
        # Execute action
        obs, reward, done, info = env.step(action)
        prev_reward = reward  # Store for next iteration
        episode_reward += reward
        steps += 1
    
    # Final update with last step's reward
    striatum.step({
        'raw_observation': obs,
        'reward': prev_reward,
        'done': True
    })
    
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

stats = striatum.get_stats()
print(f"\nStriatum stats:")
print(f"  Epsilon: {stats['epsilon']:.4f}")
print(f"  Updates: {stats['n_updates']}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# ============================================================
# VALIDATION
# ============================================================
print(f"\n{'='*60}")
print("VALIDATION")
print(f"{'='*60}")

print(f"\nBaseline (standalone Q-learning): 100%")
print(f"Striatum region:                  {successes/95:.1%}")

if successes / 95 >= 0.95:
    print(f"\n✓ SUCCESS! Striatum region matches baseline!")
    print(f"  Regional architecture validated.")
elif successes / 95 >= 0.8:
    print(f"\n⚠ CLOSE - minor tuning needed")
else:
    print(f"\n✗ FAILED - Striatum region underperforming")
    print(f"  Need to debug regional implementation")

print("="*60)
