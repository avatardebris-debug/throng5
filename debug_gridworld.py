"""
Debug GridWorld to see if it's actually functioning.
"""

import numpy as np
from throng3.environments import GridWorldAdapter

def test_gridworld_basic():
    """Test basic GridWorld functionality."""
    print("="*60)
    print("GridWorld Debug Test")
    print("="*60)
    
    env = GridWorldAdapter(size=5)
    
    print(f"\nEnvironment created:")
    print(f"  Size: {env.size}x{env.size}")
    print(f"  Start: {(0, 0)}")
    print(f"  Goal: {env.goal}")
    
    # Test reset
    obs = env.reset()
    print(f"\nAfter reset:")
    print(f"  Position: {env.pos}")
    print(f"  Observation: {obs}")
    print(f"  Obs shape: {obs.shape}")
    
    # Test manual navigation to goal
    print(f"\nManual navigation test (optimal path):")
    print(f"  Need to go: right 4 times, down 4 times")
    
    total_reward = 0
    step_count = 0
    
    # Go right 4 times
    for i in range(4):
        obs, reward, done, info = env.step(3)  # action 3 = right
        total_reward += reward
        step_count += 1
        print(f"  Step {step_count}: action=right, pos={env.pos}, reward={reward:.2f}, done={done}")
        if done:
            break
    
    # Go down 4 times
    if not done:
        for i in range(4):
            obs, reward, done, info = env.step(1)  # action 1 = down
            total_reward += reward
            step_count += 1
            print(f"  Step {step_count}: action=down, pos={env.pos}, reward={reward:.2f}, done={done}")
            if done:
                break
    
    print(f"\nFinal results:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final position: {env.pos}")
    print(f"  Reached goal: {env.pos == env.goal}")
    print(f"  Episode done: {done}")
    
    # Expected: 8 steps, reward = -0.01*7 + 1.0 = 0.93
    expected_reward = -0.01 * 7 + 1.0
    print(f"\n  Expected reward: {expected_reward:.2f}")
    print(f"  Match: {'✓ YES' if abs(total_reward - expected_reward) < 0.01 else '✗ NO'}")
    
    # Test random policy
    print(f"\n{'='*60}")
    print("Random policy test (100 episodes):")
    print(f"{'='*60}")
    
    episode_returns = []
    episode_lengths = []
    successes = 0
    
    for ep in range(100):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            action = np.random.randint(4)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        episode_returns.append(episode_reward)
        episode_lengths.append(steps)
        if env.pos == env.goal:
            successes += 1
    
    print(f"\nRandom policy results:")
    print(f"  Success rate: {successes}/100 = {successes}%")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
    print(f"  Avg episode return: {np.mean(episode_returns):.3f}")
    print(f"  Min return: {np.min(episode_returns):.3f}")
    print(f"  Max return: {np.max(episode_returns):.3f}")
    
    if successes > 0:
        print(f"\n✓ GridWorld is functioning - random policy can reach goal")
    else:
        print(f"\n✗ WARNING: Random policy never reached goal in 100 episodes!")
    
    print("="*60)

if __name__ == "__main__":
    test_gridworld_basic()
