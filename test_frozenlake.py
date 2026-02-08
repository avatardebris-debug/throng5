"""
Test FrozenLake and run validation experiment.
"""

import numpy as np
from throng3.environments.frozenlake import FrozenLakeAdapter
from simple_baseline import SimplePolicyNetwork, train_simple_baseline

def test_frozenlake_basic():
    """Test FrozenLake environment."""
    print("="*60)
    print("FrozenLake Test")
    print("="*60)
    
    env = FrozenLakeAdapter(is_slippery=False)
    
    print("\nEnvironment:")
    print("  Map:")
    for row in env.desc:
        print(f"    {row}")
    print(f"  Start: {env.start}")
    print(f"  Goal: {env.goal}")
    print(f"  Holes: {env.holes}")
    
    # Test optimal path
    print("\nOptimal path test:")
    obs = env.reset()
    print(f"  Start pos: {env.pos}")
    
    # Optimal: right, right, down, down, down, right
    optimal_actions = [2, 2, 1, 1, 1, 2]
    total_reward = 0
    
    for i, action in enumerate(optimal_actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action_name = ['left', 'down', 'right', 'up'][action]
        print(f"  Step {i+1}: {action_name} → pos={env.pos}, reward={reward}, done={done}")
        if done:
            break
    
    print(f"\n  Total reward: {total_reward}")
    print(f"  Reached goal: {env.pos == env.goal}")
    
    # Test random policy
    print(f"\n{'='*60}")
    print("Random policy (100 episodes):")
    
    successes = 0
    episode_lengths = []
    
    for ep in range(100):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = np.random.randint(4)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        episode_lengths.append(steps)
        if env.pos == env.goal:
            successes += 1
    
    print(f"  Success rate: {successes}%")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")
    
    return successes > 0


def train_frozenlake():
    """Train baseline on FrozenLake."""
    print("\n" + "="*60)
    print("Training Baseline on FrozenLake")
    print("="*60)
    
    env = FrozenLakeAdapter(is_slippery=False)
    
    # Train
    returns, network = train_simple_baseline(env, n_episodes=100, verbose=True)
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluation (50 episodes):")
    print("="*60)
    
    successes = 0
    episode_lengths = []
    
    for ep in range(50):
        obs = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            _, action = network.forward(obs, training=False)
            obs, reward, done, info = env.step(action)
            steps += 1
        
        episode_lengths.append(steps)
        if env.pos == env.goal:
            successes += 1
    
    print(f"\nEvaluation results:")
    print(f"  Success rate: {successes}/50 = {successes*2}%")
    print(f"  Avg episode length: {np.mean(episode_lengths):.1f}")
    
    # Compare to training
    early_returns = np.mean(returns[:10])
    late_returns = np.mean(returns[-10:])
    
    print(f"\nTraining progress:")
    print(f"  Early returns (1-10): {early_returns:.3f}")
    print(f"  Late returns (91-100): {late_returns:.3f}")
    print(f"  Improvement: {late_returns - early_returns:+.3f}")
    
    if successes >= 30:  # 60% success
        print(f"\n✓ SUCCESS: Network learned FrozenLake!")
        return True
    elif successes >= 10:  # 20% success
        print(f"\n⚠ PARTIAL: Some learning, but not optimal")
        return True
    else:
        print(f"\n✗ FAILURE: Network did not learn")
        return False


if __name__ == "__main__":
    # Test environment
    env_works = test_frozenlake_basic()
    
    if env_works:
        # Train and validate
        learned = train_frozenlake()
        
        if learned:
            print("\n" + "="*60)
            print("✓ FrozenLake validation successful!")
            print("Ready for N=30 transfer learning experiments")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("⚠ FrozenLake learning incomplete")
            print("May need larger network or more episodes")
            print("="*60)
    else:
        print("\n✗ FrozenLake environment has issues")
