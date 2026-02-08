"""
Simplest possible learning test: 1D corridor.
Agent starts at position 0, goal is at position 4.
Actions: 0=left, 1=right
This is SO simple that any learning algorithm should solve it.
"""

import numpy as np
from simple_baseline import SimplePolicyNetwork

class CorridorEnv:
    """1D corridor - simplest possible navigation task."""
    
    def __init__(self, length=5):
        self.length = length
        self.goal = length - 1
        self.pos = 0
        
    def reset(self):
        self.pos = 0
        return np.array([self.pos / (self.length - 1)], dtype=np.float32)
    
    def step(self, action):
        # action: 0=left, 1=right
        if action == 1:
            self.pos = min(self.pos + 1, self.length - 1)
        else:
            self.pos = max(self.pos - 1, 0)
        
        # Reward
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        obs = np.array([self.pos / (self.length - 1)], dtype=np.float32)
        return obs, reward, done, {}


def test_corridor():
    """Test if simple baseline can learn 1D corridor."""
    print("="*60)
    print("1D Corridor Learning Test")
    print("="*60)
    print("\nTask: Navigate from position 0 to position 4")
    print("Actions: 0=left, 1=right")
    print("Optimal: Go right 4 times (4 steps, reward=0.96)")
    print()
    
    env = CorridorEnv(length=5)
    
    # Create network
    network = SimplePolicyNetwork(
        n_inputs=1,
        n_hidden=32,  # Small network for simple task
        n_outputs=2,
        lr=0.01
    )
    
    episode_returns = []
    episode_lengths = []
    
    print("Training for 100 episodes...\n")
    
    for episode in range(100):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 20:
            _, action = network.forward(obs, training=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            network.store_reward(reward)
        
        network.update()
        episode_returns.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0:
            recent_returns = episode_returns[-10:]
            recent_lengths = episode_lengths[-10:]
            print(f"  Episode {episode+1}: "
                  f"avg_return={np.mean(recent_returns):.3f}, "
                  f"avg_length={np.mean(recent_lengths):.1f}")
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    
    early_returns = np.mean(episode_returns[:10])
    late_returns = np.mean(episode_returns[-10:])
    early_lengths = np.mean(episode_lengths[:10])
    late_lengths = np.mean(episode_lengths[-10:])
    
    print(f"Early episodes (1-10):")
    print(f"  Avg return: {early_returns:.3f}")
    print(f"  Avg length: {early_lengths:.1f} steps")
    
    print(f"\nLate episodes (91-100):")
    print(f"  Avg return: {late_returns:.3f}")
    print(f"  Avg length: {late_lengths:.1f} steps")
    
    print(f"\nImprovement:")
    print(f"  Return: {late_returns - early_returns:+.3f}")
    print(f"  Length: {late_lengths - early_lengths:+.1f} steps")
    
    print(f"\nOptimal performance:")
    print(f"  Return: 0.960 (4 steps)")
    print(f"  Achieved: {late_returns:.3f}")
    
    if late_returns > 0.8:
        print(f"\n✓ SUCCESS: Network learned the task!")
    elif late_returns > early_returns + 0.1:
        print(f"\n⚠ PARTIAL: Network shows learning but not optimal")
    else:
        print(f"\n✗ FAILURE: Network did not learn")
    
    print("="*60)

if __name__ == "__main__":
    test_corridor()
