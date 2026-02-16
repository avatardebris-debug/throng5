"""
Atari Adapter for Meta-Learning Validation.

Uses RAM state (128 bytes) for fast iteration and concept transfer testing.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple, Optional


class AtariAdapter:
    """
    Adapter for Atari games using RAM state.
    
    Compatible with PortableNNAgent for concept transfer experiments.
    """
    
    def __init__(self, game_name: str, max_steps: int = 10000):
        """
        Initialize Atari environment.
        
        Args:
            game_name: Atari game name (e.g., 'Breakout', 'Pong', 'SpaceInvaders')
            max_steps: Maximum steps per episode
        """
        self.game_name = game_name
        self.max_steps = max_steps
        
        # Register ALE environments
        import ale_py
        import gymnasium
        gymnasium.register_envs(ale_py)
        
        # Create environment with RAM observation
        env_id = f'ALE/{game_name}-v5'
        self.env = gymnasium.make(env_id, obs_type='ram')
        
        # State tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        self.done = False
        
        # RAM state is 128 bytes
        self.n_features = 128
        
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state."""
        obs, info = self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.done = False
        
        # RAM state is already a feature vector (128 bytes)
        return obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action (0 to n_actions-1)
            
        Returns:
            (state, reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated or (self.episode_steps >= self.max_steps)
        
        self.episode_steps += 1
        self.episode_reward += reward
        self.done = done
        
        if done:
            self.total_episodes += 1
        
        # Normalize RAM state
        state = obs.astype(np.float32) / 255.0
        
        # Add episode info
        info['episode_steps'] = self.episode_steps
        info['episode_reward'] = self.episode_reward
        
        return state, reward, done, info
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions."""
        return list(range(self.env.action_space.n))
    
    def get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        return {
            'game': self.game_name,
            'n_actions': self.env.action_space.n,
            'n_features': self.n_features,
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'total_episodes': self.total_episodes
        }
    
    def close(self):
        """Close the environment."""
        self.env.close()


def extract_features_for_concept(state: np.ndarray, game_name: str) -> Dict[str, float]:
    """
    Extract high-level features from RAM state for concept matching.
    
    This is a simple heuristic extractor. In practice, concepts would
    guide which features to extract.
    
    Args:
        state: RAM state (128 bytes, normalized)
        game_name: Name of the game
        
    Returns:
        Dictionary of named features
    """
    # Generic features that might apply across games
    features = {
        'state_mean': float(np.mean(state)),
        'state_std': float(np.std(state)),
        'state_max': float(np.max(state)),
        'state_min': float(np.min(state)),
        'nonzero_ratio': float(np.count_nonzero(state) / len(state))
    }
    
    # Game-specific feature extraction could be added here
    # based on concept suggestions from Tetra
    
    return features


if __name__ == "__main__":
    # Test the adapter
    print("Testing AtariAdapter with Breakout...")
    
    adapter = AtariAdapter('Breakout')
    print(f"Game: {adapter.game_name}")
    print(f"Features: {adapter.n_features}")
    print(f"Actions: {len(adapter.get_valid_actions())}")
    
    # Run a few random steps
    state = adapter.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")
    
    for i in range(10):
        action = np.random.choice(adapter.get_valid_actions())
        state, reward, done, info = adapter.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.1f}, done={done}")
        
        if done:
            print("Episode ended early")
            break
    
    adapter.close()
    print("\n✅ AtariAdapter test complete!")
