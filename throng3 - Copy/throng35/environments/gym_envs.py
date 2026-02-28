"""Concrete environment adapters for Gym environments."""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .adapter import EnvironmentAdapter


class GridWorldAdapter(EnvironmentAdapter):
    """
    Simple 5x5 grid navigation environment.
    
    - Start: (0, 0)
    - Goal: (4, 4)
    - Actions: 0=up, 1=down, 2=left, 3=right
    - Reward: -0.01 per step, +1.0 at goal
    """
    
    def __init__(self, size: int = 5):
        super().__init__()
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
        
    def reset(self) -> np.ndarray:
        """Reset to start position."""
        self.pos = (0, 0)
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(self.pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action and return new state."""
        # Move based on action
        x, y = self.pos
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.size - 1, x + 1)
        
        self.pos = (x, y)
        
        # Compute reward
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        # Max episode length
        if self.episode_steps >= 100:
            done = True
        
        info = {
            'pos': self.pos,
            'goal': self.goal,
            'episode_steps': self.episode_steps,
        }
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(self.pos), reward, done, info
    
    def preprocess_obs(self, obs: Tuple[int, int]) -> np.ndarray:
        """Convert (x, y) position to normalized array."""
        x, y = obs
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)


class CartPoleAdapter(EnvironmentAdapter):
    """
    Adapter for CartPole-v1 environment.
    
    - Observation: [cart_pos, cart_vel, pole_angle, pole_vel]
    - Actions: 0=left, 1=right
    """
    
    def __init__(self):
        super().__init__()
        try:
            import gymnasium as gym
            self.env = gym.make('CartPole-v1')
        except ImportError:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")
        
        # Observation bounds for normalization
        self.obs_low = np.array([-4.8, -5.0, -0.42, -5.0])
        self.obs_high = np.array([4.8, 5.0, 0.42, 5.0])
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        obs, _ = self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(obs), reward, done, info
    
    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1]."""
        # Clip to bounds then normalize
        obs_clipped = np.clip(obs, self.obs_low, self.obs_high)
        return self.normalize(obs_clipped, self.obs_low, self.obs_high)


class MountainCarAdapter(EnvironmentAdapter):
    """
    Adapter for MountainCar-v0 environment.
    
    - Observation: [position, velocity]
    - Actions: 0=left, 1=nothing, 2=right
    """
    
    def __init__(self):
        super().__init__()
        try:
            import gymnasium as gym
            self.env = gym.make('MountainCar-v0')
        except ImportError:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")
        
        # Observation bounds
        self.obs_low = np.array([-1.2, -0.07])
        self.obs_high = np.array([0.6, 0.07])
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        obs, _ = self.env.reset()
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(obs)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(obs), reward, done, info
    
    def preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to [0, 1]."""
        obs_clipped = np.clip(obs, self.obs_low, self.obs_high)
        return self.normalize(obs_clipped, self.obs_low, self.obs_high)
