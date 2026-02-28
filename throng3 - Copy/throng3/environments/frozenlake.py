"""
FrozenLake environment adapter.
Simpler than GridWorld - good for validation.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .adapter import EnvironmentAdapter


class FrozenLakeAdapter(EnvironmentAdapter):
    """
    FrozenLake 4x4 environment.
    
    Grid layout:
    S F F F
    F H F H
    F F F H
    H F F G
    
    S = Start, F = Frozen (safe), H = Hole (fail), G = Goal
    Actions: 0=left, 1=down, 2=right, 3=up
    Reward: +1.0 at goal, 0.0 otherwise
    """
    
    def __init__(self, is_slippery: bool = False):
        """
        Args:
            is_slippery: If True, actions are stochastic (33% chance of sliding)
        """
        super().__init__()
        self.is_slippery = is_slippery
        self.size = 4
        
        # Define the map
        self.desc = [
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG"
        ]
        
        # Holes and goal positions
        self.holes = [(1, 1), (1, 3), (2, 3), (3, 0)]
        self.goal = (3, 3)
        self.start = (0, 0)
        self.pos = self.start
        
    def reset(self) -> np.ndarray:
        """Reset to start position."""
        self.pos = self.start
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(self.pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action and return new state.
        
        Args:
            action: 0=left, 1=down, 2=right, 3=up
        """
        # Apply slipperiness
        if self.is_slippery and np.random.random() < 0.33:
            # Slip to a perpendicular direction
            action = np.random.choice([a for a in range(4) if a != action])
        
        # Move based on action
        x, y = self.pos
        if action == 0:  # left
            x = max(0, x - 1)
        elif action == 1:  # down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # right
            x = min(self.size - 1, x + 1)
        elif action == 3:  # up
            y = max(0, y - 1)
        
        self.pos = (x, y)
        self.episode_steps += 1
        
        # Check terminal conditions
        if self.pos in self.holes:
            reward = 0.0
            done = True
        elif self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        
        # Max episode length
        if self.episode_steps >= 100:
            done = True
        
        self.episode_reward += reward
        
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
