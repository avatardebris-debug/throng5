"""
Extended GridWorld Variants for Cross-Game Testing

Adds multiple game mechanics to test generalization:
1. Obstacles (spatial reasoning)
2. Stochastic transitions (uncertainty)
3. Sparse rewards (different reward structure)
4. Moving goal (dynamic objectives)
"""

from typing import Tuple, Dict, Any
import numpy as np
from throng3.environments.adapter import EnvironmentAdapter


class GridWorldWithObstacles(EnvironmentAdapter):
    """
    GridWorld with obstacles that block movement.
    
    Tests spatial reasoning and path planning.
    """
    
    def __init__(self, size: int = 5, obstacle_positions=None):
        super().__init__()
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
        
        # Default obstacles: create a wall
        if obstacle_positions is None:
            self.obstacles = {(2, i) for i in range(1, size - 1)}
        else:
            self.obstacles = set(obstacle_positions)
    
    def reset(self) -> np.ndarray:
        self.pos = (0, 0)
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(self.pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        x, y = self.pos
        
        # Try to move
        if action == 0:  # up
            new_pos = (x, max(0, y - 1))
        elif action == 1:  # down
            new_pos = (x, min(self.size - 1, y + 1))
        elif action == 2:  # left
            new_pos = (max(0, x - 1), y)
        elif action == 3:  # right
            new_pos = (min(self.size - 1, x + 1), y)
        
        # Check if new position is obstacle
        if new_pos in self.obstacles:
            new_pos = self.pos  # Stay in place
            reward = -0.1  # Penalty for hitting obstacle
        elif new_pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        self.pos = new_pos
        self.episode_steps += 1
        self.episode_reward += reward
        
        if self.episode_steps >= 100:
            done = True
        else:
            done = done if new_pos == self.goal else False
        
        info = {'pos': self.pos, 'goal': self.goal, 'hit_obstacle': new_pos in self.obstacles}
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(self.pos), reward, done, info
    
    def preprocess_obs(self, obs: Tuple[int, int]) -> np.ndarray:
        x, y = obs
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)


class StochasticGridWorld(EnvironmentAdapter):
    """
    GridWorld with stochastic transitions.
    
    30% chance action fails and agent moves randomly.
    Tests handling of uncertainty.
    """
    
    def __init__(self, size: int = 5, stochastic_prob: float = 0.3):
        super().__init__()
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
        self.stochastic_prob = stochastic_prob
    
    def reset(self) -> np.ndarray:
        self.pos = (0, 0)
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(self.pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        x, y = self.pos
        
        # Stochastic: random action with probability
        if np.random.random() < self.stochastic_prob:
            action = np.random.randint(0, 4)
        
        # Execute action
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(self.size - 1, x + 1)
        
        self.pos = (x, y)
        
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        if self.episode_steps >= 100:
            done = True
        
        info = {'pos': self.pos, 'goal': self.goal}
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(self.pos), reward, done, info
    
    def preprocess_obs(self, obs: Tuple[int, int]) -> np.ndarray:
        x, y = obs
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)


class SparseRewardGridWorld(EnvironmentAdapter):
    """
    GridWorld with only reward at goal (no step penalty).
    
    Tests sparse reward learning.
    """
    
    def __init__(self, size: int = 5):
        super().__init__()
        self.size = size
        self.goal = (size - 1, size - 1)
        self.pos = (0, 0)
    
    def reset(self) -> np.ndarray:
        self.pos = (0, 0)
        self.episode_steps = 0
        self.episode_reward = 0.0
        return self.preprocess_obs(self.pos)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
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
        
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = 0.0  # No step penalty
            done = False
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        if self.episode_steps >= 100:
            done = True
        
        info = {'pos': self.pos, 'goal': self.goal}
        
        if done:
            self.total_episodes += 1
        
        return self.preprocess_obs(self.pos), reward, done, info
    
    def preprocess_obs(self, obs: Tuple[int, int]) -> np.ndarray:
        x, y = obs
        return np.array([x / (self.size - 1), y / (self.size - 1)], dtype=np.float32)
