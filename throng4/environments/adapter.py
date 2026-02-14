"""Base environment adapter for throng3."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np


class EnvironmentAdapter(ABC):
    """
    Bridge between Gym environments and MetaNPipeline.
    
    Handles:
    - Observation preprocessing (flatten, normalize)
    - Action discretization/mapping
    - Episode management (reset, done tracking)
    """
    
    def __init__(self):
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.total_episodes = 0
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment and return initial observation.
        
        Returns:
            Preprocessed observation as 1D numpy array
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take action in environment.
        
        Args:
            action: Discrete action index
            
        Returns:
            obs: Preprocessed observation (1D array)
            reward: Scalar reward
            done: Episode termination flag
            info: Additional metadata
        """
        pass
    
    @abstractmethod
    def preprocess_obs(self, obs: Any) -> np.ndarray:
        """
        Convert raw observation to normalized 1D array.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            Flattened, normalized observation in [0, 1] range
        """
        pass
    
    def normalize(self, values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """
        Normalize values to [0, 1] range.
        
        Args:
            values: Array to normalize
            min_val: Minimum possible value (scalar or array)
            max_val: Maximum possible value (scalar or array)
            
        Returns:
            Normalized array
        """
        min_val = np.asarray(min_val)
        max_val = np.asarray(max_val)
        range_val = max_val - min_val
        
        # Handle zero range
        if np.any(range_val == 0):
            result = np.zeros_like(values, dtype=np.float32)
            mask = range_val != 0
            result[mask] = (values[mask] - min_val[mask]) / range_val[mask]
            return result
        
        return (values - min_val) / range_val
    
    def flatten(self, obs: Any) -> np.ndarray:
        """Flatten observation to 1D array."""
        if isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, (list, tuple)):
            return np.array(obs).flatten()
        else:
            return np.array([obs]).flatten()
