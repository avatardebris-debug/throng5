"""Base environment adapter for throng3."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np

from throng4.learning.abstract_features import (
    AbstractFeature, CORE_SIZE, EXT_MAX, empty_core, make_ext
)


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

    # ── Save/Load State (for rehearsal loop) ─────────────────────────

    @property
    def supports_save_state(self) -> bool:
        """Whether this adapter supports save/load state."""
        return False

    def save_state(self) -> Optional[bytes]:
        """Save emulator state. Returns opaque state bytes or None."""
        return None

    def load_state(self, state: bytes) -> None:
        """Restore emulator state from saved bytes."""
        pass
        
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

    # ------------------------------------------------------------------
    # Abstract Feature Protocol (two-layer portable representation)
    # ------------------------------------------------------------------

    def get_core_features(self) -> np.ndarray:
        """
        Return the 20-dim universal core vector for the current state.
        Override in each adapter. Default is all zeros (safe fallback).
        """
        return empty_core()

    def get_ext_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (ext_values, ext_mask) each shape (EXT_MAX,).
        Override in each adapter to expose game-specific extra features.
        Default: all zeros / all masked off.
        """
        return np.zeros(EXT_MAX, dtype=np.float32), np.zeros(EXT_MAX, dtype=np.float32)

    def get_abstract_features(self, action: int) -> 'AbstractFeature':
        """
        Build a full AbstractFeature for the current state + action.
        Must be called BEFORE step() — same rule as make_features().
        """
        core = self.get_core_features()
        ext, mask = self.get_ext_features()
        return AbstractFeature(core=core, ext=ext, ext_mask=mask)

    def get_blind_obs_str(self, action: int, reward: float,
                          action_name: Optional[str] = None) -> str:
        """
        Blind semantic log line using abstract field names only.
        No game-specific names — safe to send to Tetra without revealing game identity.
        """
        af = self.get_abstract_features(action)
        name = action_name or str(action)
        return af.blind_log_str(action_name=name, reward=reward, step=self.episode_steps)

