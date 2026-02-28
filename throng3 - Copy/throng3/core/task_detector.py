"""
Task Detector - Automatically Detect Task Characteristics

Monitors the context over time to determine:
- Is this supervised learning (has targets)?
- Is this RL (has rewards)?
- Is the signal clean or noisy?
- What learning mechanism should we use?

This is what enables Meta^N to adapt to different task types
without manual configuration.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass


@dataclass
class TaskCharacteristics:
    """Detected characteristics of the current task."""
    has_target: bool
    has_reward: bool
    target_frequency: float  # Fraction of steps with targets
    reward_frequency: float  # Fraction of steps with non-zero rewards
    reward_variance: float
    target_variance: float
    signal_type: str  # 'supervised', 'rl', 'hybrid', 'unknown'
    confidence: float  # How confident are we in this classification?


class TaskDetector:
    """
    Detects task characteristics from context history.
    
    This is the key to adaptive learning - by detecting what kind
    of task we're facing, Meta^2 can select the right learning mechanism.
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.history: deque = deque(maxlen=window)
        self._characteristics: Optional[TaskCharacteristics] = None
    
    def update(self, context: Dict[str, Any]):
        """
        Update detector with new context.
        
        Args:
            context: Pipeline context with input, target, reward, etc.
        """
        target = context.get('target')
        reward = context.get('reward', 0.0)
        
        self.history.append({
            'has_target': target is not None,
            'has_reward': abs(reward) > 1e-6,
            'target_value': target if target is not None else None,
            'reward_value': reward,
            'step': context.get('step', 0),
        })
    
    def get_characteristics(self) -> Optional[TaskCharacteristics]:
        """
        Analyze history and return task characteristics.
        
        Returns:
            TaskCharacteristics if enough data, None otherwise
        """
        if len(self.history) < 10:
            return None
        
        # Compute frequencies
        has_target_list = [h['has_target'] for h in self.history]
        has_reward_list = [h['has_reward'] for h in self.history]
        
        target_freq = np.mean(has_target_list)
        reward_freq = np.mean(has_reward_list)
        
        # Compute variances (for signal quality)
        rewards = [h['reward_value'] for h in self.history if h['has_reward']]
        reward_var = np.var(rewards) if len(rewards) > 1 else 0.0
        
        # For targets, we need to handle None values
        targets_present = [h['target_value'] for h in self.history 
                          if h['target_value'] is not None]
        if targets_present and len(targets_present) > 1:
            # Flatten targets if they're arrays
            target_values = []
            for t in targets_present:
                if isinstance(t, np.ndarray):
                    target_values.extend(t.flatten())
                else:
                    target_values.append(t)
            target_var = np.var(target_values) if target_values else 0.0
        else:
            target_var = 0.0
        
        # Classify signal type
        signal_type, confidence = self._classify_signal_type(
            target_freq, reward_freq, reward_var, target_var
        )
        
        self._characteristics = TaskCharacteristics(
            has_target=target_freq > 0.5,
            has_reward=reward_freq > 0.1,
            target_frequency=target_freq,
            reward_frequency=reward_freq,
            reward_variance=reward_var,
            target_variance=target_var,
            signal_type=signal_type,
            confidence=confidence,
        )
        
        return self._characteristics
    
    def _classify_signal_type(self, 
                              target_freq: float,
                              reward_freq: float,
                              reward_var: float,
                              target_var: float) -> tuple[str, float]:
        """
        Classify the task type based on signal characteristics.
        
        Returns:
            (signal_type, confidence)
        """
        # Supervised: frequent targets, rare rewards
        if target_freq > 0.8 and reward_freq < 0.1:
            return 'supervised', 0.9
        
        # RL: frequent rewards, rare targets
        elif reward_freq > 0.3 and target_freq < 0.2:
            return 'rl', 0.8
        
        # Hybrid: both targets and rewards
        elif target_freq > 0.5 and reward_freq > 0.3:
            return 'hybrid', 0.7
        
        # Unknown: not enough signal
        else:
            return 'unknown', 0.3
    
    def is_reward_meaningful(self, reward: float) -> bool:
        """
        Check if a reward value is meaningful or just noise.
        
        A reward of 0.0 could mean:
        1. Actual zero reward (meaningful)
        2. No reward signal provided (not meaningful)
        
        We distinguish by checking if we've EVER seen non-zero rewards.
        """
        if abs(reward) > 1e-6:
            return True  # Non-zero is always meaningful
        
        # Zero reward - check if we've seen any non-zero rewards
        if len(self.history) < 5:
            return False  # Not enough data
        
        has_any_reward = any(h['has_reward'] for h in self.history)
        return has_any_reward  # Meaningful if we've seen rewards before
    
    def get_recommended_mechanism(self) -> str:
        """
        Recommend which learning mechanism to use.
        
        Returns:
            'gradient', 'rl', 'hybrid', or 'explore'
        """
        chars = self.get_characteristics()
        
        if chars is None or chars.confidence < 0.5:
            return 'explore'  # Not enough data, try everything
        
        if chars.signal_type == 'supervised':
            return 'gradient'
        elif chars.signal_type == 'rl':
            return 'rl'
        elif chars.signal_type == 'hybrid':
            return 'hybrid'
        else:
            return 'explore'
    
    def reset(self):
        """Reset detector (for new task)."""
        self.history.clear()
        self._characteristics = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        chars = self.get_characteristics()
        if chars is None:
            return {'status': 'insufficient_data', 'samples': len(self.history)}
        
        return {
            'signal_type': chars.signal_type,
            'confidence': chars.confidence,
            'target_freq': chars.target_frequency,
            'reward_freq': chars.reward_frequency,
            'samples': len(self.history),
        }
