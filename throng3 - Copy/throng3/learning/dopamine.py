"""
Dopamine Reward System — Three-Factor Learning

Extends throng2's dopamine system for Meta^N:
- Reward Prediction Error (RPE) computation
- Modulates learning rates across all meta-layers
- Temporal difference learning
- Multi-timescale reward tracking (for Meta^4 GoalHierarchy)

Three-factor learning: dw = eligibility_trace * dopamine * STDP
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class DopamineConfig:
    """Configuration for dopamine system."""
    baseline: float = 0.0
    learning_rate: float = 0.1       # TD learning rate
    gamma: float = 0.99              # Discount factor
    trace_decay: float = 0.9         # Eligibility trace decay
    burst_scale: float = 2.0         # Dopamine burst amplification
    dip_scale: float = 0.5           # Dopamine dip amplification
    adaptation_rate: float = 0.01    # How fast baseline adapts


class DopamineSystem:
    """
    Dopamine neuromodulation with multi-timescale reward tracking.
    
    Provides the third factor in three-factor learning rules,
    and drives Meta^4 goal hierarchy through reward decomposition.
    """
    
    def __init__(self, config: Optional[DopamineConfig] = None):
        self.config = config or DopamineConfig()
        
        # Core state
        self.level = self.config.baseline
        self.expected_reward = 0.0
        
        # Multi-timescale tracking (for Meta^4)
        self.short_term_reward = 0.0   # τ ~ 10 steps
        self.medium_term_reward = 0.0  # τ ~ 100 steps
        self.long_term_reward = 0.0    # τ ~ 1000 steps
        
        # Value function (critic)
        self._value_estimates: Dict[str, float] = {}
        
        # Eligibility traces for three-factor learning
        self._eligibility: Dict[str, float] = {}
        
        # History
        self.rpe_history: deque = deque(maxlen=1000)
        self.reward_history: deque = deque(maxlen=1000)
        self.level_history: deque = deque(maxlen=1000)
    
    def compute_rpe(self, actual_reward: float,
                    state: Optional[str] = None,
                    next_state: Optional[str] = None) -> float:
        """
        Compute Reward Prediction Error using TD learning.
        
        RPE = reward + γ * V(next_state) - V(state)
        
        Args:
            actual_reward: Observed reward
            state: Current state identifier (for state-dependent values)
            next_state: Next state identifier
            
        Returns:
            rpe: Reward prediction error (dopamine signal)
        """
        # Get value estimates
        v_current = self._value_estimates.get(state, self.expected_reward) if state else self.expected_reward
        v_next = self._value_estimates.get(next_state, self.expected_reward) if next_state else self.expected_reward
        
        # TD error
        rpe = actual_reward + self.config.gamma * v_next - v_current
        
        # Update value estimate
        if state:
            self._value_estimates[state] = v_current + self.config.learning_rate * rpe
        
        # Update global expected reward
        self.expected_reward += self.config.learning_rate * (actual_reward - self.expected_reward)
        
        # Update dopamine level with burst/dip asymmetry
        if rpe > 0:
            self.level = self.config.baseline + rpe * self.config.burst_scale
        else:
            self.level = self.config.baseline + rpe * self.config.dip_scale
        
        # Adapt baseline slowly
        self.config.baseline += self.config.adaptation_rate * (self.level - self.config.baseline)
        
        # Update multi-timescale trackers
        self.short_term_reward = 0.9 * self.short_term_reward + 0.1 * actual_reward
        self.medium_term_reward = 0.99 * self.medium_term_reward + 0.01 * actual_reward
        self.long_term_reward = 0.999 * self.long_term_reward + 0.001 * actual_reward
        
        # Record
        self.rpe_history.append(rpe)
        self.reward_history.append(actual_reward)
        self.level_history.append(self.level)
        
        return rpe
    
    def modulate_learning_rate(self, base_rate: float) -> float:
        """
        Modulate a learning rate by current dopamine level.
        
        Positive dopamine → increase learning (reinforce)
        Negative dopamine → decrease learning (suppress)
        """
        modulation = 1.0 + np.tanh(self.level - self.config.baseline)
        return base_rate * max(modulation, 0.01)  # Never fully suppress
    
    def update_eligibility(self, key: str, value: float):
        """Update eligibility trace for three-factor learning."""
        self._eligibility[key] = value
    
    def get_eligibility(self, key: str) -> float:
        """Get eligibility trace value."""
        return self._eligibility.get(key, 0.0)
    
    def decay_eligibility(self):
        """Decay all eligibility traces."""
        decay = self.config.trace_decay
        for key in list(self._eligibility.keys()):
            self._eligibility[key] *= decay
            if abs(self._eligibility[key]) < 1e-6:
                del self._eligibility[key]
    
    def three_factor_dw(self, stdp_dw: float, eligibility: float) -> float:
        """
        Three-factor learning: dw = eligibility * dopamine * stdp_dw
        
        Combines:
        1. STDP temporal correlation (what happened)
        2. Eligibility trace (when it happened)
        3. Dopamine (whether it was good)
        """
        return eligibility * self.level * stdp_dw
    
    def get_reward_decomposition(self) -> Dict[str, float]:
        """
        Get multi-timescale reward decomposition for Meta^4.
        
        Returns:
            Dict with short/medium/long term reward estimates
        """
        return {
            'short_term': self.short_term_reward,
            'medium_term': self.medium_term_reward,
            'long_term': self.long_term_reward,
            'current_rpe': float(self.rpe_history[-1]) if self.rpe_history else 0.0,
            'dopamine_level': self.level,
            'expected_reward': self.expected_reward,
        }
    
    def get_stats(self) -> Dict[str, float]:
        """Get dopamine system statistics."""
        rpe_arr = np.array(list(self.rpe_history)) if self.rpe_history else np.array([0.0])
        return {
            'dopamine_level': self.level,
            'baseline': self.config.baseline,
            'expected_reward': self.expected_reward,
            'mean_rpe': float(np.mean(rpe_arr)),
            'std_rpe': float(np.std(rpe_arr)),
            'n_positive_rpe': int(np.sum(rpe_arr > 0)),
            'n_negative_rpe': int(np.sum(rpe_arr < 0)),
            'n_eligibility_traces': len(self._eligibility),
        }
