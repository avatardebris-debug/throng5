"""
Hebbian Learning — "Neurons that fire together wire together"

Classic correlation-based learning with modern enhancements:
- Oja's rule for normalization (prevents weight explosion)
- BCM theory for sliding threshold
- Competitive learning mode

Configurable by Meta^2 (LearningRuleSelector).
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class HebbianConfig:
    """Configuration for Hebbian learning, tunable by Meta^2."""
    learning_rate: float = 0.01
    decay: float = 0.001            # Weight decay (prevents explosion)
    normalize: bool = True           # Use Oja's normalization
    bcm: bool = False                # Use BCM sliding threshold
    bcm_tau: float = 100.0           # BCM threshold time constant
    competitive: bool = False        # Winner-take-all mode
    w_max: float = 1.0
    w_min: float = -1.0


class HebbianRule:
    """
    Hebbian learning rule with normalization and BCM variants.
    """
    
    def __init__(self, config: Optional[HebbianConfig] = None):
        self.config = config or HebbianConfig()
        
        # BCM sliding threshold per neuron
        self._bcm_threshold: Optional[np.ndarray] = None
        
        # Statistics
        self.total_updates = 0
        self.mean_dw = 0.0
    
    def compute_dw(self, pre_activity: float, post_activity: float,
                   weight: float) -> float:
        """
        Compute weight change for a single synapse.
        
        dw = η * pre * post - decay * w   (basic Hebbian + decay)
        With Oja's: dw = η * (pre * post - post² * w)
        """
        if self.config.normalize:
            # Oja's rule: self-normalizing
            dw = self.config.learning_rate * (
                pre_activity * post_activity - post_activity ** 2 * weight
            )
        else:
            # Basic Hebbian with decay
            dw = (self.config.learning_rate * pre_activity * post_activity
                  - self.config.decay * weight)
        
        self.total_updates += 1
        self.mean_dw = 0.99 * self.mean_dw + 0.01 * dw
        return dw
    
    def batch_update(self, weights: np.ndarray,
                     pre_activity: np.ndarray,
                     post_activity: np.ndarray) -> np.ndarray:
        """
        Batch Hebbian update for a weight matrix.
        
        Args:
            weights: (N_post, N_pre) weight matrix
            pre_activity: (N_pre,) activation vector
            post_activity: (N_post,) activation vector
            
        Returns:
            dW: Weight change matrix
        """
        if self.config.bcm:
            dW = self._bcm_update(weights, pre_activity, post_activity)
        elif self.config.normalize:
            dW = self._oja_update(weights, pre_activity, post_activity)
        else:
            dW = self._basic_update(weights, pre_activity, post_activity)
        
        if self.config.competitive:
            dW = self._apply_competition(dW, post_activity)
        
        # Clip
        new_weights = np.clip(weights + dW, self.config.w_min, self.config.w_max)
        return new_weights - weights
    
    def _basic_update(self, weights, pre, post) -> np.ndarray:
        """Basic Hebbian: dW = η * post ⊗ pre - λ * W"""
        return (self.config.learning_rate * np.outer(post, pre)
                - self.config.decay * weights)
    
    def _oja_update(self, weights, pre, post) -> np.ndarray:
        """Oja's rule: dW = η * (post ⊗ pre - post² * W)"""
        post_sq = post ** 2
        return self.config.learning_rate * (
            np.outer(post, pre) - np.outer(post_sq, np.ones_like(pre)) * weights
        )
    
    def _bcm_update(self, weights, pre, post) -> np.ndarray:
        """BCM theory: sliding threshold determines LTP vs LTD."""
        N_post = len(post)
        
        # Initialize threshold
        if self._bcm_threshold is None:
            self._bcm_threshold = np.ones(N_post) * 0.5
        
        # BCM nonlinearity: φ(post, θ) = post * (post - θ)
        phi = post * (post - self._bcm_threshold[:N_post])
        
        # Update threshold (sliding average of post²)
        self._bcm_threshold[:N_post] += (
            (post ** 2 - self._bcm_threshold[:N_post]) / self.config.bcm_tau
        )
        
        # Weight update
        return self.config.learning_rate * np.outer(phi, pre)
    
    def _apply_competition(self, dW, post_activity) -> np.ndarray:
        """Winner-take-all: only most active neuron learns."""
        winner = np.argmax(post_activity)
        mask = np.zeros_like(dW)
        mask[winner, :] = 1.0
        return dW * mask
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameters."""
        return {
            'learning_rate': self.config.learning_rate,
            'decay': self.config.decay,
            'normalize': float(self.config.normalize),
            'bcm': float(self.config.bcm),
            'competitive': float(self.config.competitive),
        }
    
    def set_params(self, params: Dict[str, float]):
        """Set parameters (called by Meta^2)."""
        for key, value in params.items():
            if key in ('normalize', 'bcm', 'competitive'):
                setattr(self.config, key, bool(value))
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_stats(self) -> Dict[str, float]:
        return {
            'total_updates': self.total_updates,
            'mean_dw': self.mean_dw,
        }
