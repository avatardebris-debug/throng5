"""
Gradient-Based Learning for Supervised Tasks

Simple gradient descent for when we have clean target signals.
This is what Meta^1 should use for supervised learning tasks,
instead of STDP/Hebbian which are designed for RL.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class GradientConfig:
    """Configuration for gradient-based learning."""
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    clip_grad: float = 1.0


class GradientLearner:
    """
    Gradient descent learning for supervised tasks.
    
    Computes weight updates based on target error (MSE loss).
    Much simpler and more effective than STDP/Hebbian for supervised learning.
    """
    
    def __init__(self, config: Optional[GradientConfig] = None):
        self.config = config or GradientConfig()
        self._velocity = {}  # For momentum
        self._step = 0
        self._gradient_history = []  # Store gradients for Fisher info
        self._max_history = 100  # Keep last 100 gradients
    
    def compute_weight_update(self,
                             weights: Dict[str, np.ndarray],
                             output: np.ndarray,
                             target: np.ndarray,
                             activations: np.ndarray,
                             lr_multipliers: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Compute gradient descent weight updates.
        
        Args:
            weights: Dict of weight matrices {'W_out': ..., 'W_recurrent': ...}
            output: Network output (n_outputs,)
            target: Target output (n_outputs,)
            activations: Hidden activations (n_neurons,)
            lr_multipliers: Optional per-weight learning rate multipliers (Fisher boosting)
            
        Returns:
            Dict of weight deltas
        """
        dW = {}
        
        # Output layer gradient: dL/dW_out = error * activations^T
        error = output - target  # (n_outputs,)
        
        if 'W_out' in weights:
            W_out = weights['W_out']
            # dW_out = -lr * error @ activations^T
            dW_out = -self.config.learning_rate * np.outer(error, activations)
            
            # Add weight decay
            dW_out -= self.config.weight_decay * W_out
            
            # Clip gradients
            dW_out = np.clip(dW_out, -self.config.clip_grad, self.config.clip_grad)
            
            # Apply learning rate multipliers (Fisher boosting)
            if lr_multipliers and 'W_out' in lr_multipliers:
                dW_out *= lr_multipliers['W_out']
            
            # Apply momentum
            if 'W_out' not in self._velocity:
                self._velocity['W_out'] = np.zeros_like(W_out)
            
            self._velocity['W_out'] = (self.config.momentum * self._velocity['W_out'] + 
                                       (1 - self.config.momentum) * dW_out)
            
            dW['W_out'] = self._velocity['W_out']
        
        # For recurrent weights, we'd need BPTT (backprop through time)
        # For now, just use simple Hebbian-style update modulated by error
        if 'W_recurrent' in weights:
            W_rec = weights['W_recurrent']
            
            # Simple heuristic: strengthen connections that were active when error was low
            error_magnitude = np.mean(np.abs(error))
            error_signal = 1.0 - np.clip(error_magnitude, 0, 1)  # High when error low
            
            # Hebbian update modulated by error signal
            dW_rec = (self.config.learning_rate * error_signal * 
                     np.outer(activations, activations))
            
            # Weight decay
            dW_rec -= self.config.weight_decay * W_rec
            
            # Clip
            dW_rec = np.clip(dW_rec, -self.config.clip_grad, self.config.clip_grad)
            
            dW['W_recurrent'] = dW_rec
        
        self._step += 1
        
        # Store raw gradients (before momentum) for Fisher computation
        raw_gradients = {}
        if 'W_out' in weights:
            error = output - target
            raw_gradients['W_out'] = np.outer(error, activations)
        if 'W_recurrent' in weights:
            error_magnitude = np.mean(np.abs(output - target))
            error_signal = 1.0 - np.clip(error_magnitude, 0, 1)
            raw_gradients['W_recurrent'] = error_signal * np.outer(activations, activations)
        
        # Store in history (keep last N)
        self._gradient_history.append(raw_gradients)
        if len(self._gradient_history) > self._max_history:
            self._gradient_history.pop(0)
        
        return dW
    
    def get_fisher_information(self) -> Dict[str, np.ndarray]:
        """
        Compute Fisher information matrix: F = E[(dL/dW)^2]
        
        This measures how important each weight is for the current task.
        High Fisher value = weight is critical for task performance.
        """
        if not self._gradient_history:
            return {}
        
        fisher = {}
        
        # Get all weight names from first gradient
        weight_names = self._gradient_history[0].keys()
        
        for name in weight_names:
            # Collect all gradients for this weight
            grads = [g[name] for g in self._gradient_history if name in g]
            
            if grads:
                # Fisher = E[grad^2]
                squared_grads = [g ** 2 for g in grads]
                fisher[name] = np.mean(squared_grads, axis=0)
        
        return fisher
    
    def get_stats(self) -> Dict[str, float]:
        """Get learning statistics."""
        return {
            'learning_rate': self.config.learning_rate,
            'momentum': self.config.momentum,
            'step': self._step,
            'gradient_history_size': len(self._gradient_history),
        }
    
    def reset(self):
        """Reset momentum, step counter, and gradient history."""
        self._velocity = {}
        self._step = 0
        self._gradient_history = []
