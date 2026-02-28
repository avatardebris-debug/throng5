"""
Meta^0: Dual-Head ANN Layer (Unified Compute Graph)

Implements the DoorDash-style architecture with:
- Shared backbone for feature extraction
- Q-value head for action selection
- Reward prediction head for auxiliary learning

This replaces the SNN-based NeuronLayer from Throng3.
"""

import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class MetaLayer(ABC):
    """Abstract base class for all meta-layers (copied from throng3)."""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> Any:
        """Forward pass through the layer."""
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get layer weights for meta-learning."""
        pass


def xavier_init(n_in: int, n_out: int) -> np.ndarray:
    """Xavier/Glorot initialization for weights."""
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


class ANNLayer(MetaLayer):
    """
    Meta^0: Unified Compute Graph with Dual Heads
    
    Architecture:
        Input → Shared Backbone (ReLU) → ┬→ Q-value Head (n_actions)
                                          └→ Reward Prediction Head (1)
    
    Benefits:
    - Single forward pass yields both Q-values and reward estimates
    - Auxiliary learning signal improves feature quality
    - Better transfer learning (reward features generalize)
    - MAML can optimize strategies for both heads
    """
    
    def __init__(self, 
                 n_inputs: int,
                 n_hidden: int = 128,
                 n_outputs: int = 4):
        """
        Initialize dual-head ANN layer.
        
        Args:
            n_inputs: Input dimension (state size)
            n_hidden: Hidden layer size (shared backbone)
            n_outputs: Number of actions (Q-value head output size)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        # Shared backbone
        self.W1 = xavier_init(n_inputs, n_hidden)
        self.b1 = np.zeros(n_hidden)
        
        # Q-value head (for action selection)
        self.W_q = xavier_init(n_hidden, n_outputs)
        self.b_q = np.zeros(n_outputs)
        
        # Reward prediction head (auxiliary task)
        self.W_r = xavier_init(n_hidden, 1)
        self.b_r = np.zeros(1)
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Forward pass through dual-head network.
        
        Args:
            x: Input state (n_inputs,)
        
        Returns:
            Dict with:
                - 'q_values': Q-values for all actions (n_outputs,)
                - 'reward_pred': Predicted reward (scalar)
                - 'hidden': Hidden layer activations (for analysis)
        """
        # Shared backbone
        z1 = self.W1.T @ x + self.b1
        h = np.maximum(0, z1)  # ReLU activation
        
        # Clamp to prevent overflow
        h = np.clip(h, -1e6, 1e6)
        np.nan_to_num(h, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Q-value head
        q_values = self.W_q.T @ h + self.b_q
        np.nan_to_num(q_values, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Reward prediction head
        reward_pred = (self.W_r.T @ h + self.b_r)[0]
        if not np.isfinite(reward_pred):
            reward_pred = 0.0
        
        # Cache for backward pass
        self.cache = {
            'x': x.copy(),
            'z1': z1,
            'h': h,
            'q_values': q_values,
            'reward_pred': reward_pred
        }
        
        return {
            'q_values': q_values,
            'reward_pred': reward_pred,
            'hidden': h
        }
    
    def backward_q(self, td_error: float, action: int, lr: float = 0.001):
        """
        Backward pass for Q-learning loss.
        
        Args:
            td_error: Temporal difference error
            action: Action that was taken
            lr: Learning rate
        """
        # Gradient of Q-loss w.r.t. Q-values
        dq = np.zeros(self.n_outputs)
        dq[action] = -td_error  # Negative because we minimize (target - pred)^2
        
        # Backprop through Q-head
        dW_q = np.outer(self.cache['h'], dq)
        db_q = dq
        dh_q = self.W_q @ dq
        
        # Backprop through ReLU
        dz1 = dh_q * (self.cache['z1'] > 0)
        
        # Backprop through backbone
        dW1 = np.outer(self.cache['x'], dz1)
        db1 = dz1
        
        # Update weights
        self.W_q -= lr * dW_q
        self.b_q -= lr * db_q
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
    
    def backward_reward(self, reward_error: float, lr: float = 0.001, 
                       aux_weight: float = 0.1):
        """
        Backward pass for reward prediction loss (auxiliary task).
        
        Args:
            reward_error: Prediction error (actual - predicted)
            lr: Learning rate
            aux_weight: Weight for auxiliary loss
        """
        # Gradient of reward loss
        dr = -reward_error  # Negative because we minimize (target - pred)^2
        
        # Backprop through reward head
        dW_r = np.outer(self.cache['h'], [dr])
        db_r = np.array([dr])
        dh_r = self.W_r.flatten() * dr
        
        # Backprop through ReLU
        dz1 = dh_r * (self.cache['z1'] > 0)
        
        # Backprop through backbone
        dW1 = np.outer(self.cache['x'], dz1)
        db1 = dz1
        
        # Update weights (scaled by aux_weight)
        self.W_r -= lr * aux_weight * dW_r
        self.b_r -= lr * aux_weight * db_r
        self.W1 -= lr * aux_weight * dW1
        self.b1 -= lr * aux_weight * db1
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all layer weights for meta-learning."""
        return {
            'W1': self.W1.copy(),
            'b1': self.b1.copy(),
            'W_q': self.W_q.copy(),
            'b_q': self.b_q.copy(),
            'W_r': self.W_r.copy(),
            'b_r': self.b_r.copy()
        }
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set layer weights (for meta-learning)."""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W_q = weights['W_q'].copy()
        self.b_q = weights['b_q'].copy()
        self.W_r = weights['W_r'].copy()
        self.b_r = weights['b_r'].copy()
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return (self.W1.size + self.b1.size + 
                self.W_q.size + self.b_q.size +
                self.W_r.size + self.b_r.size)
    
    def __repr__(self) -> str:
        return (f"ANNLayer(inputs={self.n_inputs}, hidden={self.n_hidden}, "
                f"outputs={self.n_outputs}, params={self.get_num_parameters()})")
