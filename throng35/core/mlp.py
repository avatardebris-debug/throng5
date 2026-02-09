"""
Simple Multi-Layer Perceptron for Q-Learning

Replaces linear Q-function with non-linear neural network.
"""

import numpy as np
from typing import Tuple, Optional


class SimpleMLP:
    """
    Simple 2-layer MLP for Q-function approximation.
    
    Architecture: input → hidden1 (ReLU) → hidden2 (ReLU) → output
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: Tuple[int, int] = (64, 64),
                 output_dim: int = 4,
                 learning_rate: float = 0.01):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer sizes
            output_dim: Number of actions
            learning_rate: Learning rate for gradient descent
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dims[0])
        
        self.W2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        
        self.W3 = np.random.randn(hidden_dims[1], output_dim) * np.sqrt(2.0 / hidden_dims[1])
        self.b3 = np.zeros(output_dim)
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.
        
        Args:
            x: Input features (input_dim,)
        
        Returns:
            Q-values for all actions (output_dim,)
        """
        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        q_values = a2 @ self.W3 + self.b3
        
        # Cache for backward pass
        self.cache = {
            'x': x,
            'z1': z1, 'a1': a1,
            'z2': z2, 'a2': a2,
            'q_values': q_values
        }
        
        return q_values
    
    def backward(self, target_q: float, action: int) -> None:
        """
        Backward pass - update weights using gradient descent.
        
        Args:
            target_q: Target Q-value for the action taken
            action: Action that was taken
        """
        # Compute TD error
        predicted_q = self.cache['q_values'][action]
        td_error = target_q - predicted_q
        
        # Gradient of loss w.r.t. output
        dq = np.zeros(self.output_dim)
        dq[action] = -td_error  # Negative because we minimize (target - pred)^2
        
        # Backprop through layer 3
        dW3 = np.outer(self.cache['a2'], dq)
        db3 = dq
        da2 = dq @ self.W3.T
        
        # Backprop through ReLU
        dz2 = da2 * (self.cache['z2'] > 0)
        
        # Backprop through layer 2
        dW2 = np.outer(self.cache['a1'], dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T
        
        # Backprop through ReLU
        dz1 = da1 * (self.cache['z1'] > 0)
        
        # Backprop through layer 1
        dW1 = np.outer(self.cache['x'], dz1)
        db1 = dz1
        
        # Update weights
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        return self.forward(state)
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size + 
                self.W3.size + self.b3.size)
