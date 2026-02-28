"""
Neural Q-Learner - Uses MLP for non-linear function approximation

Replaces linear Q-learning with neural network.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from throng35.core.mlp import SimpleMLP


@dataclass
class NeuralQLearningConfig:
    """Configuration for neural Q-learning."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    hidden_dims: tuple = (64, 64)


class NeuralQLearner:
    """
    Q-learning with neural network function approximation.
    
    Uses MLP to approximate Q(s,a) instead of linear function.
    """
    
    def __init__(self, 
                 n_states: int,
                 n_actions: int,
                 config: Optional[NeuralQLearningConfig] = None):
        """
        Initialize neural Q-learner.
        
        Args:
            n_states: State feature dimension
            n_actions: Number of actions
            config: Configuration
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config or NeuralQLearningConfig()
        
        # Neural network for Q-function
        self.q_network = SimpleMLP(
            input_dim=n_states,
            hidden_dims=self.config.hidden_dims,
            output_dim=n_actions,
            learning_rate=self.config.learning_rate
        )
        
        # Exploration
        self.epsilon = self.config.epsilon
        
        # Statistics
        self.n_updates = 0
        self.total_td_error = 0.0
        self.episode_count = 0
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state features
        
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_network.get_q_values(state)
            return int(np.argmax(q_values))
    
    def update(self,
               state: np.ndarray,
               action: int,
               reward: float,
               next_state: np.ndarray,
               done: bool) -> float:
        """
        Update Q-network using Q-learning update.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        
        Returns:
            TD error
        """
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.q_network.get_q_values(next_state)
            target_q = reward + self.config.gamma * np.max(next_q_values)
        
        # Get current Q-value prediction
        current_q_values = self.q_network.forward(state)
        current_q = current_q_values[action]
        
        # TD error
        td_error = target_q - current_q
        
        # Update network via backprop
        self.q_network.backward(target_q, action)
        
        # Update statistics
        self.n_updates += 1
        self.total_td_error += abs(td_error)
        
        return td_error
    
    def end_episode(self):
        """Called at end of episode."""
        self.episode_count += 1
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
    
    def get_stats(self) -> dict:
        """Get learner statistics."""
        return {
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'avg_td_error': self.total_td_error / max(1, self.n_updates),
            'n_params': self.q_network.get_num_parameters()
        }
