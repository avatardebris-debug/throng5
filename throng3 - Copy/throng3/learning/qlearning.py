"""
Q-Learning with Linear Function Approximation

Implements Q-learning for reinforcement learning with temporal credit assignment.
Uses linear function approximation: Q(s,a) ≈ w^T φ(s,a)
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class QLearningConfig:
    """Configuration for Q-learning."""
    learning_rate: float = 0.1
    gamma: float = 0.95  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    epsilon_decay: float = 0.995  # Decay per episode
    epsilon_min: float = 0.01
    
    # Regularization
    l2_reg: float = 0.0001  # Weight decay
    clip_td_error: float = 10.0  # Clip TD error for stability


class QLearner:
    """
    Q-learning with linear function approximation.
    
    Q(s,a) = W[a] @ s
    
    where:
    - s is the state vector (neuron activations)
    - a is the action index
    - W is the weight matrix (n_actions × n_states)
    """
    
    def __init__(self, n_states: int, n_actions: int, config: Optional[QLearningConfig] = None):
        """
        Initialize Q-learner.
        
        Args:
            n_states: Dimension of state vector
            n_actions: Number of discrete actions
            config: Q-learning configuration
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config or QLearningConfig()
        
        # Q-function weights: Q(s,a) = W[a] @ s
        self.W = np.zeros((n_actions, n_states), dtype=np.float32)
        
        # Initialize with small random weights for symmetry breaking
        self.W += np.random.randn(n_actions, n_states) * 0.01
        
        # Exploration rate (decays over time)
        self.epsilon = self.config.epsilon
        
        # Statistics
        self.n_updates = 0
        self.total_td_error = 0.0
        self.episode_count = 0
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q(s,a) for all actions.
        
        Args:
            state: State vector (n_states,)
            
        Returns:
            Q-values for each action (n_actions,)
        """
        # Ensure state is the right size
        if len(state) < self.n_states:
            padded = np.zeros(self.n_states, dtype=np.float32)
            padded[:len(state)] = state
            state = padded
        elif len(state) > self.n_states:
            state = state[:self.n_states]
        
        return self.W @ state
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: State vector
            explore: If True, use ε-greedy. If False, always greedy.
            
        Returns:
            Selected action index
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> float:
        """
        Q-learning update.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            TD error (for diagnostics)
        """
        # Ensure states are the right size
        if len(state) < self.n_states:
            padded = np.zeros(self.n_states, dtype=np.float32)
            padded[:len(state)] = state
            state = padded
        elif len(state) > self.n_states:
            state = state[:self.n_states]
        
        if len(next_state) < self.n_states:
            padded = np.zeros(self.n_states, dtype=np.float32)
            padded[:len(next_state)] = next_state
            next_state = padded
        elif len(next_state) > self.n_states:
            next_state = next_state[:self.n_states]
        
        # Current Q-value
        q_current = self.W[action] @ state
        
        # TD target
        if done:
            td_target = reward
        else:
            q_next_max = np.max(self.W @ next_state)
            td_target = reward + self.config.gamma * q_next_max
        
        # TD error
        td_error = td_target - q_current
        
        # Clip TD error for stability
        td_error = np.clip(td_error, -self.config.clip_td_error, self.config.clip_td_error)
        
        # Gradient descent update: W[a] += α * td_error * s
        self.W[action] += self.config.learning_rate * td_error * state
        
        # L2 regularization (weight decay)
        if self.config.l2_reg > 0:
            self.W[action] *= (1 - self.config.l2_reg)
        
        # Update statistics
        self.n_updates += 1
        self.total_td_error += abs(td_error)
        
        return td_error
    
    def decay_epsilon(self):
        """Decay exploration rate (call at end of episode)."""
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        self.episode_count += 1
    
    def reset_episode(self):
        """Reset episode-specific state."""
        self.decay_epsilon()
    
    def get_stats(self) -> dict:
        """Get learning statistics."""
        avg_td_error = self.total_td_error / max(self.n_updates, 1)
        return {
            'epsilon': self.epsilon,
            'n_updates': self.n_updates,
            'avg_td_error': avg_td_error,
            'episode_count': self.episode_count,
            'mean_q_weight': np.mean(np.abs(self.W)),
            'max_q_weight': np.max(np.abs(self.W)),
        }
    
    def save_weights(self, path: str):
        """Save Q-function weights."""
        np.save(path, self.W)
    
    def load_weights(self, path: str):
        """Load Q-function weights."""
        self.W = np.load(path)
