"""
DQN Learner with Reward Prediction

Implements Q-learning through the dual-head ANN with:
- Experience replay for stable learning
- Dual loss: TD error + reward prediction error
- Epsilon-greedy exploration
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Configuration for DQN learner."""
    buffer_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    learning_rate: float = 0.001
    aux_loss_weight: float = 0.1  # Weight for reward prediction loss


class DQNLearner:
    """
    Q-learning through dual-head ANN with experience replay.
    
    Features:
    - Experience replay buffer for stable learning
    - Dual loss: Q-learning + reward prediction
    - Epsilon-greedy exploration with decay
    """
    
    def __init__(self, ann_layer, config: Optional[DQNConfig] = None):
        """
        Initialize DQN learner.
        
        Args:
            ann_layer: ANNLayer instance (dual-head network)
            config: DQN configuration
        """
        self.ann = ann_layer
        self.config = config or DQNConfig()
        
        # Experience replay buffer
        self.buffer = deque(maxlen=self.config.buffer_size)
        
        # Exploration
        self.epsilon = self.config.epsilon
        
        # Metrics
        self.n_updates = 0
        self.total_td_error = 0.0
        self.total_reward_error = 0.0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use epsilon-greedy (False = greedy only)
        
        Returns:
            Selected action index
        """
        if explore and np.random.rand() < self.epsilon:
            # Random exploration
            return np.random.randint(self.ann.n_outputs)
        else:
            # Greedy action
            output = self.ann.forward(state)
            q_values = output['q_values']
            return int(np.argmax(q_values))
    
    def store_transition(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.buffer.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        })
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Single-step update (for online learning).
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state
            done: Episode termination flag
        
        Returns:
            Dict with td_error and reward_error
        """
        # Store in buffer
        self.store_transition(state, action, reward, next_state, done)
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            next_output = self.ann.forward(next_state)
            target_q = reward + self.config.gamma * np.max(next_output['q_values'])
        
        # Get current Q-value and reward prediction
        current_output = self.ann.forward(state)
        current_q = current_output['q_values'][action]
        reward_pred = current_output['reward_pred']
        
        # Compute errors
        td_error = target_q - current_q
        reward_error = reward - reward_pred
        
        # Backprop both losses
        self.ann.backward_q(td_error, action, lr=self.config.learning_rate)
        self.ann.backward_reward(reward_error, lr=self.config.learning_rate,
                                aux_weight=self.config.aux_loss_weight)
        
        # Update metrics
        self.n_updates += 1
        self.total_td_error += abs(td_error)
        self.total_reward_error += abs(reward_error)
        
        # Decay epsilon
        if done:
            self.epsilon = max(self.epsilon * self.config.epsilon_decay,
                             self.config.epsilon_min)
        
        return {
            'td_error': td_error,
            'reward_error': reward_error
        }
    
    def batch_update(self) -> Optional[Dict[str, float]]:
        """
        Batch update using experience replay.
        
        Returns:
            Dict with mean td_error and reward_error, or None if buffer too small
        """
        if len(self.buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.config.batch_size, 
                                  replace=False)
        batch = [self.buffer[i] for i in indices]
        
        total_td_error = 0.0
        total_reward_error = 0.0
        
        for transition in batch:
            # Compute target
            if transition['done']:
                target_q = transition['reward']
            else:
                next_output = self.ann.forward(transition['next_state'])
                target_q = (transition['reward'] + 
                           self.config.gamma * np.max(next_output['q_values']))
            
            # Get current values
            current_output = self.ann.forward(transition['state'])
            current_q = current_output['q_values'][transition['action']]
            reward_pred = current_output['reward_pred']
            
            # Compute errors
            td_error = target_q - current_q
            reward_error = transition['reward'] - reward_pred
            
            # Backprop
            self.ann.backward_q(td_error, transition['action'], 
                              lr=self.config.learning_rate)
            self.ann.backward_reward(reward_error, lr=self.config.learning_rate,
                                   aux_weight=self.config.aux_loss_weight)
            
            total_td_error += abs(td_error)
            total_reward_error += abs(reward_error)
        
        self.n_updates += 1
        
        return {
            'mean_td_error': total_td_error / self.config.batch_size,
            'mean_reward_error': total_reward_error / self.config.batch_size
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        output = self.ann.forward(state)
        return output['q_values']
    
    def get_reward_prediction(self, state: np.ndarray) -> float:
        """Get reward prediction for a state."""
        output = self.ann.forward(state)
        return output['reward_pred']
    
    def reset_episode(self):
        """Reset episode-specific state."""
        # Epsilon decay happens in update() on done=True
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            'n_updates': self.n_updates,
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer),
            'mean_td_error': (self.total_td_error / max(1, self.n_updates)),
            'mean_reward_error': (self.total_reward_error / max(1, self.n_updates))
        }
