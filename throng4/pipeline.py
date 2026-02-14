"""
Simple Pipeline Wrapper for Throng4

Provides a minimal interface for running the dual-head ANN on RL tasks.
This is a simplified version of throng3's MetaNPipeline, focused on
getting the dual-head architecture working with GridWorld.
"""

import numpy as np
from typing import Dict, Any, Optional
from throng4.layers.meta0_ann import ANNLayer
from throng4.learning.dqn import DQNLearner, DQNConfig


class SimplePipeline:
    """
    Minimal pipeline for Throng4 dual-head ANN.
    
    This is a lightweight wrapper that doesn't use the full Meta^N stack yet.
    Focus: Get dual-head ANN + DQN working on GridWorld.
    """
    
    def __init__(self, 
                 n_inputs: int,
                 n_outputs: int,
                 n_hidden: int = 128,
                 config: Optional[DQNConfig] = None):
        """
        Initialize simple pipeline.
        
        Args:
            n_inputs: State dimension
            n_outputs: Number of actions
            n_hidden: Hidden layer size
            config: DQN configuration
        """
        # Create dual-head ANN
        self.ann = ANNLayer(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs
        )
        
        # Create DQN learner
        self.learner = DQNLearner(self.ann, config or DQNConfig())
        
        # Tracking
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.episode_count = 0
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        return self.learner.select_action(state, explore=explore)
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update from transition.
        
        Returns:
            Dict with td_error and reward_error
        """
        errors = self.learner.update(state, action, reward, next_state, done)
        self.step_count += 1
        
        if done:
            self.episode_count += 1
        
        return errors
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        return self.learner.get_q_values(state)
    
    def get_reward_prediction(self, state: np.ndarray) -> float:
        """Get reward prediction for a state."""
        return self.learner.get_reward_prediction(state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        learner_stats = self.learner.get_stats()
        return {
            **learner_stats,
            'episodes': self.episode_count,
            'steps': self.step_count,
            'ann_params': self.ann.get_num_parameters()
        }
    
    def reset_episode(self):
        """Reset for new episode."""
        self.learner.reset_episode()
