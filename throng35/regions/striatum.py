"""
Striatum Region — Q-Learning for Goal-Directed Behavior

The Striatum region handles action selection using Q-learning.
It maintains its own action→reward→learn loop with proper RL timing.
"""

from typing import Any, Dict, Optional
import numpy as np

from throng35.regions.region_base import RegionBase
from throng35.learning.qlearning import QLearner, QLearningConfig


class StriatumRegion(RegionBase):
    """
    Striatum region implementing Q-learning.
    
    Key features:
    - Uses RAW observations (not neuron activations)
    - Proper RL timing: action → reward → learn
    - Independent from other regions' timing
    """
    
    def __init__(self, 
                 n_states: int,
                 n_actions: int,
                 config: Optional[QLearningConfig] = None):
        """
        Initialize Striatum region.
        
        Args:
            n_states: Dimension of observation space
            n_actions: Number of possible actions
            config: Q-learning configuration
        """
        super().__init__(region_name="Striatum")
        
        # Q-learning component
        self.qlearner = QLearner(
            n_states=n_states,
            n_actions=n_actions,
            config=config or QLearningConfig()
        )
        
        # State tracking for temporal credit assignment
        self.prev_state: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.last_td_error: float = 0.0
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in Striatum.
        
        Expected inputs:
            - 'raw_observation': Current environment observation
            - 'reward': Reward from previous action (if any)
            - 'done': Whether episode is complete
        
        Returns:
            - 'action': Selected action
            - 'q_values': Q-values for current state
            - 'td_error': TD error from update (for modulating other regions)
        """
        obs = region_input['raw_observation']
        reward = region_input.get('reward', 0.0)
        done = region_input.get('done', False)
        
        # Q-learning update (if we have previous state/action)
        if self.prev_state is not None and self.prev_action is not None:
            self.last_td_error = self.qlearner.update(
                self.prev_state,
                self.prev_action,
                reward,
                obs,
                done
            )
        
        # Select action for current state
        action = self.qlearner.select_action(obs, explore=True)
        q_values = self.qlearner.get_q_values(obs)
        
        # Store for next update
        self.prev_state = np.array(obs).copy()
        self.prev_action = action
        
        # Reset on episode end
        if done:
            self.qlearner.reset_episode()
            self.prev_state = None
            self.prev_action = None
        
        return {
            'action': action,
            'q_values': q_values,
            'td_error': self.last_td_error,
            'epsilon': self.qlearner.config.epsilon
        }
    
    def reset(self):
        """Reset Striatum for new episode."""
        self.qlearner.reset_episode()
        self.prev_state = None
        self.prev_action = None
        self.last_td_error = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Q-learning statistics."""
        stats = self.qlearner.get_stats()
        stats['region'] = 'Striatum'
        stats['last_td_error'] = self.last_td_error
        return stats
