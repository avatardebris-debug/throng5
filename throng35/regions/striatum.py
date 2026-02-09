"""
Striatum Region — Q-Learning for Goal-Directed Behavior

The Striatum region handles action selection using Q-learning.
It maintains its own action→reward→learn loop with proper RL timing.
"""

from typing import Any, Dict, Optional
import numpy as np
import time
import sys

from throng35.regions.region_base import RegionBase
from throng35.learning.qlearning import QLearner, QLearningConfig


class StriatumRegion(RegionBase):
    """
    Striatum region implementing Q-learning.
    
    Key features:
    - Learns from raw observations (not neuron activations)
    - Proper RL timing (action → reward → learn)
    - Independent Q-learning loop
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
        
        # Resource tracking (for future optimization)
        self._step_times = []
        self._update_counts = []
        self.n_states = n_states
        self.n_actions = n_actions
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in Striatum.
        
        Expected inputs:
            - 'raw_observation': Raw environment observation
            - 'reward': Reward from previous action
            - 'done': Whether episode is complete
        
        Returns:
            - 'action': Selected action
            - 'q_values': Q-values for current state
            - 'td_error': Temporal difference error
            - 'epsilon': Current exploration rate
        """
        t0 = time.time()  # Resource tracking
        
        obs = region_input['raw_observation']
        reward = region_input.get('reward', 0.0)
        done = region_input.get('done', False)
        
        # Q-learning update (if we have previous state/action)
        did_update = False
        if self.prev_state is not None and self.prev_action is not None:
            self.last_td_error = self.qlearner.update(
                self.prev_state,
                self.prev_action,
                reward,
                obs,
                done
            )
            did_update = True
        
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
        
        # Resource tracking
        self._step_times.append(time.time() - t0)
        self._update_counts.append(1 if did_update else 0)
        if len(self._step_times) > 100:  # Keep last 100
            self._step_times.pop(0)
            self._update_counts.pop(0)
        
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
    
    def get_state_signature(self) -> Dict[str, Any]:
        """Return input/output signature for this region."""
        return {
            'inputs': {
                'raw_observation': {
                    'type': np.ndarray,
                    'required': True,
                    'shape': (self.n_states,),
                    'description': 'Raw environment observation'
                },
                'reward': {
                    'type': float,
                    'required': False,
                    'description': 'Reward from previous action'
                },
                'done': {
                    'type': bool,
                    'required': False,
                    'description': 'Episode termination flag'
                }
            },
            'outputs': {
                'action': {
                    'type': int,
                    'description': 'Selected action index'
                },
                'q_values': {
                    'type': np.ndarray,
                    'shape': (self.n_actions,),
                    'description': 'Q-values for all actions'
                },
                'td_error': {
                    'type': float,
                    'description': 'Temporal difference error'
                },
                'epsilon': {
                    'type': float,
                    'description': 'Current exploration rate'
                }
            }
        }
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Return resource usage metrics (enables future optimization)."""
        return {
            'compute_ms': np.mean(self._step_times) * 1000 if self._step_times else 0.0,
            'memory_mb': sys.getsizeof(self.qlearner.W) / 1024**2,
            'updates_per_step': np.mean(self._update_counts) if self._update_counts else 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Striatum statistics."""
        return {
            'region': 'Striatum',
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'epsilon': self.qlearner.config.epsilon,
            'n_updates': self.qlearner.n_updates,
            'mean_q_weight': np.mean(np.abs(self.qlearner.W)),
            'resource_usage': self.get_resource_usage()
        }
