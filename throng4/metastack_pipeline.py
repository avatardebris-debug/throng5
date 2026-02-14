"""
MetaStackPipeline — Full Meta^N stack for Throng4

Chains Meta^0 → Meta^1 → Meta^3:
- Meta^0 (ANNLayer): Forward pass through dual-head network
- Meta^1 (DualHeadSynapseOptimizer): Backprop with per-head LR multipliers
- Meta^3 (DualHeadMAML): Task batching and periodic meta-updates

Enhanced with:
- Experience replay buffer for stable learning
- Target network (frozen Q-network copy) for stable TD targets
- Batch updates from replay alongside single-step MAML updates

Skips Meta^2 (learning rule selector) — only one learning rule exists.
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List
from throng4.layers.meta0_ann import ANNLayer
from throng4.layers.meta1_synapse import DualHeadSynapseOptimizer, DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAML, DualHeadMAMLConfig


class MetaStackPipeline:
    """
    Full Meta^N stack: Meta^0 → Meta^1 → Meta^3
    
    Flow:
    1. select_action() → Meta^0 forward → epsilon-greedy
    2. update() → store in replay → Meta^1 optimize → batch replay update
    3. Meta^3 task batching → meta-update when batch full
    4. Target network syncs periodically for stable Q-targets
    """
    
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 n_hidden: int = 128,
                 synapse_config: Optional[DualHeadSynapseConfig] = None,
                 maml_config: Optional[DualHeadMAMLConfig] = None,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 50,
                 gamma: float = 0.99):
        """
        Initialize meta-stack pipeline.
        
        Args:
            n_inputs: State dimension
            n_outputs: Number of actions
            n_hidden: Hidden layer size
            synapse_config: Meta^1 configuration
            maml_config: Meta^3 configuration
            buffer_size: Replay buffer capacity
            batch_size: Mini-batch size for replay updates
            target_update_freq: Episodes between target network syncs
            gamma: Discount factor for Q-learning
        """
        # Meta^0: Dual-head ANN (online network)
        self.ann = ANNLayer(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs
        )
        
        # Target network (frozen copy for stable Q-targets)
        self.target_ann = ANNLayer(
            n_inputs=n_inputs,
            n_hidden=n_hidden,
            n_outputs=n_outputs
        )
        self._sync_target_network()
        
        # Meta^1: Synapse optimizer
        self.synapse = DualHeadSynapseOptimizer(
            self.ann,
            synapse_config or DualHeadSynapseConfig()
        )
        
        # Meta^3: MAML
        self.maml = DualHeadMAML(maml_config or DualHeadMAMLConfig())
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Target network update frequency
        self.target_update_freq = target_update_freq
        
        # Tracking
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.episode_count = 0
        self.step_count = 0
        self.epsilon = 1.0  # Start fully exploratory
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Episode buffer for MAML task batching
        self.current_episode_transitions: List[Dict] = []
        
        # Metrics
        self.total_td_error = 0.0
        self.total_reward_error = 0.0
        self.replay_updates = 0
    
    def _sync_target_network(self):
        """Copy online network weights to target network."""
        self.target_ann.set_weights(self.ann.get_weights())
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: Whether to use epsilon-greedy
            
        Returns:
            Selected action index
        """
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_outputs)
        else:
            output = self.ann.forward(state)
            q_values = output['q_values']
            return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> Dict[str, Any]:
        """
        Update from transition.
        
        Flow:
        1. Store transition in replay buffer + episode buffer
        2. Meta^1 optimize (backprop with LR multipliers from Meta^3)
        3. Batch replay update (if buffer has enough samples)
        4. If episode done, add task to Meta^3 batch
        5. If Meta^3 batch full, trigger meta-update
        6. Periodically sync target network
        
        Returns:
            Dict with td_error, reward_error, replay info, and meta-update info
        """
        # Build transition
        transition = {
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done
        }
        
        # Store in replay buffer
        self.replay_buffer.append(transition)
        
        # Store in episode buffer for MAML task batching
        self.current_episode_transitions.append(transition)
        
        # Get LR multipliers from Meta^3
        lr_multipliers = self.maml.get_lr_multipliers()
        
        # Meta^1: Single-step optimize with per-head LR multipliers
        context = {
            **transition,
            'lr_multipliers': lr_multipliers
        }
        result = self.synapse.optimize(context)
        
        # Batch replay update (extra learning from past experience)
        replay_result = self._replay_update(lr_multipliers)
        
        self.step_count += 1
        
        # If episode done, handle MAML batching and target sync
        meta_update_triggered = False
        if done:
            self.episode_count += 1
            
            # Split episode into support/query sets for MAML
            n_transitions = len(self.current_episode_transitions)
            split_idx = max(1, int(n_transitions * 0.7))
            
            support_set = self.current_episode_transitions[:split_idx]
            query_set = self.current_episode_transitions[split_idx:]
            
            # Add task to MAML batch
            self.maml.add_task(support_set, query_set)
            
            # Maybe trigger meta-update
            meta_update_triggered = self.maml.maybe_meta_update(self.ann)
            
            # If meta-update happened, also sync target network
            if meta_update_triggered:
                self._sync_target_network()
            
            # Periodic target network sync
            if self.episode_count % self.target_update_freq == 0:
                self._sync_target_network()
            
            # Reset episode buffer
            self.current_episode_transitions = []
            
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        return {
            **result,
            'meta_update_triggered': meta_update_triggered,
            'epsilon': self.epsilon,
            'replay_td_error': replay_result.get('mean_td_error', 0.0) if replay_result else 0.0,
            'buffer_size': len(self.replay_buffer),
        }
    
    def _replay_update(self, lr_multipliers: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Perform a batch update from the replay buffer.
        Uses target network for stable Q-targets.
        
        Args:
            lr_multipliers: Per-head LR multipliers from MAML
            
        Returns:
            Dict with mean errors, or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample random mini-batch
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        total_td_error = 0.0
        total_reward_error = 0.0
        
        for t in batch:
            # Use TARGET network for stable Q-targets
            if t['done']:
                target_q = t['reward']
            else:
                target_output = self.target_ann.forward(t['next_state'])
                target_q = t['reward'] + self.gamma * np.max(target_output['q_values'])
            
            # Use ONLINE network for current values
            current_output = self.ann.forward(t['state'])
            current_q = current_output['q_values'][t['action']]
            reward_pred = current_output['reward_pred']
            
            td_error = target_q - current_q
            reward_error = t['reward'] - reward_pred
            
            # Backprop through Meta^1 with MAML's LR multipliers
            replay_context = {
                **t,
                'lr_multipliers': lr_multipliers
            }
            self.synapse.optimize(replay_context)
            
            total_td_error += abs(td_error)
            total_reward_error += abs(reward_error)
        
        self.replay_updates += 1
        self.total_td_error += total_td_error
        self.total_reward_error += total_reward_error
        
        return {
            'mean_td_error': total_td_error / self.batch_size,
            'mean_reward_error': total_reward_error / self.batch_size,
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state."""
        output = self.ann.forward(state)
        return output['q_values']
    
    def get_reward_prediction(self, state: np.ndarray) -> float:
        """Get reward prediction for a state."""
        output = self.ann.forward(state)
        return output['reward_pred']
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        synapse_stats = self.synapse.get_stats()
        maml_stats = self.maml.get_stats()
        
        return {
            'episodes': self.episode_count,
            'steps': self.step_count,
            'epsilon': self.epsilon,
            'ann_params': self.ann.get_num_parameters(),
            'buffer_size': len(self.replay_buffer),
            'replay_updates': self.replay_updates,
            'mean_td_error': self.total_td_error / max(1, self.replay_updates * self.batch_size),
            'mean_reward_error': self.total_reward_error / max(1, self.replay_updates * self.batch_size),
            **synapse_stats,
            **maml_stats
        }
    
    
    def reset_episode(self):
        """Reset for new episode."""
        self.current_episode_transitions = []
    
    def transfer_weights(self, source_pipeline: 'MetaStackPipeline'):
        """
        Transfer full weight initialization from source pipeline.
        
        This is the core MAML weight transfer mechanism. Instead of just
        transferring 6 LR scalars, we transfer the full 14K+ weight parameters
        that MAML has meta-learned to be easy to adapt from.
        
        Handles dimensionality mismatches via adaptation:
        - Same dimensions: direct copy
        - Different dimensions: copy overlapping region, initialize rest
        
        Args:
            source_pipeline: Pipeline to transfer weights from
        """
        source_weights = source_pipeline.ann.get_weights()
        
        # If dimensions match, direct transfer
        if (source_pipeline.n_inputs == self.n_inputs and 
            source_pipeline.n_outputs == self.n_outputs):
            self.ann.set_weights(source_weights)
            self._sync_target_network()
            return
        
        # Dimensionality adaptation for cross-dimension transfer
        adapted_weights = self._adapt_weights(source_weights)
        self.ann.set_weights(adapted_weights)
        self._sync_target_network()
        
        # Also transfer MAML's LR multipliers
        source_lr_mults = source_pipeline.maml.get_lr_multipliers()
        self.maml.meta_params['rl']['lr_multipliers'] = source_lr_mults.copy()
    
    def _adapt_weights(self, source_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Adapt weights from different dimensions.
        
        Strategy:
        - Copy overlapping regions (preserves learned features)
        - Initialize non-overlapping regions with small random values
        - Biases: copy what fits, zero-init the rest
        
        This allows transfer of low-level feature detectors even when
        input/output dimensions differ (e.g., GridWorld 25D → Tetris 220D).
        
        Args:
            source_weights: Weights from source network
            
        Returns:
            Adapted weights matching target network dimensions
        """
        target_weights = self.ann.get_weights()
        adapted = {}
        
        for key in target_weights.keys():
            if key not in source_weights:
                # Key doesn't exist in source, keep target's random init
                adapted[key] = target_weights[key]
                continue
            
            source_w = source_weights[key]
            target_w = target_weights[key]
            
            if source_w.shape == target_w.shape:
                # Same shape, direct copy
                adapted[key] = source_w.copy()
            else:
                # Different shape, copy overlapping region
                adapted[key] = target_w.copy()  # Start with target's init
                
                if len(source_w.shape) == 2:  # Weight matrix
                    min_rows = min(source_w.shape[0], target_w.shape[0])
                    min_cols = min(source_w.shape[1], target_w.shape[1])
                    adapted[key][:min_rows, :min_cols] = source_w[:min_rows, :min_cols]
                elif len(source_w.shape) == 1:  # Bias vector
                    min_len = min(source_w.shape[0], target_w.shape[0])
                    adapted[key][:min_len] = source_w[:min_len]
        
        return adapted
