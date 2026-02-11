"""
Meta^1: SynapseOptimizer — Self-Tuning via STDP/Hebbian/Pruning

Operates on Meta^0's weights. Applies learning rules to modify
the connection strengths between neurons.

Responsibilities:
- Apply STDP learning based on spike timing
- Apply Hebbian learning based on activity correlations
- Prune weak connections (Nash pruning)
- Manage eligibility traces for three-factor learning
- Report weight statistics UP to Meta^2

This layer is told WHICH learning rule to use by Meta^2,
but it decides HOW to apply it (rates, timing, etc.).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority
from throng3.learning.stdp import STDPRule, STDPConfig
from throng3.learning.hebbian import HebbianRule, HebbianConfig
from throng3.learning.qlearning import QLearner, QLearningConfig
from throng3.learning.dopamine import DopamineSystem, DopamineConfig
from throng3.learning.pruning import NashPruner, PruningConfig
from throng3.learning.gradient import GradientLearner, GradientConfig


@dataclass
class SynapseConfig:
    """Configuration for Meta^1 synapse optimization."""
    # Learning mechanism selection
    learning_mode: str = 'auto'  # 'auto', 'gradient', 'rl', 'hybrid'
    
    # Learning rules
    default_rule: str = 'both'  # 'stdp', 'hebbian', or 'both'
    rule_selection_mode: str = 'fixed'  # 'fixed' or 'bandit'
    evaluation_window: int = 100
    
    # STDP parameters
    stdp_a_plus: float = 0.01
    stdp_a_minus: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    
    # Hebbian parameters
    hebbian_lr: float = 0.001
    hebbian_normalize: bool = True
    
    # Dopamine modulation
    dopamine_decay: float = 0.95
    dopamine_sensitivity: float = 0.1
    
    # Q-learning
    use_qlearning: bool = True
    q_learning_rate: float = 0.1
    q_gamma: float = 0.95
    q_epsilon: float = 0.1
    q_epsilon_decay: float = 0.995
    q_epsilon_min: float = 0.01
    
    # General parameters
    learning_rate: float = 0.01
    prune_interval: int = 100        # Steps between pruning
    dopamine_modulation: bool = True
    weight_clip: float = 5.0
    consolidation_threshold: float = 0.8  # Stability needed to consolidate


class SynapseOptimizer(MetaLayer):
    """
    Meta^1: Synapse/weight self-tuning.
    
    Applies learning rules from Meta^2 to modify Meta^0's weights.
    Manages the interplay between STDP, Hebbian, and pruning.
    """
    
    def __init__(self, config: Optional[SynapseConfig] = None, **kwargs):
        cfg = config or SynapseConfig()
        super().__init__(level=1, name="SynapseOptimizer", config=vars(cfg))
        self.synapse_config = cfg
        
        # Learning rules (all available, Meta^2 selects which to use)
        self.stdp = STDPRule()
        self.hebbian = HebbianRule()
        self.dopamine = DopamineSystem()
        self.pruner = NashPruner()
        self.gradient = GradientLearner(GradientConfig())
        
        # Current active rule
        self.active_rule = cfg.default_rule  # 'stdp', 'hebbian', 'both'
        
        # Learning mode (gradient vs RL vs hybrid)
        self.learning_mode = cfg.learning_mode
        
        # Weight change accumulator (applied to Meta^0)
        self._pending_dW: Dict[str, np.ndarray] = {}
        
        # Tracking
        self._total_weight_updates = 0
        self._rule_usage: Dict[str, int] = {'stdp': 0, 'hebbian': 0, 'both': 0}
        self._last_weight_stats: Dict[str, float] = {}
        
        # Reference to Meta^0's state (populated during optimize)
        self._neuron_state: Optional[Dict] = None
        
        # Spike history for STDP temporal offset
        self._prev_spikes: Optional[np.ndarray] = None
        self._prev_activations: Optional[np.ndarray] = None
        
        # Hebbian temporal history
        self._hebbian_prev_activations: Optional[np.ndarray] = None
        
        # Q-learning (optional, for RL tasks)
        self.qlearner: Optional[QLearner] = None
        if self.synapse_config.use_qlearning:
            # Will be initialized when we know n_states and n_actions
            self._qlearner_config = QLearningConfig(
                learning_rate=self.synapse_config.q_learning_rate,
                gamma=self.synapse_config.q_gamma,
                epsilon=self.synapse_config.q_epsilon,
                epsilon_decay=self.synapse_config.q_epsilon_decay,
                epsilon_min=self.synapse_config.q_epsilon_min,
            )
            self._q_prev_state: Optional[np.ndarray] = None
            self._q_prev_action: Optional[int] = None
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one synapse optimization step.
        
        1. Get current weights and activations from Meta^0
        2. Apply active learning rule(s)
        3. Apply dopamine modulation
        4. Periodic pruning
        5. Signal weight change statistics UP
        """
        self.process_inbox()
        
        # Get Meta^0 state from context
        weights = context.get('weights', {})
        activations = context.get('activations', np.array([]))
        spikes = context.get('spikes', np.array([]))
        reward = context.get('reward', 0.0)
        
        W_recurrent = weights.get('W_recurrent', None)
        if W_recurrent is None:
            return {
                'loss': self.metrics.loss,
                'dW': {},
                'rpe': 0.0,
                'active_rule': self.active_rule,
                'metrics': self.metrics,
            }
        
        # Compute dopamine RPE (this will be updated by the new dopamine.update call)
        rpe = self.dopamine.compute_rpe(reward) # Keep for now, but new update will handle it
        
        # Apply learning based on mode
        dW = np.zeros_like(W_recurrent)
        
        if self.learning_mode == 'gradient':
            # Gradient-based learning for supervised tasks
            target = context.get('target')
            output = context.get('output')
            
            if target is not None and output is not None:
                # Get LR multipliers from Meta^3 (Fisher boosting)
                lr_multipliers = context.get('lr_multipliers', {})
                
                dW_dict = self.gradient.compute_weight_update(
                    weights, output, target, activations,
                    lr_multipliers=lr_multipliers
                )
                # Extract recurrent weight updates
                if 'W_recurrent' in dW_dict:
                    dW = dW_dict['W_recurrent']
                # Store output weight updates separately
                if 'W_out' in dW_dict:
                    self._pending_dW['W_out'] = dW_dict['W_out']
        
        elif self.learning_mode == 'rl':
            # Bio-inspired learning for RL tasks
            if self.active_rule != 'none':
                if self.active_rule in ('stdp', 'both'):
                    dW_stdp = self._apply_stdp(W_recurrent, activations)
                    dW += dW_stdp
                    self._rule_usage['stdp'] += 1
                
                if self.active_rule in ('hebbian', 'both'):
                    dW_hebb = self._apply_hebbian(W_recurrent, activations, reward)
                    dW += dW_hebb
                    self._rule_usage['hebbian'] += 1
        
        elif self.learning_mode == 'hybrid':
            # Hybrid: gradient + bio-inspired
            target = context.get('target')
            output = context.get('output')
            
            # Gradient component
            if target is not None and output is not None:
                dW_dict = self.gradient.compute_weight_update(
                    weights, output, target, activations
                )
                if 'W_recurrent' in dW_dict:
                    dW += 0.7 * dW_dict['W_recurrent']  # 70% gradient
                if 'W_out' in dW_dict:
                    self._pending_dW['W_out'] = dW_dict['W_out']
            
            # Bio-inspired component
            if self.active_rule != 'none':
                if self.active_rule in ('stdp', 'both'):
                    dW_stdp = self._apply_stdp(W_recurrent, activations)
                    dW += 0.3 * dW_stdp  # 30% bio-inspired
                    self._rule_usage['stdp'] += 1
                
                if self.active_rule in ('hebbian', 'both'):
                    dW_hebb = self._apply_hebbian(W_recurrent, activations, reward)
                    dW += 0.3 * dW_hebb
                    self._rule_usage['hebbian'] += 1
        
        else:  # 'auto' or unknown - default to bio-inspired for now
            # Only apply bio-inspired rules if not 'none'
            if self.active_rule != 'none':
                if self.active_rule in ('stdp', 'both'):
                    dW_stdp = self._apply_stdp(W_recurrent, activations)
                    dW += dW_stdp
                    self._rule_usage['stdp'] += 1
                
                if self.active_rule in ('hebbian', 'both'):
                    dW_hebb = self._apply_hebbian(W_recurrent, activations, reward)
                    dW += dW_hebb
                    self._rule_usage['hebbian'] += 1
        
        # Dopamine modulation
        if self.synapse_config.dopamine_modulation:
            dopamine_level = self.dopamine.compute_rpe(reward)
            modulated_lr = self.dopamine.modulate_learning_rate(self.synapse_config.learning_rate)
            dW *= modulated_lr / max(self.synapse_config.learning_rate, 1e-8)
        
        # Q-learning update (if enabled)
        td_error = 0.0
        q_values = None
        if self.synapse_config.use_qlearning:
            # CRITICAL FIX: Use raw observations, NOT neuron activations!
            # Q-learning needs consistent, meaningful state representation
            raw_obs = context.get('raw_observation', None)
            if raw_obs is None:
                # Fallback: use input if available (pipeline provides 'input', not 'input_data')
                raw_obs = context.get('input', activations)
            
            # Initialize Q-learner on first call with RAW observation dimension
            if self.qlearner is None:
                n_states = len(raw_obs)  # Use raw obs dim (e.g., 2 for [x,y])
                n_actions = context.get('n_outputs', 4)
                self.qlearner = QLearner(n_states, n_actions, self._qlearner_config)
            
            # Get current state (raw observation) and action
            current_state = raw_obs
            current_action = context.get('action', 0)  # Action taken
            done = context.get('done', False)
            
            # Q-learning update (if we have previous state)
            if self._q_prev_state is not None and self._q_prev_action is not None:
                td_error = self.qlearner.update(
                    self._q_prev_state,
                    self._q_prev_action,
                    reward,
                    current_state,
                    done
                )
                
                # Use TD error to modulate bio-inspired learning
                # Positive TD error = better than expected, strengthen
                # Negative TD error = worse than expected, weaken
                dW *= (1.0 + 0.1 * np.clip(td_error, -1, 1))
            
            # Store for next update (copy to avoid reference issues)
            self._q_prev_state = np.array(current_state).copy()
            self._q_prev_action = current_action
            
            # Reset on episode end
            if done:
                self.qlearner.reset_episode()
                self._q_prev_state = None
                self._q_prev_action = None
            
            # Get Q-values for action selection (using raw obs)
            q_values = self.qlearner.get_q_values(current_state)
        
        # Clip weight changes
        dW = np.clip(dW, -0.1, 0.1)
        
        # Store pending weight change for Meta^0
        self._pending_dW['W_recurrent'] = dW
        
        # Periodic pruning
        if self._optimization_step % self.synapse_config.prune_interval == 0:
            pruned_W, prune_stats = self.pruner.prune(
                W_recurrent + dW, activations
            )
            self._pending_dW['W_recurrent'] = pruned_W - W_recurrent
            
            # Signal pruning results UP
            self.signal(
                direction=SignalDirection.UP,
                signal_type=SignalType.PRUNE,
                payload=prune_stats,
            )
        
        # Compute metrics
        total_change = float(np.mean(np.abs(dW)))
        weight_magnitude = float(np.mean(np.abs(W_recurrent)))
        
        loss = 1.0 - min(total_change / max(weight_magnitude, 1e-8), 1.0)
        accuracy = float(np.mean(np.abs(dW) > 1e-6))  # Fraction of weights updated
        
        self.metrics.update(loss, accuracy)
        self.metrics.n_active_connections = int(np.sum(np.abs(W_recurrent + dW) > 1e-8))
        self._total_weight_updates += 1
        
        # Send suggestion DOWN to Meta^0 to apply weight changes
        weight_deltas = {'W_recurrent_delta': dW}
        
        # Include W_out delta if we have it (from gradient learning)
        if 'W_out' in self._pending_dW:
            weight_deltas['W_out_delta'] = self._pending_dW['W_out']
        
        self.signal(
            direction=SignalDirection.DOWN,
            signal_type=SignalType.SUGGESTION,
            payload=weight_deltas,
            target_level=0,
            requires_response=True,
        )
        
        # Send statistics UP to Meta^2
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'active_rule': self.active_rule,
                'mean_dW': total_change,
                'rpe': rpe,
                'dopamine_level': self.dopamine.level,
                'stdp_stats': self.stdp.get_stats(),
                'hebbian_stats': self.hebbian.get_stats(),
                'rule_usage': dict(self._rule_usage),
            },
        )
        
        return {
            'loss': loss,
            'dW': self._pending_dW,
            'rpe': rpe,
            'active_rule': self.active_rule,
            'metrics': self.metrics,
            'q_values': q_values,  # For action selection
            'td_error': td_error,  # For diagnostics
        }
    
    def _apply_stdp(self, weights: np.ndarray, spikes: np.ndarray) -> np.ndarray:
        """Apply STDP learning rule with proper temporal offset."""
        if len(spikes) == 0:
            return np.zeros_like(weights)
        
        N = weights.shape[0]
        n_spikes = min(len(spikes), N)
        
        # Current spikes are post-synaptic
        post_spikes = np.zeros(N)
        post_spikes[:n_spikes] = spikes[:n_spikes]
        
        # Previous spikes are pre-synaptic (temporal offset)
        if self._prev_spikes is None:
            # First call: no temporal offset yet
            pre_spikes = np.zeros(N)
        else:
            pre_spikes = self._prev_spikes.copy()
        
        # Store current spikes for next iteration
        self._prev_spikes = post_spikes.copy()
        
        return self.stdp.batch_update(weights, pre_spikes, post_spikes)
    
    def _apply_hebbian(self, weights: np.ndarray, activations: np.ndarray, reward: float) -> np.ndarray:
        """Apply Hebbian learning rule with temporal structure and reward modulation."""
        if len(activations) == 0:
            return np.zeros_like(weights)
        
        N = weights.shape[0]
        n_act = min(len(activations), N)
        
        # Current activations are post-synaptic
        post = np.zeros(N)
        post[:n_act] = activations[:n_act]
        
        # Previous activations are pre-synaptic (temporal offset)
        if self._hebbian_prev_activations is None:
            # First call: no temporal offset yet, store and return
            self._hebbian_prev_activations = post.copy()
            return np.zeros_like(weights)
        
        pre = self._hebbian_prev_activations.copy()
        
        # Store current for next iteration
        self._hebbian_prev_activations = post.copy()
        
        # Apply Hebbian update
        dW = self.hebbian.batch_update(weights, pre, post)
        
        # Reward modulation (three-factor learning)
        # Positive reward strengthens, negative weakens
        reward_factor = np.clip(reward, -1.0, 1.0)
        dW *= reward_factor
        
        return dW
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state: learning rule stats and weight change patterns."""
        stdp_stats = self.stdp.get_stats()
        hebb_stats = self.hebbian.get_stats()
        da_stats = self.dopamine.get_stats()
        
        return np.array([
            stdp_stats['ltp_fraction'],
            stdp_stats['ltd_fraction'],
            stdp_stats['mean_dw'],
            hebb_stats['mean_dw'],
            da_stats['dopamine_level'],
            da_stats['mean_rpe'],
            da_stats['expected_reward'],
            float(self.active_rule == 'stdp'),
            float(self.active_rule == 'hebbian'),
            float(self.active_rule == 'both'),
            self.metrics.loss,
            self.metrics.accuracy,
            self.metrics.stability,
            self.metrics.improvement_rate,
            float(self._total_weight_updates),
            self.pruner.get_stats()['last_sparsity'],
        ])
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply a suggestion (usually from Meta^2 about which rule to use)."""
        applied = False
        
        if 'learning_mode' in suggestion:
            new_mode = suggestion['learning_mode']
            if new_mode in ('gradient', 'rl', 'hybrid', 'auto'):
                self.learning_mode = new_mode
                applied = True
        
        if 'active_rule' in suggestion:
            new_rule = suggestion['active_rule']
            if new_rule in ('stdp', 'hebbian', 'both'):
                self.active_rule = new_rule
                applied = True
        
        if 'learning_rate' in suggestion:
            self.synapse_config.learning_rate = suggestion['learning_rate']
            applied = True
        
        if 'stdp_params' in suggestion:
            self.stdp.set_params(suggestion['stdp_params'])
            applied = True
        
        if 'hebbian_params' in suggestion:
            self.hebbian.set_params(suggestion['hebbian_params'])
            applied = True
        
        if 'pruning_params' in suggestion:
            self.pruner.set_params(suggestion['pruning_params'])
            applied = True
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate a suggestion."""
        score = 0.5
        reasons = []
        
        if 'active_rule' in suggestion:
            new_rule = suggestion['active_rule']
            if new_rule in ('stdp', 'hebbian', 'both'):
                # Accept rule changes if current performance is poor
                if self.metrics.improvement_rate < 0:
                    score = 0.9
                    reasons.append("Performance declining, willing to try new rule")
                elif self.metrics.stability < 0.5:
                    score = 0.7
                    reasons.append("Unstable, open to rule change")
                else:
                    score = 0.4  # Stable and improving, prefer current
                    reasons.append("Currently stable, reluctant to change")
            else:
                score = 0.0
                reasons.append(f"Unknown rule: {new_rule}")
        
        if 'learning_rate' in suggestion:
            lr = suggestion['learning_rate']
            if 1e-5 <= lr <= 1.0:
                score = max(score, 0.6)
                reasons.append(f"Learning rate {lr} in valid range")
            else:
                score = 0.1
                reasons.append(f"Learning rate {lr} out of range")
        
        return score, "; ".join(reasons) if reasons else "No evaluation criteria"
    
    def _self_optimize_weights(self):
        """Tune own learning rates based on performance."""
        # If improving, slightly decrease learning rate (fine-tune)
        # If stagnant, slightly increase (explore more)
        if self.metrics.improvement_rate > 0.01:
            self.synapse_config.learning_rate *= 0.999
        elif self.metrics.improvement_rate < -0.01:
            self.synapse_config.learning_rate *= 1.001
        
        # Clip
        self.synapse_config.learning_rate = np.clip(
            self.synapse_config.learning_rate, 1e-5, 0.1
        )
    
    def _self_optimize_synapses(self):
        """Decay eligibility traces."""
        self.stdp.decay_traces()
        self.dopamine.decay_eligibility()
    
    def _self_optimize_neurons(self):
        """No structural changes at Meta^1."""
        pass
