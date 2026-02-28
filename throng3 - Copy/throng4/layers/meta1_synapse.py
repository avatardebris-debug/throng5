"""
Meta^1: Dual-Head Synapse Optimizer for Throng4

Simplified from Throng3's SynapseOptimizer. Removes SNN-specific rules
(STDP/Hebbian) and focuses on gradient-based learning through the
dual-head ANN.

Handles two loss signals:
  1. TD error → backprop to Q-head + backbone
  2. Reward prediction error → backprop to reward-head + backbone

Also manages per-head learning rate multipliers for MAML integration.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class DualHeadSynapseConfig:
    """Configuration for dual-head synapse optimizer."""
    base_lr: float = 0.001
    aux_loss_weight: float = 0.1       # Weight for reward prediction loss
    dopamine_modulation: bool = True
    dopamine_decay: float = 0.95
    clip_gradients: float = 1.0
    # Per-head learning rate scaling
    backbone_lr_scale: float = 1.0
    q_head_lr_scale: float = 1.0
    reward_head_lr_scale: float = 0.5  # Slower for auxiliary task


class DualHeadSynapseOptimizer:
    """
    Meta^1: Gradient-based synapse optimization for dual-head ANN.
    
    Key difference from Throng3's SynapseOptimizer:
    - No STDP or Hebbian rules (those were for SNN spikes)
    - Gradient-based updates through both heads
    - Per-head learning rate multipliers (MAML-compatible)
    - Dopamine modulation preserved (reward scaling)
    """
    
    def __init__(self, ann_layer, config: Optional[DualHeadSynapseConfig] = None):
        """
        Args:
            ann_layer: ANNLayer instance (dual-head network)
            config: Optimizer configuration
        """
        self.ann = ann_layer
        self.config = config or DualHeadSynapseConfig()
        
        # Per-head learning rate multipliers (Meta^3/MAML can adjust these)
        self.lr_multipliers = {
            'W1': self.config.backbone_lr_scale,
            'b1': self.config.backbone_lr_scale,
            'W_q': self.config.q_head_lr_scale,
            'b_q': self.config.q_head_lr_scale,
            'W_r': self.config.reward_head_lr_scale,
            'b_r': self.config.reward_head_lr_scale,
        }
        
        # Dopamine state
        self.dopamine_level = 0.0
        self.reward_baseline = 0.0
        
        # Metrics
        self.n_updates = 0
        self.cumulative_td_error = 0.0
        self.cumulative_reward_error = 0.0
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one optimization step.
        
        Expected context:
            - state: Current state
            - action: Action taken
            - reward: Reward received
            - next_state: Next state
            - done: Episode termination
            - lr_multipliers: Optional per-weight LR multipliers from Meta^3
        
        Returns:
            Dict with td_error, reward_error, metrics
        """
        state = context['state']
        action = context['action']
        reward = context['reward']
        next_state = context['next_state']
        done = context['done']
        
        # Apply MAML lr_multipliers if provided
        lr_mults = context.get('lr_multipliers', self.lr_multipliers)
        if lr_mults:
            self.lr_multipliers.update(lr_mults)
        
        # Dopamine modulation
        effective_lr = self.config.base_lr
        if self.config.dopamine_modulation:
            rpe = reward - self.reward_baseline
            self.dopamine_level = (self.config.dopamine_decay * self.dopamine_level + 
                                  (1 - self.config.dopamine_decay) * rpe)
            self.reward_baseline = (self.config.dopamine_decay * self.reward_baseline + 
                                   (1 - self.config.dopamine_decay) * reward)
            # Modulate LR: higher for surprising rewards
            dopamine_scale = 1.0 + np.clip(self.dopamine_level, -0.5, 2.0)
            effective_lr *= dopamine_scale
        
        # === Q-learning update (primary task) ===
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            next_output = self.ann.forward(next_state)
            target_q = reward + 0.99 * np.max(next_output['q_values'])
        
        # Get current Q-value and reward prediction
        current_output = self.ann.forward(state)
        current_q = current_output['q_values'][action]
        reward_pred = current_output['reward_pred']
        
        # Compute errors
        td_error = target_q - current_q
        reward_error = reward - reward_pred
        
        # Clip gradients
        td_error = np.clip(td_error, -self.config.clip_gradients, self.config.clip_gradients)
        reward_error = np.clip(reward_error, -self.config.clip_gradients, self.config.clip_gradients)
        
        # === Backprop with per-head learning rates ===
        
        # Save weights before update (for Meta^3 tracking)
        weights_before = self.ann.get_weights()
        
        # Q-head backward pass (with per-weight LR multipliers)
        self._backward_q_with_multipliers(td_error, action, effective_lr)
        
        # Reward-head backward pass (with per-weight LR multipliers)
        self._backward_reward_with_multipliers(reward_error, effective_lr)
        
        # Track weight changes (for Meta^3)
        weights_after = self.ann.get_weights()
        weight_changes = {
            k: np.mean(np.abs(weights_after[k] - weights_before[k]))
            for k in weights_before
        }
        
        # Update metrics
        self.n_updates += 1
        self.cumulative_td_error += abs(td_error)
        self.cumulative_reward_error += abs(reward_error)
        
        return {
            'td_error': td_error,
            'reward_error': reward_error,
            'effective_lr': effective_lr,
            'dopamine_level': self.dopamine_level,
            'weight_changes': weight_changes,
            'lr_multipliers': self.lr_multipliers.copy(),
        }
    
    def _backward_q_with_multipliers(self, td_error: float, action: int, base_lr: float):
        """Backward pass for Q-head with per-weight LR multipliers."""
        cache = self.ann.cache
        
        # Q-head gradient
        dq = np.zeros(self.ann.n_outputs)
        dq[action] = -td_error
        
        # Backprop through Q-head
        dW_q = np.outer(cache['h'], dq)
        db_q = dq
        dh_q = self.ann.W_q @ dq
        
        # Backprop through backbone (ReLU)
        dz1 = dh_q * (cache['z1'] > 0)
        dW1 = np.outer(cache['x'], dz1)
        db1 = dz1
        
        # Apply updates with per-weight multipliers
        self.ann.W_q -= base_lr * self.lr_multipliers['W_q'] * dW_q
        self.ann.b_q -= base_lr * self.lr_multipliers['b_q'] * db_q
        self.ann.W1 -= base_lr * self.lr_multipliers['W1'] * dW1
        self.ann.b1 -= base_lr * self.lr_multipliers['b1'] * db1
    
    def _backward_reward_with_multipliers(self, reward_error: float, base_lr: float):
        """Backward pass for reward-head with per-weight LR multipliers."""
        cache = self.ann.cache
        aux_weight = self.config.aux_loss_weight
        
        # Reward-head gradient
        dr = -reward_error
        
        # Backprop through reward-head
        dW_r = np.outer(cache['h'], [dr])
        db_r = np.array([dr])
        dh_r = self.ann.W_r.flatten() * dr
        
        # Backprop through backbone (ReLU)
        dz1 = dh_r * (cache['z1'] > 0)
        dW1 = np.outer(cache['x'], dz1)
        db1 = dz1
        
        # Apply updates with per-weight multipliers (scaled by aux_weight)
        self.ann.W_r -= base_lr * aux_weight * self.lr_multipliers['W_r'] * dW_r
        self.ann.b_r -= base_lr * aux_weight * self.lr_multipliers['b_r'] * db_r
        self.ann.W1 -= base_lr * aux_weight * self.lr_multipliers['W1'] * dW1
        self.ann.b1 -= base_lr * aux_weight * self.lr_multipliers['b1'] * db1
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all weights (for Meta^3/MAML)."""
        return self.ann.get_weights()
    
    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set all weights (for Meta^3/MAML)."""
        self.ann.set_weights(weights)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            'n_updates': self.n_updates,
            'mean_td_error': self.cumulative_td_error / max(1, self.n_updates),
            'mean_reward_error': self.cumulative_reward_error / max(1, self.n_updates),
            'dopamine_level': self.dopamine_level,
            'lr_multipliers': self.lr_multipliers.copy(),
        }
