"""
Meta^3: MAML for Dual-Head ANN (Throng4)

Adapts Throng3's TaskConditionedMAML to work with the dual-head
ANN architecture. Key changes:

1. _forward() uses dual-head ANN forward pass
2. _compute_gradient() handles both Q-head and reward-head gradients
3. lr_multipliers are per-head (W1, W_q, W_r) instead of per-SNN-weight
4. Meta-learning optimizes strategies for BOTH heads

The inner loop now does:
  - Forward through dual-head ANN
  - Compute TD error + reward prediction error
  - Backprop with per-head learning rates
  - Return adapted weights
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class DualHeadMAMLConfig:
    """Configuration for dual-head MAML."""
    meta_lr: float = 0.001           # Outer loop learning rate
    inner_lr: float = 0.01           # Inner loop learning rate
    inner_steps: int = 5             # Steps per inner loop
    meta_batch_size: int = 4         # Tasks per meta-update
    first_order: bool = True         # First-order MAML (faster)
    aux_loss_weight: float = 0.1     # Weight for reward prediction in inner loop
    gamma: float = 0.99              # Q-learning discount
    # Per-head meta-learning (can MAML learn different rates per head?)
    learn_per_head_lr: bool = True
    max_grad_norm: float = 10.0      # Gradient clipping threshold


class DualHeadMAML:
    """
    Meta^3: MAML adapted for dual-head ANN.
    
    Learns:
    - Good weight initializations for both Q-head and reward-head
    - Per-head learning rate multipliers
    - Task-specific adaptation strategies
    
    Architecture:
        Meta^3 (MAML) → lr_multipliers → Meta^1 (DualHeadSynapseOptimizer)
                       → weight init    → Meta^0 (ANNLayer)
    """
    
    def __init__(self, config: Optional[DualHeadMAMLConfig] = None):
        self.config = config or DualHeadMAMLConfig()
        
        # Learned meta-parameters per task type
        self.meta_params = {
            'rl': {
                'lr_multipliers': {
                    'W1': 1.0, 'b1': 1.0,
                    'W_q': 1.0, 'b_q': 1.0,
                    'W_r': 0.5, 'b_r': 0.5,  # Slower for aux task
                },
                'inner_lr': self.config.inner_lr,
                'inner_steps': self.config.inner_steps,
            },
        }
        
        # Meta-optimizer state (Adam)
        self.adam_state = {'m': {}, 'v': {}, 't': 0}
        
        # Task batch for meta-update
        self.meta_batch: List[Dict[str, Any]] = []
        
        # Stats
        self.meta_updates = 0
        self.tasks_seen = 0
    
    def get_lr_multipliers(self) -> Dict[str, float]:
        """Get current per-head learning rate multipliers."""
        return self.meta_params['rl']['lr_multipliers'].copy()
    
    def inner_loop(self,
                   ann_layer,
                   transitions: List[Dict[str, Any]],
                   inner_lr: Optional[float] = None,
                   inner_steps: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Inner loop: Adapt ANN weights to a set of transitions.
        
        This is the "fast adaptation" part of MAML.
        
        Args:
            ann_layer: ANNLayer instance
            transitions: List of {state, action, reward, next_state, done}
            inner_lr: Learning rate (default: from meta_params)
            inner_steps: Number of gradient steps
        
        Returns:
            Adapted weights dict
        """
        lr = inner_lr or self.meta_params['rl']['inner_lr']
        steps = inner_steps or self.meta_params['rl']['inner_steps']
        lr_mults = self.meta_params['rl']['lr_multipliers']
        
        # Copy current weights
        adapted = ann_layer.get_weights()
        
        # Store original weights to restore later
        original_weights = ann_layer.get_weights()
        
        for step in range(steps):
            # Set adapted weights
            ann_layer.set_weights(adapted)
            
            # Accumulate gradients over transitions
            grad_accum = {k: np.zeros_like(v) for k, v in adapted.items()}
            n_samples = 0
            
            for t in transitions:
                # Forward pass
                output = ann_layer.forward(t['state'])
                
                # Compute target Q
                if t['done']:
                    target_q = t['reward']
                else:
                    next_output = ann_layer.forward(t['next_state'])
                    target_q = t['reward'] + self.config.gamma * np.max(next_output['q_values'])
                
                # Current forward (needed for cache)
                output = ann_layer.forward(t['state'])
                
                # TD error
                td_error = target_q - output['q_values'][t['action']]
                
                # Reward prediction error
                reward_error = t['reward'] - output['reward_pred']
                
                # Compute gradients manually (for MAML we need explicit gradients)
                grads = self._compute_dual_gradients(
                    ann_layer, td_error, t['action'], reward_error
                )
                
                for k in grad_accum:
                    if k in grads:
                        g = grads[k]
                        if np.any(np.isnan(g)) or np.any(np.isinf(g)):
                            continue  # Skip corrupted gradients
                        grad_accum[k] += g
                n_samples += 1
            
            # Average, clip, and apply
            if n_samples > 0:
                for k in adapted:
                    if k in grad_accum:
                        grad = grad_accum[k] / n_samples
                        # Clip gradient norm
                        grad_norm = np.linalg.norm(grad)
                        if grad_norm > self.config.max_grad_norm:
                            grad = grad * (self.config.max_grad_norm / grad_norm)
                        mult = lr_mults.get(k, 1.0)
                        update = lr * mult * grad
                        if not np.any(np.isnan(update)):
                            adapted[k] -= update
        
        # Restore original weights (caller decides whether to use adapted)
        ann_layer.set_weights(original_weights)
        
        return adapted
    
    def _compute_dual_gradients(self, ann_layer, td_error: float, 
                                 action: int, reward_error: float) -> Dict[str, np.ndarray]:
        """
        Compute gradients for both heads.
        
        Returns gradient dict for all weight matrices.
        """
        cache = ann_layer.cache
        aux_w = self.config.aux_loss_weight
        
        # === Q-head gradients ===
        dq = np.zeros(ann_layer.n_outputs)
        dq[action] = -td_error
        
        dW_q = np.outer(cache['h'], dq)
        db_q = dq
        dh_q = ann_layer.W_q @ dq
        
        dz1_q = dh_q * (cache['z1'] > 0)
        dW1_q = np.outer(cache['x'], dz1_q)
        db1_q = dz1_q
        
        # === Reward-head gradients ===
        dr = -reward_error
        
        dW_r = np.outer(cache['h'], [dr])
        db_r = np.array([dr])
        dh_r = ann_layer.W_r.flatten() * dr
        
        dz1_r = dh_r * (cache['z1'] > 0)
        dW1_r = np.outer(cache['x'], dz1_r)
        db1_r = dz1_r
        
        # === Combine backbone gradients ===
        return {
            'W1': dW1_q + aux_w * dW1_r,
            'b1': db1_q + aux_w * db1_r,
            'W_q': dW_q,
            'b_q': db_q,
            'W_r': aux_w * dW_r,
            'b_r': aux_w * db_r,
        }
    
    def meta_update(self, ann_layer, task_batch: List[Dict[str, Any]]):
        """
        Meta-learning outer loop update.
        
        Each task in task_batch should have:
            - support_set: List of transitions for inner loop adaptation
            - query_set: List of transitions for meta-gradient evaluation
        
        Args:
            ann_layer: ANNLayer instance
            task_batch: List of task dicts
        """
        if not task_batch:
            return
        
        # Get current meta-initialization
        meta_init = ann_layer.get_weights()
        
        # Accumulate meta-gradients
        meta_grad = {k: np.zeros_like(v) for k, v in meta_init.items()}
        lr_grad = {k: 0.0 for k in self.meta_params['rl']['lr_multipliers']}
        
        for task in task_batch:
            support = task['support_set']
            query = task['query_set']
            
            # Inner loop: adapt to support set
            adapted_weights = self.inner_loop(ann_layer, support)
            
            # Evaluate adapted weights on query set
            # Set adapted weights temporarily
            ann_layer.set_weights(adapted_weights)
            
            # Compute query loss gradient
            query_grad = {k: np.zeros_like(v) for k, v in adapted_weights.items()}
            n_query = 0
            
            for t in query:
                output = ann_layer.forward(t['state'])
                
                if t['done']:
                    target_q = t['reward']
                else:
                    next_output = ann_layer.forward(t['next_state'])
                    target_q = t['reward'] + self.config.gamma * np.max(next_output['q_values'])
                
                output = ann_layer.forward(t['state'])
                td_error = target_q - output['q_values'][t['action']]
                reward_error = t['reward'] - output['reward_pred']
                
                grads = self._compute_dual_gradients(
                    ann_layer, td_error, t['action'], reward_error
                )
                
                for k in query_grad:
                    if k in grads:
                        query_grad[k] += grads[k]
                n_query += 1
            
            if n_query > 0:
                for k in meta_grad:
                    if k in query_grad:
                        g = query_grad[k] / n_query
                        if not np.any(np.isnan(g)) and not np.any(np.isinf(g)):
                            # Clip
                            gnorm = np.linalg.norm(g)
                            if gnorm > self.config.max_grad_norm:
                                g = g * (self.config.max_grad_norm / gnorm)
                            meta_grad[k] += g
            
            # Compute LR multiplier gradients (first-order approx)
            if self.config.learn_per_head_lr:
                for k in lr_grad:
                    if k in query_grad and n_query > 0:
                        weight_change = np.mean(np.abs(adapted_weights[k] - meta_init[k]))
                        query_loss = np.mean(np.abs(query_grad[k])) if k in query_grad else 0
                        val = weight_change * query_loss
                        if np.isfinite(val):
                            lr_grad[k] += val
        
        # Average meta-gradients
        n_tasks = len(task_batch)
        for k in meta_grad:
            meta_grad[k] /= n_tasks
        
        # Restore meta-initialization
        ann_layer.set_weights(meta_init)
        
        # Apply meta-gradient (Adam)
        self._apply_adam_update(ann_layer, meta_grad)
        
        # Update lr_multipliers
        if self.config.learn_per_head_lr:
            lr_mults = self.meta_params['rl']['lr_multipliers']
            for k in lr_grad:
                lr_grad[k] /= n_tasks
                # Small update to lr_multipliers
                lr_mults[k] -= self.config.meta_lr * 0.1 * lr_grad[k]
                lr_mults[k] = np.clip(lr_mults[k], 0.01, 10.0)
        
        self.meta_updates += 1
        self.tasks_seen += n_tasks
    
    def _apply_adam_update(self, ann_layer, meta_grad: Dict[str, np.ndarray]):
        """Apply Adam optimizer for meta-gradient with NaN protection."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        self.adam_state['t'] += 1
        t = self.adam_state['t']
        
        weights = ann_layer.get_weights()
        
        for k, grad in meta_grad.items():
            # Skip NaN/Inf gradients
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
                continue
            
            if k not in self.adam_state['m']:
                self.adam_state['m'][k] = np.zeros_like(grad)
                self.adam_state['v'][k] = np.zeros_like(grad)
            
            # Adam moments
            self.adam_state['m'][k] = beta1 * self.adam_state['m'][k] + (1 - beta1) * grad
            self.adam_state['v'][k] = beta2 * self.adam_state['v'][k] + (1 - beta2) * grad**2
            
            # Bias correction
            m_hat = self.adam_state['m'][k] / (1 - beta1**t)
            v_hat = self.adam_state['v'][k] / (1 - beta2**t)
            
            # Update with NaN guard
            update = self.config.meta_lr * m_hat / (np.sqrt(v_hat) + eps)
            if not np.any(np.isnan(update)):
                weights[k] -= update
        
        ann_layer.set_weights(weights)
    
    def add_task(self, support_set: List[Dict], query_set: List[Dict]):
        """Add a task to the meta-batch."""
        self.meta_batch.append({
            'support_set': support_set,
            'query_set': query_set,
        })
    
    def maybe_meta_update(self, ann_layer) -> bool:
        """
        Perform meta-update if batch is full.
        
        Returns:
            True if meta-update was performed
        """
        if len(self.meta_batch) >= self.config.meta_batch_size:
            self.meta_update(ann_layer, self.meta_batch)
            self.meta_batch = []
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MAML statistics."""
        return {
            'meta_updates': self.meta_updates,
            'tasks_seen': self.tasks_seen,
            'pending_tasks': len(self.meta_batch),
            'lr_multipliers': self.meta_params['rl']['lr_multipliers'].copy(),
            'adam_step': self.adam_state['t'],
        }
