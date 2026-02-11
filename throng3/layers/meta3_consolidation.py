"""
Meta^3: Weight Consolidation Layer

Implements Elastic Weight Consolidation (EWC) to prevent catastrophic
interference. Tracks which weights are important across tasks and
protects them from being overwritten.

Key idea: After each task, compute Fisher information (importance) for
each weight. Weights that are important across multiple tasks get
protected. Task-specific weights stay plastic.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class ConsolidationConfig:
    """Configuration for weight consolidation."""
    ewc_lambda: float = 1000.0  # Strength of consolidation penalty
    fisher_decay: float = 0.95   # Decay old Fisher info over time
    min_fisher: float = 1e-8     # Minimum Fisher value (numerical stability)


class WeightConsolidation(MetaLayer):
    """
    Meta^3: Decides which weights are shared knowledge vs task-specific.
    
    Uses Elastic Weight Consolidation (EWC) to:
    1. Track weight importance via Fisher information
    2. Protect important weights from changing
    3. Allow task-specific weights to adapt
    
    This prevents catastrophic interference and enables compound transfer.
    """
    
    def __init__(self, config: Optional[ConsolidationConfig] = None):
        consolidation_config = config or ConsolidationConfig()
        
        # Initialize parent MetaLayer with dict config
        config_dict = {
            'ewc_lambda': consolidation_config.ewc_lambda,
            'fisher_decay': consolidation_config.fisher_decay,
            'min_fisher': consolidation_config.min_fisher,
        }
        super().__init__(level=3, name="WeightConsolidation", config=config_dict)
        
        # Store our config separately
        self.consolidation_config = consolidation_config
        
        # Fisher information: tracks weight importance
        self.fisher_info: Dict[str, np.ndarray] = {}
        
        # Per-task Fisher history (for intersection computation)
        self.task_fisher_history: Dict[str, List[np.ndarray]] = {}
        
        # Optimal weights from previous tasks
        self.optimal_weights: Dict[str, List[np.ndarray]] = {}
        
        # Task counter
        self.task_count = 0
        
        # Stats
        self.consolidation_penalty = 0.0
        self.protected_weights_pct = 0.0
    
    def consolidate_task(self, weights: Dict[str, np.ndarray], 
                        fisher: Dict[str, np.ndarray]):
        """
        Called after a task completes. Stores important weights and
        accumulates Fisher information.
        
        Args:
            weights: Current weight matrices
            fisher: Fisher information from this task
        """
        self.task_count += 1
        
        # Store optimal weights for this task
        for name, W in weights.items():
            if name not in self.optimal_weights:
                self.optimal_weights[name] = []
            self.optimal_weights[name].append(W.copy())
        
        # Accumulate Fisher information
        for name, F in fisher.items():
            # Store this task's Fisher info for intersection computation
            if name not in self.task_fisher_history:
                self.task_fisher_history[name] = []
            self.task_fisher_history[name].append(F.copy())
            
            if name not in self.fisher_info:
                # First task: just store Fisher info
                self.fisher_info[name] = F.copy()
            else:
                # Subsequent tasks: accumulate with decay
                self.fisher_info[name] = (
                    self.consolidation_config.fisher_decay * self.fisher_info[name] + F
                )
            
            # Ensure numerical stability
            self.fisher_info[name] = np.maximum(
                self.fisher_info[name], 
                self.consolidation_config.min_fisher
            )
        
        # Update stats
        self._update_stats()
    
    def compute_ewc_penalty(self, current_weights: Dict[str, np.ndarray]) -> float:
        """
        Compute EWC penalty: λ * Σ F_i * (w_i - w_i*)^2
        
        This penalizes changing weights that were important for previous tasks.
        
        Args:
            current_weights: Current weight matrices
            
        Returns:
            EWC penalty (scalar)
        """
        if self.task_count == 0:
            # No previous tasks, no penalty
            return 0.0
        
        penalty = 0.0
        
        for name, W_current in current_weights.items():
            if name not in self.fisher_info:
                continue
            
            F = self.fisher_info[name]
            
            # Penalty for deviating from ALL previous optimal weights
            for W_opt in self.optimal_weights.get(name, []):
                diff = W_current - W_opt
                penalty += np.sum(F * diff ** 2)
        
        self.consolidation_penalty = self.consolidation_config.ewc_lambda * penalty
        return self.consolidation_penalty
    
    def compute_penalty_gradient(self, current_weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute gradient of EWC penalty w.r.t. weights.
        
        This is added to the regular gradient to prevent changing important weights.
        
        dL_ewc/dW = 2λ * Σ F * (W - W*)
        
        Args:
            current_weights: Current weight matrices
            
        Returns:
            Gradient of EWC penalty for each weight matrix
        """
        if self.task_count == 0:
            return {}
        
        penalty_grad = {}
        
        for name, W_current in current_weights.items():
            if name not in self.fisher_info:
                continue
            
            F = self.fisher_info[name]
            grad = np.zeros_like(W_current)
            
            # Sum gradient from all previous tasks
            for W_opt in self.optimal_weights.get(name, []):
                diff = W_current - W_opt
                grad += 2.0 * self.consolidation_config.ewc_lambda * F * diff
            
            penalty_grad[name] = grad
        
        return penalty_grad
    
    def compute_fisher_intersection(self, percentile: float = 80.0) -> Dict[str, np.ndarray]:
        """
        Identify weights important across multiple tasks (Fisher intersection).
        
        These are the "shared representations" - weights that matter for all tasks.
        
        Args:
            percentile: Threshold percentile for "important" (default: top 20%)
            
        Returns:
            Dict of binary masks (1 = shared, 0 = task-specific)
        """
        if self.task_count < 2:
            # Need at least 2 tasks for intersection
            return {}
        
        intersection = {}
        
        for name, task_fishers in self.task_fisher_history.items():
            if len(task_fishers) < 2:
                continue
            
            # Compute threshold for each task (top percentile)
            thresholds = [np.percentile(F, percentile) for F in task_fishers]
            
            # Create binary masks: 1 if Fisher > threshold
            masks = [F > thresh for F, thresh in zip(task_fishers, thresholds)]
            
            # Intersection: high Fisher in ALL tasks
            # This identifies weights important across all tasks
            intersection_mask = np.all(masks, axis=0).astype(float)
            
            intersection[name] = intersection_mask
        
        return intersection
    
    def get_lr_multipliers(self, boost_factor: float = 2.0, 
                          percentile: float = 80.0,
                          invert_for_rl: bool = True) -> Dict[str, np.ndarray]:
        """
        Get learning rate multipliers for each weight.
        
        For supervised learning: Boost high-Fisher weights (shared features)
        For RL: Boost LOW-Fisher weights (spare capacity, avoid destabilizing policy)
        
        Args:
            boost_factor: Multiplier for target weights (default: 2x)
            percentile: Threshold for Fisher intersection
            invert_for_rl: If True, boost LOW-overlap weights (for RL)
                          If False, boost HIGH-overlap weights (for supervised)
            
        Returns:
            Dict of LR multipliers (boost_factor for target, 1.0 for others)
        """
        intersection = self.compute_fisher_intersection(percentile)
        
        if not intersection:
            # No intersection yet, return uniform multipliers
            return {}
        
        multipliers = {}
        for name, mask in intersection.items():
            if invert_for_rl:
                # RL: Boost LOW-overlap weights (spare capacity)
                # Invert mask: 1 -> 0, 0 -> 1
                inverted_mask = 1.0 - mask
                multipliers[name] = 1.0 + inverted_mask * (boost_factor - 1.0)
            else:
                # Supervised: Boost HIGH-overlap weights (shared features)
                multipliers[name] = 1.0 + mask * (boost_factor - 1.0)
        
        return multipliers
    
    def _update_stats(self):
        """Update statistics about consolidation."""
        if not self.fisher_info:
            self.protected_weights_pct = 0.0
            return
        
        # Count weights with high Fisher values (protected)
        total_weights = 0
        protected_weights = 0
        
        for name, F in self.fisher_info.items():
            total_weights += F.size
            # Consider weight "protected" if Fisher > median
            median_fisher = np.median(F)
            protected_weights += np.sum(F > median_fisher)
        
        if total_weights > 0:
            self.protected_weights_pct = protected_weights / total_weights * 100
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Meta^3 optimization: Decide which weights to protect.
        
        This is called during training to compute EWC penalty.
        """
        # Get current weights from Meta^0
        weights = context.get('weights', {})
        
        # Compute EWC penalty
        penalty = self.compute_ewc_penalty(weights)
        
        # Send penalty gradient to Meta^1 (if requested)
        penalty_grad = {}
        if context.get('request_penalty_gradient', False):
            penalty_grad = self.compute_penalty_gradient(weights)
        
        # Compute learning rate multipliers (Fisher boosting)
        lr_multipliers = self.get_lr_multipliers(
            boost_factor=context.get('fisher_boost_factor', 2.0),
            invert_for_rl=context.get('fisher_invert_for_rl', True)  # Default: invert for RL
        )
        
        return {
            'ewc_penalty': penalty,
            'penalty_gradient': penalty_grad,
            'lr_multipliers': lr_multipliers,
            'task_count': self.task_count,
            'protected_weights_pct': self.protected_weights_pct,
            'consolidation_strength': self.consolidation_config.ewc_lambda,
        }
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass (no-op for Meta^3, just passes through)."""
        return input_data
    
    def reset(self):
        """Reset consolidation state (for new experiment)."""
        self.fisher_info = {}
        self.task_fisher_history = {}
        self.optimal_weights = {}
        self.task_count = 0
        self.consolidation_penalty = 0.0
        self.protected_weights_pct = 0.0
    
    # Abstract method implementations
    def _compute_state_vector(self) -> np.ndarray:
        """Compute state vector for holographic encoding."""
        # Return Fisher info as state
        if not self.fisher_info:
            return np.zeros(10)
        
        # Flatten all Fisher values
        values = []
        for F in self.fisher_info.values():
            values.extend(F.flatten()[:10])  # Take first 10 values
        
        return np.array(values[:64] if len(values) >= 64 else values + [0]*(64-len(values)))
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestion (e.g., change EWC lambda)."""
        if 'ewc_lambda' in suggestion:
            self.consolidation_config.ewc_lambda = suggestion['ewc_lambda']
            return True
        return False
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> tuple:
        """Evaluate suggestion."""
        # Meta^3 is pretty autonomous, accept most suggestions
        return (0.8, "Meta^3 consolidation accepts most suggestions")
    
    def _self_optimize_weights(self):
        """Optimize EWC lambda based on performance."""
        pass  # Could tune ewc_lambda here
    
    def _self_optimize_synapses(self):
        """No synapse-level optimization for Meta^3."""
        pass
    
    def _self_optimize_neurons(self):
        """No neuron-level optimization for Meta^3."""
        pass

