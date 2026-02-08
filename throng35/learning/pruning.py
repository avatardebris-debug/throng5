"""
Nash Pruning — Game-Theoretic Connection Pruning

Adapted from throng2's Nash equilibrium pruning.
Models the network as a game where neurons compete for resources.
Connections that don't contribute are pruned.

Enhanced for Meta^N:
- Configurable by Meta^2 and Meta^5
- Reports pruning decisions UP for architecture search
- Supports regrowth (not just pruning)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass 
class PruningConfig:
    """Configuration for Nash pruning."""
    threshold: float = 0.05          # Minimum weight magnitude to keep
    resource_budget: float = 1.0     # Total resources per neuron
    competition: float = 0.5         # Competition strength (0-1)
    regrowth_rate: float = 0.01     # Fraction of pruned connections to regrow
    target_sparsity: float = 0.7    # Target connection sparsity
    min_connections: int = 5         # Minimum connections per neuron


class NashPruner:
    """
    Game-theoretic pruning with regrowth.
    
    Balances efficiency (fewer connections) with performance
    (maintain information flow).
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        
        # Statistics
        self.total_pruned = 0
        self.total_regrown = 0
        self.pruning_history: List[Dict] = []
    
    def compute_payoffs(self, weights: np.ndarray,
                       activities: np.ndarray) -> np.ndarray:
        """
        Compute payoff for each connection.
        
        Payoff = |weight| * activity_correlation - cost
        
        Args:
            weights: (N, N) weight matrix
            activities: (N,) neuron activities
            
        Returns:
            (N, N) payoff matrix
        """
        N = len(activities)
        
        # Benefit: weight magnitude * pre-post activity correlation
        activity_outer = np.outer(activities, activities)
        benefit = np.abs(weights) * np.abs(activity_outer)
        
        # Cost: proportional to total connections per neuron
        connection_count = np.sum(np.abs(weights) > 1e-8, axis=1, keepdims=True)
        cost = self.config.competition * connection_count / max(N, 1)
        
        payoffs = benefit - cost
        return payoffs
    
    def prune(self, weights: np.ndarray,
              activities: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Prune weak connections based on Nash equilibrium.
        
        Args:
            weights: (N, N) weight matrix
            activities: (N,) neuron activities
            
        Returns:
            (pruned_weights, stats_dict)
        """
        N = weights.shape[0]
        payoffs = self.compute_payoffs(weights, activities)
        
        # Create mask: keep connections with positive payoff above threshold
        mask = np.abs(weights) > self.config.threshold
        mask = mask & (payoffs > 0)
        
        # Ensure minimum connections per neuron
        for i in range(N):
            if np.sum(mask[i]) < self.config.min_connections:
                # Keep top-k connections by payoff
                top_k = np.argsort(payoffs[i])[-self.config.min_connections:]
                mask[i, top_k] = True
        
        # Apply mask
        pruned_weights = weights * mask
        
        # Count pruned
        n_before = np.sum(np.abs(weights) > 1e-8)
        n_after = np.sum(np.abs(pruned_weights) > 1e-8)
        n_pruned = n_before - n_after
        
        # Optional regrowth
        n_regrow = 0
        if self.config.regrowth_rate > 0:
            n_regrow = self._regrow(pruned_weights, activities)
        
        # Compute current sparsity
        total_possible = N * N
        current_sparsity = 1.0 - (n_after + n_regrow) / max(total_possible, 1)
        
        stats = {
            'n_pruned': int(n_pruned),
            'n_regrown': n_regrow,
            'n_connections': int(n_after + n_regrow),
            'sparsity': float(current_sparsity),
            'mean_payoff': float(np.mean(payoffs)),
            'target_sparsity': self.config.target_sparsity,
        }
        
        self.total_pruned += n_pruned
        self.total_regrown += n_regrow
        self.pruning_history.append(stats)
        
        return pruned_weights, stats
    
    def _regrow(self, weights: np.ndarray, activities: np.ndarray) -> int:
        """
        Regrow a fraction of connections in high-activity regions.
        
        Returns:
            Number of connections regrown
        """
        N = weights.shape[0]
        n_zero = np.sum(np.abs(weights) < 1e-8)
        n_to_regrow = int(n_zero * self.config.regrowth_rate)
        
        if n_to_regrow == 0:
            return 0
        
        # Find zero connections
        zero_mask = np.abs(weights) < 1e-8
        zero_indices = np.argwhere(zero_mask)
        
        if len(zero_indices) == 0:
            return 0
        
        # Prioritize connections between active neurons
        scores = np.array([
            activities[i] * activities[j] if i != j else 0
            for i, j in zero_indices
        ])
        
        # Select top-scoring zeros for regrowth
        n_to_regrow = min(n_to_regrow, len(zero_indices))
        top_k = np.argsort(scores)[-n_to_regrow:]
        
        for idx in top_k:
            i, j = zero_indices[idx]
            weights[i, j] = np.random.randn() * 0.01  # Small random init
        
        return n_to_regrow
    
    def get_params(self) -> Dict[str, float]:
        return {
            'threshold': self.config.threshold,
            'competition': self.config.competition,
            'regrowth_rate': self.config.regrowth_rate,
            'target_sparsity': self.config.target_sparsity,
        }
    
    def set_params(self, params: Dict[str, float]):
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_stats(self) -> Dict:
        return {
            'total_pruned': self.total_pruned,
            'total_regrown': self.total_regrown,
            'history_length': len(self.pruning_history),
            'last_sparsity': (
                self.pruning_history[-1]['sparsity'] 
                if self.pruning_history else 0.0
            ),
        }
