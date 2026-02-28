"""
STDP — Spike-Timing-Dependent Plasticity

Biological temporal learning rule adapted from throng2.
Pre fires BEFORE post → strengthen (LTP)
Pre fires AFTER post → weaken (LTD)

Enhanced for Meta^N with:
- Configurable time constants (can be tuned by Meta^2)
- Eligibility traces for three-factor learning
- Batch mode for efficiency
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class STDPConfig:
    """Configuration for STDP, tunable by higher meta-layers."""
    tau_plus: float = 0.020      # LTP time constant (seconds)
    tau_minus: float = 0.020     # LTD time constant (seconds)
    A_plus: float = 0.01         # LTP learning rate
    A_minus: float = 0.012       # LTD learning rate (slightly asymmetric)
    w_max: float = 1.0           # Maximum weight
    w_min: float = -1.0          # Minimum weight
    trace_decay: float = 0.95    # Eligibility trace decay


class STDPRule:
    """
    Spike-Timing-Dependent Plasticity learning rule.
    
    Supports both spike-by-spike and batch computation.
    Parameters can be modified by Meta^2 (LearningRuleSelector).
    """
    
    def __init__(self, config: Optional[STDPConfig] = None):
        self.config = config or STDPConfig()
        
        # Spike time tracking
        self.pre_traces: Dict[int, float] = defaultdict(float)   # Eligibility traces
        self.post_traces: Dict[int, float] = defaultdict(float)
        
        # Statistics
        self.total_updates = 0
        self.total_ltp = 0
        self.total_ltd = 0
        self.mean_dw = 0.0
    
    def record_spike(self, neuron_id: int, is_pre: bool, time: float):
        """Record a spike event and update traces."""
        if is_pre:
            self.pre_traces[neuron_id] = 1.0  # Reset trace to max
        else:
            self.post_traces[neuron_id] = 1.0
    
    def compute_dw(self, pre_id: int, post_id: int) -> float:
        """
        Compute weight change for a single synapse.
        
        Uses eligibility traces for efficient computation.
        
        Returns:
            dw: Weight change to apply
        """
        pre_trace = self.pre_traces.get(pre_id, 0.0)
        post_trace = self.post_traces.get(post_id, 0.0)
        
        # LTP: pre trace active when post spikes
        ltp = self.config.A_plus * pre_trace
        # LTD: post trace active when pre spikes
        ltd = -self.config.A_minus * post_trace
        
        dw = ltp + ltd
        
        # Track statistics
        self.total_updates += 1
        if dw > 0:
            self.total_ltp += 1
        elif dw < 0:
            self.total_ltd += 1
        self.mean_dw = 0.99 * self.mean_dw + 0.01 * dw
        
        return dw
    
    def batch_update(self, weights: np.ndarray,
                     pre_spikes: np.ndarray, post_spikes: np.ndarray,
                     dt: float = 0.001) -> np.ndarray:
        """
        Batch STDP update for a weight matrix.
        
        Args:
            weights: (N_post, N_pre) weight matrix
            pre_spikes: (N_pre,) binary spike vector
            post_spikes: (N_post,) binary spike vector
            dt: Time step
            
        Returns:
            dW: Weight change matrix
        """
        N_post, N_pre = weights.shape
        
        # Update traces
        pre_trace = getattr(self, '_batch_pre_trace', np.zeros(N_pre))
        post_trace = getattr(self, '_batch_post_trace', np.zeros(N_post))
        
        # Decay traces
        decay_pre = np.exp(-dt / self.config.tau_plus)
        decay_post = np.exp(-dt / self.config.tau_minus)
        pre_trace = pre_trace * decay_pre + pre_spikes
        post_trace = post_trace * decay_post + post_spikes
        
        # Store for next call
        self._batch_pre_trace = pre_trace
        self._batch_post_trace = post_trace
        
        # Compute weight changes
        # LTP: pre trace contributes when post spikes
        dW_ltp = self.config.A_plus * np.outer(post_spikes, pre_trace)
        # LTD: post trace contributes when pre spikes
        dW_ltd = -self.config.A_minus * np.outer(post_trace, pre_spikes)
        
        dW = dW_ltp + dW_ltd
        
        # Clip resulting weights
        new_weights = np.clip(weights + dW, self.config.w_min, self.config.w_max)
        dW = new_weights - weights
        
        return dW
    
    def decay_traces(self):
        """Decay all eligibility traces."""
        decay = self.config.trace_decay
        for k in self.pre_traces:
            self.pre_traces[k] *= decay
        for k in self.post_traces:
            self.post_traces[k] *= decay
    
    def get_params(self) -> Dict[str, float]:
        """Get current parameters (for Meta^2 to read)."""
        return {
            'tau_plus': self.config.tau_plus,
            'tau_minus': self.config.tau_minus,
            'A_plus': self.config.A_plus,
            'A_minus': self.config.A_minus,
            'w_max': self.config.w_max,
            'w_min': self.config.w_min,
        }
    
    def set_params(self, params: Dict[str, float]):
        """Set parameters (called by Meta^2)."""
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get learning statistics."""
        total = max(self.total_updates, 1)
        return {
            'total_updates': self.total_updates,
            'ltp_fraction': self.total_ltp / total,
            'ltd_fraction': self.total_ltd / total,
            'mean_dw': self.mean_dw,
        }
