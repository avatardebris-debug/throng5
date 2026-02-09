"""
Adaptive Router — Learn Which Regions to Activate

Intelligently gates regions based on task context to improve efficiency
while maintaining performance.
"""

from typing import Dict, Any, Optional
import numpy as np


class AdaptiveRouter:
    """
    Adaptive router that learns which regions to activate.
    
    Strategy:
    - Striatum: Always active (primary action selection)
    - Cortex: Gate when TD-error is stable (policy converged)
    - Hippocampus: Gate when sequences are short (no temporal patterns)
    """
    
    def __init__(self, 
                 cortex_td_threshold: float = 0.05,
                 hippocampus_sequence_threshold: int = 3,
                 warmup_steps: int = 100):
        """
        Initialize adaptive router.
        
        Args:
            cortex_td_threshold: Gate Cortex when TD-error < this
            hippocampus_sequence_threshold: Gate Hippocampus when sequence < this
            warmup_steps: Always activate all regions for first N steps
        """
        self.cortex_td_threshold = cortex_td_threshold
        self.hippocampus_sequence_threshold = hippocampus_sequence_threshold
        self.warmup_steps = warmup_steps
        
        # Tracking
        self.step_count = 0
        self.td_error_history = []
        self.max_td_history = 20
        
        # Statistics
        self.activation_counts = {
            'striatum': 0,
            'cortex': 0,
            'hippocampus': 0
        }
        self.total_steps = 0
    
    def should_activate(self, 
                       region_name: str, 
                       context: Dict[str, Any]) -> bool:
        """
        Decide if a region should be activated this step.
        
        Args:
            region_name: Name of region ('striatum', 'cortex', 'hippocampus')
            context: Current context with keys:
                - 'td_error': Current TD-error (for Cortex gating)
                - 'sequence_length': Current sequence length (for Hippocampus)
                - 'episode_step': Step within episode
        
        Returns:
            True if region should be activated
        """
        self.total_steps += 1
        
        # Warmup: activate all regions
        if self.step_count < self.warmup_steps:
            self.step_count += 1
            self.activation_counts[region_name] += 1
            return True
        
        # Striatum: Always active (primary controller)
        if region_name == 'striatum':
            self.activation_counts[region_name] += 1
            return True
        
        # Cortex: Gate when TD-error is stable
        if region_name == 'cortex':
            td_error = abs(context.get('td_error', 1.0))
            
            # Track TD-error history
            self.td_error_history.append(td_error)
            if len(self.td_error_history) > self.max_td_history:
                self.td_error_history.pop(0)
            
            # Gate if recent TD-errors are consistently low
            if len(self.td_error_history) >= 10:
                recent_avg = np.mean(self.td_error_history[-10:])
                if recent_avg < self.cortex_td_threshold:
                    # Stable policy - gate Cortex
                    return False
            
            self.activation_counts[region_name] += 1
            return True
        
        # Hippocampus: Gate when sequences are short
        if region_name == 'hippocampus':
            sequence_length = context.get('sequence_length', 0)
            
            # Gate if sequence is too short for temporal patterns
            if sequence_length < self.hippocampus_sequence_threshold:
                return False
            
            self.activation_counts[region_name] += 1
            return True
        
        # Unknown region: activate by default
        return True
    
    def reset_episode(self):
        """Reset episode-specific state."""
        # Keep TD-error history across episodes (meta-learning)
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if self.total_steps == 0:
            return {
                'total_steps': 0,
                'activation_rates': {}
            }
        
        return {
            'total_steps': self.total_steps,
            'activation_rates': {
                region: count / self.total_steps
                for region, count in self.activation_counts.items()
            },
            'cortex_gating_rate': 1.0 - (self.activation_counts.get('cortex', 0) / self.total_steps),
            'hippocampus_gating_rate': 1.0 - (self.activation_counts.get('hippocampus', 0) / self.total_steps),
            'avg_recent_td_error': np.mean(self.td_error_history[-10:]) if self.td_error_history else 0.0
        }
