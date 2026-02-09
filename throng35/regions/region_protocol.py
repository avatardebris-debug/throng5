"""
Region Protocol — Stable Interface Contract

This protocol defines the minimal stable contract that all regions must implement.
These interfaces MUST NOT change - they are the foundation for future optimizations.
"""

from typing import Protocol, Dict, Any, runtime_checkable
import numpy as np


@runtime_checkable
class RegionProtocol(Protocol):
    """
    Minimal stable contract for all brain regions.
    
    This protocol ensures we can:
    1. Swap region implementations without breaking code
    2. Add optimization wrappers transparently
    3. Benchmark different approaches
    4. Add efficiency optimizations later (Nash pruning, adaptive gating, etc.)
    
    DO NOT CHANGE THIS INTERFACE - extend with new protocols if needed.
    """
    
    region_name: str
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in this region.
        
        Must be idempotent for the same input (same input → same output).
        
        Args:
            region_input: Dict with region-specific inputs
            
        Returns:
            Dict with at least these keys:
            - Primary region outputs (region-specific)
            - 'metrics': Optional performance metrics
        """
        ...
    
    def reset(self) -> None:
        """
        Reset episodic state for new episode.
        
        MUST preserve meta-knowledge (learned weights, patterns, etc.).
        Only resets temporary/episodic state.
        """
        ...
    
    def get_state_signature(self) -> Dict[str, Any]:
        """
        Return expected input/output signatures.
        
        Enables:
        - Automatic validation
        - Route optimization in Executive
        - Compatibility checking between regions
        
        Returns:
            {
                'inputs': {
                    'key_name': {'type': type, 'required': bool, 'shape': tuple},
                    ...
                },
                'outputs': {
                    'key_name': {'type': type, 'shape': tuple},
                    ...
                }
            }
        """
        ...
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Return current resource usage metrics.
        
        Enables future optimizations:
        - Adaptive pruning (Nash pruning from Throng1)
        - Load balancing across regions
        - Efficiency optimization
        - Adaptive gating (skip expensive regions when not needed)
        
        Returns:
            {
                'compute_ms': Average compute time per step (milliseconds),
                'memory_mb': Approximate memory usage (megabytes),
                'updates_per_step': Average number of learning updates per step
            }
        """
        ...


@runtime_checkable
class OptimizableRegion(Protocol):
    """
    Extended protocol for regions that support optimization.
    
    This is OPTIONAL - only implement if region can benefit from:
    - Weight pruning
    - Connection sparsification
    - State compression
    """
    
    def get_weights(self) -> Any:
        """Return region weights for inspection/pruning."""
        ...
    
    def set_weights(self, weights: Any) -> None:
        """Set region weights (e.g., after pruning)."""
        ...
    
    def get_weight_importance(self) -> np.ndarray:
        """
        Return importance score for each weight.
        
        Enables Nash pruning - identify low-contribution connections.
        """
        ...
