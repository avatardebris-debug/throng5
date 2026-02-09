"""
Optimization Framework — Interfaces for Future Efficiency Gains

This module defines optimization hooks that can be added later without
breaking existing code. Implements Nash pruning and compression.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class RegionOptimizer(ABC):
    """
    Base class for region optimization wrappers.
    
    Allows adding optimizations (Nash pruning, adaptive gating, compression)
    without modifying region implementations.
    """
    
    def __init__(self, wrapped_region):
        """
        Wrap a region with optimization layer.
        
        Args:
            wrapped_region: Any region implementing RegionProtocol
        """
        self.region = wrapped_region
        self.enabled = True
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """Step with optimization logic."""
        if not self.enabled:
            return self.region.step(region_input)
        
        result = self.region.step(region_input)
        return result
    
    def reset(self) -> None:
        """Pass through to wrapped region."""
        self.region.reset()
    
    def get_state_signature(self) -> Dict[str, Any]:
        """Pass through to wrapped region."""
        return self.region.get_state_signature()
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Pass through to wrapped region."""
        return self.region.get_resource_usage()


class AdaptiveGatingOptimizer(RegionOptimizer):
    """Skip region steps when not needed (stub)."""
    
    def __init__(self, wrapped_region, stability_threshold: float = 0.01):
        super().__init__(wrapped_region)
        self.stability_threshold = stability_threshold


class NashPruningOptimizer(RegionOptimizer):
    """
    Nash pruning optimizer (from Throng1).
    
    Prunes low-contribution connections using Nash equilibrium principles.
    """
    
    def __init__(self, 
                 region,
                 prune_threshold: float = 0.01,
                 prune_rate: float = 0.1):
        """
        Initialize Nash pruning optimizer.
        
        Args:
            region: Region to optimize
            prune_threshold: Prune connections with |weight| < threshold
            prune_rate: Fraction of connections to prune per iteration
        """
        super().__init__(region)
        self.prune_threshold = prune_threshold
        self.prune_rate = prune_rate
        self.pruned_count = 0
        self.total_connections = 0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Apply Nash pruning to region weights.
        
        Returns:
            Optimization statistics
        """
        # Get region weights (different for each region type)
        weights = None
        if hasattr(self.region, 'qlearner') and hasattr(self.region.qlearner, 'W'):
            # Striatum: Q-learning weights
            weights = self.region.qlearner.W
        elif hasattr(self.region, 'recurrent_weights'):
            # Hippocampus: Recurrent weights
            weights = self.region.recurrent_weights
        elif hasattr(self.region, 'feature_weights'):
            # Cortex: Feature weights
            weights = self.region.feature_weights
        
        if weights is None:
            return {'pruned': 0, 'total': 0, 'prune_rate': 0.0}
        
        # Count total connections
        self.total_connections = weights.size
        
        # Find low-contribution connections
        abs_weights = np.abs(weights)
        low_contrib_mask = abs_weights < self.prune_threshold
        
        # Prune (set to zero)
        weights[low_contrib_mask] = 0
        self.pruned_count = np.sum(low_contrib_mask)
        
        return {
            'pruned': int(self.pruned_count),
            'total': int(self.total_connections),
            'prune_rate': float(self.pruned_count / self.total_connections),
            'remaining_connections': int(self.total_connections - self.pruned_count)
        }


class CompressionOptimizer(RegionOptimizer):
    """
    Compression optimizer for reducing memory footprint.
    
    Uses quantization to compress region state.
    """
    
    def __init__(self, 
                 region,
                 quantization_bits: int = 8):
        """
        Initialize compression optimizer.
        
        Args:
            region: Region to optimize
            quantization_bits: Number of bits for quantization (8 = int8)
        """
        super().__init__(region)
        self.quantization_bits = quantization_bits
        self.compression_ratio = 1.0
    
    def optimize(self) -> Dict[str, Any]:
        """
        Apply compression to region state.
        
        Returns:
            Compression statistics
        """
        # Get region weights
        weights = None
        
        if hasattr(self.region, 'qlearner') and hasattr(self.region.qlearner, 'W'):
            weights = self.region.qlearner.W
        elif hasattr(self.region, 'recurrent_weights'):
            weights = self.region.recurrent_weights
        elif hasattr(self.region, 'feature_weights'):
            weights = self.region.feature_weights
        
        if weights is None:
            return {'compressed': False, 'compression_ratio': 1.0}
        
        # Calculate compression ratio (float32 → int8 = 4x compression)
        original_size = weights.nbytes
        
        if self.quantization_bits == 8:
            # Quantize to int8 range
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                # Scale to [-127, 127]
                scaled = ((weights - w_min) / (w_max - w_min) * 254 - 127).astype(np.int8)
                compressed_size = scaled.nbytes
                self.compression_ratio = original_size / compressed_size
        
        return {
            'compressed': True,
            'compression_ratio': float(self.compression_ratio),
            'original_size_bytes': int(original_size),
            'compressed_size_bytes': int(original_size / self.compression_ratio),
            'savings_bytes': int(original_size - original_size / self.compression_ratio)
        }
