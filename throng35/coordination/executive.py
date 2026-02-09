"""
Executive Controller — Coordinate Brain Regions

The Executive manages multiple brain regions, routing information
appropriately and coordinating their interactions.

Phase C Enhancement: Adaptive routing with region gating.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from throng35.regions.region_base import RegionBase
from throng35.coordination.adaptive_router import AdaptiveRouter


class ExecutiveController:
    """
    Executive controller for coordinating brain regions.
    
    Phase C: Enhanced with adaptive routing and region gating.
    
    Features:
    - Routes information to appropriate regions
    - Manages region interactions
    - Adaptively gates regions for efficiency
    - Combines outputs for action selection
    """
    
    def __init__(self, 
                 regions: Dict[str, RegionBase],
                 enable_gating: bool = True,
                 router_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Executive controller.
        
        Args:
            regions: Dict mapping region names to RegionBase instances
            enable_gating: Enable adaptive region gating
            router_config: Configuration for AdaptiveRouter
        """
        self.regions = regions
        self.step_count = 0
        self.enable_gating = enable_gating
        
        # Adaptive router for gating
        if enable_gating:
            config = router_config or {}
            self.router = AdaptiveRouter(**config)
        else:
            self.router = None
        
        # Episode tracking
        self.episode_step = 0
    
    def step(self, 
             raw_observation: np.ndarray,
             reward: float,
             done: bool,
             activations: np.ndarray) -> Dict[str, Any]:
        """
        Coordinate all regions for one step.
        
        Args:
            raw_observation: Raw environment observation
            reward: Reward signal
            done: Episode termination flag
            activations: Neuron activations for Cortex
        
        Returns:
            Combined outputs from all regions
        """
        results = {}
        self.episode_step += 1
        
        # Striatum step (Q-learning) - ALWAYS ACTIVE
        if 'striatum' in self.regions:
            striatum_out = self.regions['striatum'].step({
                'raw_observation': raw_observation,
                'reward': reward,
                'done': done
            })
            results['striatum'] = striatum_out
            results['action'] = striatum_out['action']  # Primary action
            results['td_error'] = striatum_out.get('td_error', 0.0)
        
        # Build context for routing decisions
        context = {
            'td_error': results.get('td_error', 0.0),
            'sequence_length': 0,  # Will be updated by Hippocampus
            'episode_step': self.episode_step
        }
        
        # Cortex step (Hebbian) - GATED
        if 'cortex' in self.regions:
            should_activate = True
            if self.router:
                should_activate = self.router.should_activate('cortex', context)
            
            if should_activate:
                cortex_out = self.regions['cortex'].step({
                    'activations': activations,
                    'td_error': results.get('td_error', 0.0)
                })
                results['cortex'] = cortex_out
            else:
                # Region gated - use cached/default output
                results['cortex'] = {'gated': True}
        
        # Hippocampus step (STDP) - GATED
        if 'hippocampus' in self.regions:
            # Get current sequence length for gating decision
            if hasattr(self.regions['hippocampus'], 'sequence_buffer'):
                context['sequence_length'] = len(self.regions['hippocampus'].sequence_buffer)
            
            should_activate = True
            if self.router:
                should_activate = self.router.should_activate('hippocampus', context)
            
            if should_activate:
                hippocampus_out = self.regions['hippocampus'].step({
                    'state_representation': raw_observation,
                    'reward': reward,
                    'done': done
                })
                results['hippocampus'] = hippocampus_out
            else:
                # Region gated - skip expensive computation
                results['hippocampus'] = {'gated': True}
        
        self.step_count += 1
        return results
    
    def reset(self):
        """Reset all regions for new episode."""
        for region in self.regions.values():
            region.reset()
        
        if self.router:
            self.router.reset_episode()
        
        self.step_count = 0
        self.episode_step = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all regions and router."""
        stats = {'step_count': self.step_count}
        
        # Region stats
        for name, region in self.regions.items():
            if hasattr(region, 'get_stats'):
                stats[name] = region.get_stats()
        
        # Router stats
        if self.router:
            stats['router'] = self.router.get_stats()
        
        return stats
