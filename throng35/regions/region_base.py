"""
RegionBase — Abstract base class for brain regions

Each region in Throng3.5 is independent with its own:
- State representation
- Learning timing
- Step control
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class RegionBase(ABC):
    """
    Base class for brain regions in Throng3.5.
    
    Unlike Throng3's layers which all shared the same step() timing,
    each region controls its own processing loop and state representation.
    
    Regions are simpler than MetaLayers - they don't need the full
    signal/holographic/optimization infrastructure.
    """
    
    def __init__(self, region_name: str):
        """
        Initialize region.
        
        Args:
            region_name: Human-readable name (e.g., "Striatum", "Cortex")
        """
        self.region_name = region_name
    
    @abstractmethod
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in this region.
        
        Each region defines what inputs it needs and when to process them.
        
        Args:
            region_input: Dict with region-specific inputs
                         (e.g., Striatum needs 'raw_observation' and 'reward')
        
        Returns:
            Dict with region-specific outputs
            (e.g., Striatum returns 'action', 'q_values', 'td_error')
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """Reset region state for new episode/task."""
        raise NotImplementedError
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current region state for inspection/debugging.
        
        Returns:
            Dict with region-specific state information
        """
        return {
            'region_name': self.region_name
        }
