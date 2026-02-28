"""
Cortex Region — Hebbian Learning for Pattern Recognition

The Cortex region handles feature learning using Hebbian plasticity.
It learns patterns from neuron activations, modulated by reward signals.
"""

from typing import Any, Dict, Optional
import numpy as np
import time
import sys

from throng35.regions.region_base import RegionBase
from throng35.learning.hebbian import HebbianRule, HebbianConfig


class CortexRegion(RegionBase):
    """
    Cortex region implementing Hebbian learning.
    
    Key features:
    - Learns from neuron activations (not raw observations)
    - Pattern-based learning (correlation strengthening)
    - Modulated by TD-error from Striatum (reward prediction)
    """
    
    def __init__(self,
                 n_neurons: int = 100,
                 n_features: int = 10,
                 config: Optional[HebbianConfig] = None):
        """
        Initialize Cortex region.
        
        Args:
            n_neurons: Number of neurons to process
            n_features: Number of feature patterns to extract
            config: Hebbian learning configuration
        """
        super().__init__(region_name="Cortex")
        
        self.n_neurons = n_neurons
        self.n_features = n_features
        
        # Hebbian learning component
        self.hebbian = HebbianRule(config or HebbianConfig())
        
        # Feature extraction weights (neurons → features)
        self.feature_weights = np.random.randn(n_features, n_neurons) * 0.01
        
        # Pattern history for analysis
        self.pattern_history = []
        self.max_history = 100
        
        # Resource tracking (for future optimization)
        self._step_times = []
        self._update_counts = []
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in Cortex.
        
        Expected inputs:
            - 'activations': Neuron activation patterns
            - 'td_error': TD-error from Striatum (optional, for modulation)
        
        Returns:
            - 'features': Extracted feature vector
            - 'patterns': Learned pattern strengths
        """
        t0 = time.time()  # Resource tracking
        
        activations = region_input.get('activations', np.zeros(self.n_neurons))
        td_error = region_input.get('td_error', 0.0)
        
        # Ensure correct size
        if len(activations) != self.n_neurons:
            # Pad or truncate
            if len(activations) < self.n_neurons:
                activations = np.pad(activations, (0, self.n_neurons - len(activations)))
            else:
                activations = activations[:self.n_neurons]
        
        # Extract features (activation → feature space)
        features = self.feature_weights @ activations
        features = np.tanh(features)  # Nonlinearity
        
        # Hebbian update (modulated by reward prediction error)
        # Positive TD-error = better than expected, strengthen patterns
        # Negative TD-error = worse than expected, weaken patterns
        modulation = 1.0 + 0.1 * np.clip(td_error, -1, 1)
        
        # Use batch_update: weights, pre_activity, post_activity
        # Here: pre=activations (input), post=features (output)
        weight_changes = self.hebbian.batch_update(
            self.feature_weights,
            activations,
            features
        )
        
        # Apply modulated updates
        self.feature_weights += weight_changes * modulation
        
        # Track patterns
        pattern_strength = np.linalg.norm(features)
        self.pattern_history.append(pattern_strength)
        if len(self.pattern_history) > self.max_history:
            self.pattern_history.pop(0)
        
        # Resource tracking
        self._step_times.append(time.time() - t0)
        self._update_counts.append(1)  # Always updates
        if len(self._step_times) > 100:
            self._step_times.pop(0)
            self._update_counts.pop(0)
        
        return {
            'features': features,
            'pattern_strength': pattern_strength,
            'avg_pattern_strength': np.mean(self.pattern_history) if self.pattern_history else 0.0,
            'feature_weights_norm': np.linalg.norm(self.feature_weights)
        }
    
    def reset(self):
        """Reset Cortex for new episode."""
        # Keep learned weights (they're meta-knowledge)
        # Only reset episodic pattern history
        self.pattern_history = []
    
    def get_state_signature(self) -> Dict[str, Any]:
        """Return input/output signature for this region."""
        return {
            'inputs': {
                'activations': {
                    'type': np.ndarray,
                    'required': True,
                    'shape': (self.n_neurons,),
                    'description': 'Neuron activation patterns'
                },
                'td_error': {
                    'type': float,
                    'required': False,
                    'description': 'TD-error from Striatum for modulation'
                }
            },
            'outputs': {
                'features': {
                    'type': np.ndarray,
                    'shape': (self.n_features,),
                    'description': 'Extracted feature vector'
                },
                'pattern_strength': {
                    'type': float,
                    'description': 'Current pattern strength'
                },
                'avg_pattern_strength': {
                    'type': float,
                    'description': 'Average pattern strength over history'
                },
                'feature_weights_norm': {
                    'type': float,
                    'description': 'Frobenius norm of feature weights'
                }
            }
        }
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Return resource usage metrics (enables future optimization)."""
        return {
            'compute_ms': np.mean(self._step_times) * 1000 if self._step_times else 0.0,
            'memory_mb': sys.getsizeof(self.feature_weights) / 1024**2,
            'updates_per_step': np.mean(self._update_counts) if self._update_counts else 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Cortex statistics."""
        return {
            'region': 'Cortex',
            'n_neurons': self.n_neurons,
            'n_features': self.n_features,
            'avg_pattern_strength': np.mean(self.pattern_history) if self.pattern_history else 0.0,
            'feature_weights_norm': np.linalg.norm(self.feature_weights),
            'pattern_history_len': len(self.pattern_history),
            'resource_usage': self.get_resource_usage()
        }
