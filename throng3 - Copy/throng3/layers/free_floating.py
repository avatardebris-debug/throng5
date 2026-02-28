"""
Free-Floating Neurons Layer

Neurons that exist outside the holographic structure and can wire freely
based on local activity and reward. Tests hypothesis that holographic
interference prevents local associative learning.
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from throng3.core.meta_layer import MetaLayer, SignalDirection, SignalType


@dataclass
class FreeFloatingConfig:
    """Configuration for free-floating neurons."""
    n_neurons: int = 256
    learning_rate: float = 0.01
    sparsity: float = 0.9  # 90% sparse connections
    activation: str = 'tanh'
    reward_modulation: bool = True
    hebbian_learning: bool = True


class FreeFloatingLayer(MetaLayer):
    """
    Free-floating neurons that wire independently of holographic structure.
    
    Key differences from Meta^0:
    - NOT projected into holographic space
    - Local Hebbian plasticity only
    - Direct reward modulation
    - Can form arbitrary associations
    - Reads from holographic but doesn't write to it
    """
    
    def __init__(self, config: Optional[FreeFloatingConfig] = None):
        # Convert config to dict for MetaLayer compatibility
        cfg = config or FreeFloatingConfig()
        config_dict = {
            'n_neurons': cfg.n_neurons,
            'learning_rate': cfg.learning_rate,
            'sparsity': cfg.sparsity,
            'activation': cfg.activation,
            'reward_modulation': cfg.reward_modulation,
            'hebbian_learning': cfg.hebbian_learning,
            'global_gate': 1.0,  # Required by fractal stack
        }
        super().__init__(level=0.5, name="FreeFloating", config=config_dict)
        self.ff_config = cfg  # Keep original config
        
        self.n = cfg.n_neurons
        self.activations = np.zeros(self.n)
        self.prev_activations = np.zeros(self.n)
        
        # Local recurrent weights (sparse)
        self.W_local = self._init_sparse_weights(self.n, self.n)
        
        # Connections to/from holographic layer
        self.W_from_holo = np.random.randn(self.n, 128) * 0.01
        self.W_to_output = None  # Set dynamically based on output size
        
        # Learning state
        self.prev_reward = 0.0
        
    def _init_sparse_weights(self, n_post, n_pre):
        """Initialize sparse random weights."""
        W = np.random.randn(n_post, n_pre) * (0.1 / np.sqrt(n_pre))
        
        # Apply sparsity mask
        mask = np.random.random((n_post, n_pre)) > self.ff_config.sparsity
        W *= mask
        
        # No self-connections
        if n_post == n_pre:
            np.fill_diagonal(W, 0)
        
        return W
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update free-floating neurons with local Hebbian learning.
        
        Key: Does NOT write to holographic state, only reads from it.
        """
        self.process_inbox()
        
        # Get holographic state (read-only)
        holo_state = context.get('holographic_state', np.zeros(128))
        reward = context.get('reward', 0.0)
        
        # Compute activations
        holo_input = self.W_from_holo @ holo_state
        local_input = self.W_local @ self.activations
        
        combined = holo_input + local_input
        self.activations = self._activation_fn(combined)
        
        # Local Hebbian learning (reward-modulated)
        if self.ff_config.hebbian_learning and self._optimization_step > 0:
            self._apply_local_hebbian(reward)
        
        # Store for next iteration
        self.prev_activations = self.activations.copy()
        self.prev_reward = reward
        
        # Update metrics
        sparsity = np.sum(np.abs(self.W_local) > 1e-6) / self.W_local.size
        self.metrics.update(
            loss=1.0 - np.abs(reward),  # Proxy loss
            accuracy=max(0, reward)  # Proxy accuracy
        )
        
        # Signal UP (performance only, not state)
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'sparsity': sparsity,
                'mean_activation': np.mean(np.abs(self.activations)),
                'active_neurons': np.sum(np.abs(self.activations) > 0.1),
            }
        )
        
        return {
            'activations': self.activations,
            'sparsity': sparsity,
            'metrics': self.metrics,
        }
    
    def _apply_local_hebbian(self, reward: float):
        """
        Apply local Hebbian learning with reward modulation.
        
        Key: This is PURE local plasticity, no holographic interference.
        """
        pre = self.prev_activations
        post = self.activations
        
        # Hebbian update: dW = pre × post
        dW = np.outer(post, pre)
        
        # Reward modulation (three-factor learning)
        reward_factor = np.clip(reward, -1.0, 1.0)
        dW *= reward_factor * self.ff_config.learning_rate
        
        # Apply update
        self.W_local += dW
        
        # Maintain sparsity (prune small weights)
        threshold = 0.001
        self.W_local[np.abs(self.W_local) < threshold] = 0
        
        # Clip weights
        self.W_local = np.clip(self.W_local, -5.0, 5.0)
    
    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        fn = self.ff_config.activation
        if fn == 'relu':
            return np.maximum(0, x)
        elif fn == 'tanh':
            return np.tanh(x)
        elif fn == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        return np.tanh(x)
    
    def get_output_contribution(self, n_outputs: int) -> np.ndarray:
        """
        Get this layer's contribution to the output.
        
        Free-floating neurons can contribute directly to output,
        bypassing holographic layer.
        """
        if self.W_to_output is None or self.W_to_output.shape != (n_outputs, self.n):
            self.W_to_output = np.random.randn(n_outputs, self.n) * 0.01
        
        return self.W_to_output @ self.activations
    
    def update_output_weights(self, output_error: np.ndarray, learning_rate: float = 0.01):
        """
        Update output weights based on error signal.
        
        This allows free-floating neurons to learn output associations.
        """
        if self.W_to_output is None:
            return
        
        # Gradient descent on output weights
        dW = np.outer(output_error, self.activations) * learning_rate
        self.W_to_output += dW
        
        # Clip
        self.W_to_output = np.clip(self.W_to_output, -5.0, 5.0)
    
    # ================================================================
    # Abstract method implementations (required by MetaLayer)
    # ================================================================
    
    def _compute_state_vector(self) -> np.ndarray:
        """Compute state vector for holographic encoding."""
        # Return activations + weight statistics
        weight_stats = np.array([
            np.mean(np.abs(self.W_local)),
            np.std(self.W_local),
            np.sum(np.abs(self.W_local) > 1e-6) / self.W_local.size,  # sparsity
        ])
        return np.concatenate([self.activations[:64], weight_stats])
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestion from another layer."""
        # Free-floating neurons ignore suggestions (autonomous)
        return False
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> tuple:
        """Evaluate suggestion."""
        # Always reject (autonomous layer)
        return (0.0, "Free-floating layer operates autonomously")
    
    def _self_optimize_weights(self):
        """Weight-level optimization (done in _apply_local_hebbian)."""
        pass
    
    def _self_optimize_synapses(self):
        """Synapse-level optimization (sparsity pruning)."""
        # Already done in _apply_local_hebbian
        pass
    
    def _self_optimize_neurons(self):
        """Neuron-level optimization (could add neurogenesis here)."""
        pass
    
    def snapshot(self) -> Dict[str, Any]:
        """Override snapshot to handle float level."""
        state_vector = self._compute_state_vector()
        
        # Use integer seed for RandomState (convert float level to int)
        seed = int(self.level * 100) * 42 + 7
        
        if len(state_vector) > self._holographic_dim:
            rng = np.random.RandomState(seed)
            projection = rng.randn(self._holographic_dim, len(state_vector))
            projection /= np.sqrt(self._holographic_dim)
            holographic = projection @ state_vector
        else:
            holographic = state_vector
        
        return {
            "level": self.level,
            "name": self.name,
            "metrics": {
                "loss": self.metrics.loss,
                "accuracy": self.metrics.accuracy,
                "efficiency": self.metrics.efficiency,
                "stability": self.metrics.stability,
                "improvement_rate": self.metrics.improvement_rate,
                "step_count": self.metrics.step_count,
                "n_parameters": self.metrics.n_parameters,
                "n_active_connections": self.metrics.n_active_connections,
            },
            "state_vector": holographic,
            "config": self.config.copy(),
            "optimization_step": self._optimization_step,
            "timestamp": time.time(),
        }
