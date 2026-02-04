"""
Meta^0: NeuronLayer — Raw Neural Substrate

The foundation layer. Contains the actual neurons, weights, and
activations that process information. All other meta-layers
ultimately modify this layer's behavior.

Features:
- Sparse weight matrix with KDTree spatial indexing (from throng2)
- Event-based activation (only process when input changes)
- Multiple neuron types (excitatory, inhibitory)
- Configurable activation functions
- Spatial organization (neurons have positions)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class NeuronConfig:
    """Configuration for the neuron layer."""
    n_neurons: int = 1000
    n_inputs: int = 64
    n_outputs: int = 32
    sparsity: float = 0.9           # Fraction of zero weights
    exc_ratio: float = 0.8          # Fraction excitatory (vs inhibitory)
    activation: str = 'relu'        # relu, tanh, sigmoid, linear
    use_spatial: bool = True        # Spatial organization
    spatial_dims: int = 3           # 2D or 3D spatial layout
    dt: float = 0.001              # Simulation timestep
    tau_membrane: float = 0.020    # Membrane time constant
    threshold: float = 1.0         # Spike threshold
    reset: float = 0.0             # Reset potential after spike
    refractory: float = 0.002     # Refractory period


class NeuronLayer(MetaLayer):
    """
    Meta^0: Raw neural substrate.
    
    The lowest meta-level. Contains neurons, weights, and processes
    input to produce output. All other layers act ON this layer.
    """
    
    def __init__(self, config: Optional[NeuronConfig] = None, **kwargs):
        cfg = config or NeuronConfig()
        super().__init__(level=0, name="NeuronLayer", config=vars(cfg))
        self.neuron_config = cfg
        
        # Initialize neurons
        self.n = cfg.n_neurons
        self.n_inputs = cfg.n_inputs
        self.n_outputs = cfg.n_outputs
        
        # State vectors
        self.membrane_potential = np.zeros(self.n)
        self.activations = np.zeros(self.n)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.refractory_timer = np.zeros(self.n)
        
        # Neuron types: 1 = excitatory, -1 = inhibitory
        self.neuron_type = np.ones(self.n)
        n_inh = int(self.n * (1 - cfg.exc_ratio))
        self.neuron_type[-n_inh:] = -1
        
        # Weight matrices
        self._init_weights()
        
        # Spatial positions (for KDTree-style queries)
        if cfg.use_spatial:
            self.positions = np.random.rand(self.n, cfg.spatial_dims)
        else:
            self.positions = None
        
        # Input/output projection matrices
        self.W_in = np.random.randn(self.n, cfg.n_inputs) * 0.1
        self.W_out = np.random.randn(cfg.n_outputs, self.n) * 0.1
        
        # Event tracking
        self._spike_history: deque = deque(maxlen=1000)
        self._activity_history: deque = deque(maxlen=100)
        self._time = 0.0
        
        # Update metrics
        self.metrics.n_parameters = (
            self.W_recurrent.size + self.W_in.size + self.W_out.size
        )
        self.metrics.n_active_connections = int(
            np.sum(np.abs(self.W_recurrent) > 1e-8)
        )
    
    def _init_weights(self):
        """Initialize sparse recurrent weight matrix."""
        cfg = self.neuron_config
        
        # Create sparse random matrix
        self.W_recurrent = np.random.randn(self.n, self.n) * (0.1 / np.sqrt(self.n))
        
        # Apply sparsity mask
        mask = np.random.random((self.n, self.n)) > cfg.sparsity
        self.W_recurrent *= mask
        
        # No self-connections
        np.fill_diagonal(self.W_recurrent, 0)
        
        # Apply Dale's law (neuron type determines sign)
        for i in range(self.n):
            if self.neuron_type[i] > 0:
                self.W_recurrent[i] = np.abs(self.W_recurrent[i])
            else:
                self.W_recurrent[i] = -np.abs(self.W_recurrent[i])
    
    def _activation_fn(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        fn = self.neuron_config.activation
        if fn == 'relu':
            return np.maximum(0, x)
        elif fn == 'tanh':
            return np.tanh(x)
        elif fn == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        elif fn == 'linear':
            return x
        return np.maximum(0, x)
    
    # ================================================================
    # MetaLayer interface implementation
    # ================================================================
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one forward pass / optimization step.
        
        Processes any pending input and updates neuron states.
        """
        self.process_inbox()
        
        # Get input from context
        input_data = context.get('input', np.zeros(self.n_inputs))
        if len(input_data) != self.n_inputs:
            input_data = np.zeros(self.n_inputs)
        
        # Forward pass
        output = self.forward(input_data)
        
        # Compute loss if target provided
        target = context.get('target', None)
        loss = 0.0
        if target is not None and len(target) == self.n_outputs:
            loss = float(np.mean((output - target) ** 2))
            accuracy = float(1.0 - min(loss, 1.0))
        else:
            accuracy = float(np.mean(self.activations > 0))  # Activity as proxy
        
        self.metrics.update(loss, accuracy)
        self.metrics.complexity = float(self.metrics.n_active_connections) / max(self.n ** 2, 1)
        
        # Signal performance UP
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'loss': loss,
                'accuracy': accuracy,
                'mean_activity': float(np.mean(self.activations)),
                'spike_rate': float(np.mean(self.spikes)),
                'n_active': int(np.sum(self.activations > 0)),
            },
        )
        
        return {
            'output': output,
            'loss': loss,
            'accuracy': accuracy,
            'spike_rate': float(np.mean(self.spikes)),
            'metrics': self.metrics,
        }
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neuron layer.
        
        Args:
            input_data: (n_inputs,) input vector
            
        Returns:
            (n_outputs,) output vector
        """
        cfg = self.neuron_config
        dt = cfg.dt
        
        # Input current
        I_input = self.W_in @ input_data
        
        # Recurrent current
        I_recurrent = self.W_recurrent @ self.activations
        
        # Total current
        I_total = I_input + I_recurrent
        
        # Update membrane potential (leaky integrate)
        decay = np.exp(-dt / cfg.tau_membrane)
        self.membrane_potential = self.membrane_potential * decay + I_total * (1 - decay)
        
        # Refractory period
        self.refractory_timer = np.maximum(0, self.refractory_timer - dt)
        refractory_mask = self.refractory_timer <= 0
        
        # Spike detection
        self.spikes = (self.membrane_potential >= cfg.threshold) & refractory_mask
        
        # Reset spiking neurons
        self.membrane_potential[self.spikes] = cfg.reset
        self.refractory_timer[self.spikes] = cfg.refractory
        
        # Activations (rate-coded from membrane potential)
        self.activations = self._activation_fn(self.membrane_potential) * refractory_mask
        
        # Output projection
        output = self.W_out @ self.activations
        
        # Track
        self._time += dt
        self._activity_history.append(float(np.mean(self.activations)))
        
        return output
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get all weight matrices (for Meta^1 to modify)."""
        return {
            'W_recurrent': self.W_recurrent,
            'W_in': self.W_in,
            'W_out': self.W_out,
        }
    
    def set_weights(self, name: str, weights: np.ndarray):
        """Set a weight matrix (called by Meta^1)."""
        if name == 'W_recurrent':
            self.W_recurrent = weights
        elif name == 'W_in':
            self.W_in = weights
        elif name == 'W_out':
            self.W_out = weights
        
        self.metrics.n_active_connections = int(
            np.sum(np.abs(self.W_recurrent) > 1e-8)
        )
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state: compressed representation of neuron state."""
        # Combine key statistics into a fixed-size vector
        state = np.concatenate([
            [np.mean(self.activations), np.std(self.activations)],
            [np.mean(self.membrane_potential), np.std(self.membrane_potential)],
            [float(np.mean(self.spikes))],
            [np.mean(np.abs(self.W_recurrent)), np.std(self.W_recurrent.ravel())],
            [np.mean(np.abs(self.W_in)), np.mean(np.abs(self.W_out))],
            # Spatial stats of weights (eigenvalue-like features)
            np.sort(np.abs(np.mean(self.W_recurrent, axis=0)))[-10:],
            # Recent activity pattern
            np.array(list(self._activity_history))[-20:] if self._activity_history else np.zeros(20),
        ])
        return state
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply a parameter change suggestion."""
        applied = False
        
        if 'W_recurrent_delta' in suggestion:
            self.W_recurrent += suggestion['W_recurrent_delta']
            applied = True
        
        if 'W_in_delta' in suggestion:
            self.W_in += suggestion['W_in_delta']
            applied = True
        
        if 'W_out_delta' in suggestion:
            self.W_out += suggestion['W_out_delta']
            applied = True
        
        if 'activation' in suggestion:
            self.neuron_config.activation = suggestion['activation']
            applied = True
        
        if 'threshold' in suggestion:
            self.neuron_config.threshold = suggestion['threshold']
            applied = True
        
        if applied:
            self.metrics.n_active_connections = int(
                np.sum(np.abs(self.W_recurrent) > 1e-8)
            )
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate a suggestion before applying."""
        score = 0.5
        reasons = []
        
        if 'W_recurrent_delta' in suggestion:
            delta = suggestion['W_recurrent_delta']
            # Reject very large changes
            magnitude = np.mean(np.abs(delta))
            current_mag = np.mean(np.abs(self.W_recurrent))
            ratio = magnitude / max(current_mag, 1e-8)
            
            if ratio > 0.5:
                score = 0.2
                reasons.append(f"Weight change too large ({ratio:.2f}x current)")
            elif ratio < 0.01:
                score = 0.7
                reasons.append("Moderate weight change")
            else:
                score = 0.8
                reasons.append(f"Reasonable weight change ({ratio:.3f}x)")
        
        if 'threshold' in suggestion:
            new_thresh = suggestion['threshold']
            if 0.1 <= new_thresh <= 10.0:
                score = max(score, 0.7)
                reasons.append(f"Threshold {new_thresh} in range")
            else:
                score = 0.1
                reasons.append(f"Threshold {new_thresh} out of range")
        
        return score, "; ".join(reasons) if reasons else "No specific evaluation"
    
    def _self_optimize_weights(self):
        """Weight-level: normalize and decay weights."""
        # Weight decay
        self.W_recurrent *= 0.9999
        
        # Clip extreme values
        self.W_recurrent = np.clip(self.W_recurrent, -5.0, 5.0)
    
    def _self_optimize_synapses(self):
        """Synapse-level: handled by Meta^1, but do basic cleanup."""
        # Remove near-zero weights below noise floor
        noise_floor = 1e-6
        mask = np.abs(self.W_recurrent) > noise_floor
        self.W_recurrent *= mask
    
    def _self_optimize_neurons(self):
        """Neuron-level: no structural changes at Meta^0 (that's Meta^5's job)."""
        pass
