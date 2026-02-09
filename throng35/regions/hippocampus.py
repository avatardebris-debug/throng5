"""
Hippocampus Region — Sequence and Episodic Memory

Implements STDP-based learning for:
- Temporal sequences (action sequences, state transitions)
- Episodic memory (remembering successful paths)
- Context-dependent recall

Key features:
- STDP timing: neurons that fire together, wire together
- Sequence prediction: predict next state given current state
- Episodic replay: strengthen successful trajectories
"""

from typing import Any, Dict, Optional, List
import numpy as np
import time
import sys

from throng35.regions.region_base import RegionBase
from throng35.learning.stdp import STDPRule, STDPConfig


class HippocampusRegion(RegionBase):
    """
    Hippocampus region implementing STDP for sequence learning.
    
    Key features:
    - Learns temporal sequences (state → next_state)
    - Episodic memory (stores successful trajectories)
    - Context-dependent recall
    - STDP timing-dependent plasticity
    """
    
    def __init__(self,
                 n_neurons: int = 200,
                 sequence_length: int = 10,
                 config: Optional[STDPConfig] = None):
        """
        Initialize Hippocampus region.
        
        Args:
            n_neurons: Number of neurons for sequence representation
            sequence_length: Maximum sequence length to track
            config: STDP learning configuration
        """
        super().__init__(region_name="Hippocampus")
        
        self.n_neurons = n_neurons
        self.sequence_length = sequence_length
        
        # STDP learning component
        self.stdp = STDPRule(config or STDPConfig())
        
        # Recurrent weights for sequence learning (neuron → neuron)
        self.recurrent_weights = np.random.randn(n_neurons, n_neurons) * 0.01
        np.fill_diagonal(self.recurrent_weights, 0)  # No self-connections
        
        # Input weights (state → neurons)
        # Will be sized dynamically based on first input
        self.input_weights = None
        
        # Sequence buffer (stores recent activations)
        self.sequence_buffer: List[np.ndarray] = []
        self.max_buffer_size = sequence_length
        
        # Episodic memory (stores successful sequences)
        self.episodic_memory: List[List[np.ndarray]] = []
        self.max_episodes = 100
        
        # Current episode buffer
        self.current_episode: List[np.ndarray] = []
        
        # Resource tracking
        self._step_times = []
        self._update_counts = []
    
    def step(self, region_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step in Hippocampus.
        
        Expected inputs:
            - 'state_representation': Current state vector
            - 'reward': Reward signal (for episodic memory)
            - 'done': Episode termination flag
        
        Returns:
            - 'sequence_prediction': Predicted next state
            - 'episodic_match': Similarity to stored episodes
            - 'activations': Current neuron activations
        """
        t0 = time.time()
        
        state = region_input.get('state_representation', np.zeros(10))
        reward = region_input.get('reward', 0.0)
        done = region_input.get('done', False)
        
        # Initialize input weights on first call
        if self.input_weights is None:
            self.input_weights = np.random.randn(self.n_neurons, len(state)) * 0.01
        
        # Compute current activations (state → neurons)
        activations = self.input_weights @ state
        
        # Add recurrent contribution (previous activations → current)
        if len(self.sequence_buffer) > 0:
            prev_activations = self.sequence_buffer[-1]
            recurrent_input = self.recurrent_weights @ prev_activations
            activations += recurrent_input
        
        # Nonlinearity
        activations = np.tanh(activations)
        
        # STDP update (strengthen temporal connections)
        did_update = False
        if len(self.sequence_buffer) > 0:
            prev_activations = self.sequence_buffer[-1]
            
            # STDP: strengthen connections from prev → current
            weight_changes = self.stdp.batch_update(
                self.recurrent_weights,
                prev_activations,
                activations
            )
            
            self.recurrent_weights += weight_changes
            did_update = True
        
        # Add to sequence buffer
        self.sequence_buffer.append(activations.copy())
        if len(self.sequence_buffer) > self.max_buffer_size:
            self.sequence_buffer.pop(0)
        
        # Add to current episode
        self.current_episode.append(activations.copy())
        
        # Predict next state (using recurrent weights)
        sequence_prediction = self.recurrent_weights @ activations
        
        # Check episodic memory match
        episodic_match = self._compute_episodic_match(activations)
        
        # On episode end, store if successful
        if done:
            if reward > 0:  # Successful episode
                self._store_episode()
            self.current_episode = []
            self.sequence_buffer = []
        
        # Resource tracking
        self._step_times.append(time.time() - t0)
        self._update_counts.append(1 if did_update else 0)
        if len(self._step_times) > 100:
            self._step_times.pop(0)
            self._update_counts.pop(0)
        
        return {
            'sequence_prediction': sequence_prediction,
            'episodic_match': episodic_match,
            'activations': activations,
            'sequence_length': len(self.sequence_buffer)
        }
    
    def _compute_episodic_match(self, current_activations: np.ndarray) -> float:
        """
        Compute similarity to stored episodic memories.
        
        Returns average cosine similarity to stored episodes.
        """
        if len(self.episodic_memory) == 0:
            return 0.0
        
        similarities = []
        for episode in self.episodic_memory:
            # Compare to each activation in episode
            for stored_activation in episode:
                # Cosine similarity
                norm_current = np.linalg.norm(current_activations)
                norm_stored = np.linalg.norm(stored_activation)
                if norm_current > 0 and norm_stored > 0:
                    similarity = np.dot(current_activations, stored_activation) / (norm_current * norm_stored)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _store_episode(self):
        """Store current episode in episodic memory."""
        if len(self.current_episode) > 0:
            self.episodic_memory.append(self.current_episode.copy())
            
            # Limit memory size
            if len(self.episodic_memory) > self.max_episodes:
                self.episodic_memory.pop(0)
    
    def reset(self):
        """Reset Hippocampus for new episode."""
        # Keep learned weights and episodic memory (meta-knowledge)
        # Only reset episodic buffers
        self.sequence_buffer = []
        self.current_episode = []
    
    def get_state_signature(self) -> Dict[str, Any]:
        """Return input/output signature for this region."""
        return {
            'inputs': {
                'state_representation': {
                    'type': np.ndarray,
                    'required': True,
                    'description': 'Current state vector'
                },
                'reward': {
                    'type': float,
                    'required': False,
                    'description': 'Reward signal for episodic storage'
                },
                'done': {
                    'type': bool,
                    'required': False,
                    'description': 'Episode termination flag'
                }
            },
            'outputs': {
                'sequence_prediction': {
                    'type': np.ndarray,
                    'shape': (self.n_neurons,),
                    'description': 'Predicted next state activations'
                },
                'episodic_match': {
                    'type': float,
                    'description': 'Similarity to stored episodes'
                },
                'activations': {
                    'type': np.ndarray,
                    'shape': (self.n_neurons,),
                    'description': 'Current neuron activations'
                },
                'sequence_length': {
                    'type': int,
                    'description': 'Current sequence buffer length'
                }
            }
        }
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Return resource usage metrics (enables future optimization)."""
        return {
            'compute_ms': np.mean(self._step_times) * 1000 if self._step_times else 0.0,
            'memory_mb': (sys.getsizeof(self.recurrent_weights) + 
                         sys.getsizeof(self.episodic_memory)) / 1024**2,
            'updates_per_step': np.mean(self._update_counts) if self._update_counts else 0.0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Hippocampus statistics."""
        return {
            'region': 'Hippocampus',
            'n_neurons': self.n_neurons,
            'sequence_length': len(self.sequence_buffer),
            'episodic_memories': len(self.episodic_memory),
            'recurrent_weights_norm': np.linalg.norm(self.recurrent_weights),
            'resource_usage': self.get_resource_usage()
        }
