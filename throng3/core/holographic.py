"""
Holographic State — Any Slice Contains Information About the Whole

The holographic principle in Meta^N: every layer's state snapshot
contains a compressed projection of the entire system state.

This enables:
1. Recovery from layer failure using any surviving layer
2. Cross-scale reasoning (any layer can reason about the whole)
3. Redundancy without full duplication (efficient)
4. Distributed consensus about system state

Implementation uses random projections (Johnson-Lindenstrauss lemma)
to create fixed-size representations that preserve essential structure.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import hashlib
import json


@dataclass
class HolographicState:
    """
    Global holographic state assembled from all layer snapshots.
    
    Each layer contributes its state vector, which gets projected
    into a shared holographic space. The combined state can be
    queried from any perspective (layer).
    """
    
    dim: int = 128                    # Holographic space dimensionality
    n_layers: int = 6                 # Expected number of meta-layers
    decay: float = 0.95              # How fast old state fades
    
    # Internal state
    _projections: Dict[int, np.ndarray] = field(default_factory=dict)
    _timestamps: Dict[int, float] = field(default_factory=dict)
    _combined: Optional[np.ndarray] = field(default=None)
    _version: int = 0
    
    def __post_init__(self):
        """Initialize projection matrices and combined state."""
        self._combined = np.zeros(self.dim)
        self._projection_matrices = {}
        
        # Create deterministic projection matrices for each layer
        for level in range(self.n_layers + 2):  # Extra room for dynamic layers
            rng = np.random.RandomState(level * 137 + 42)
            self._projection_matrices[level] = rng.randn(self.dim, self.dim)
            self._projection_matrices[level] /= np.sqrt(self.dim)
    
    def update_layer(self, level: int, state_vector: np.ndarray):
        """
        Update the holographic state with a layer's state vector.
        
        Args:
            level: Meta-level of the contributing layer
            state_vector: Layer's compressed state (from snapshot)
        """
        # Ensure we have a projection matrix for this level
        if level not in self._projection_matrices:
            rng = np.random.RandomState(level * 137 + 42)
            self._projection_matrices[level] = rng.randn(self.dim, self.dim)
            self._projection_matrices[level] /= np.sqrt(self.dim)
        
        # Resize state vector to match dim
        if len(state_vector) < self.dim:
            padded = np.zeros(self.dim)
            padded[:len(state_vector)] = state_vector
            state_vector = padded
        elif len(state_vector) > self.dim:
            state_vector = state_vector[:self.dim]
        
        # Project into holographic space
        projected = self._projection_matrices[level] @ state_vector
        
        # Decay old projection for this layer
        if level in self._projections:
            old = self._projections[level]
            self._combined -= old * self._get_weight(level)
        
        # Store new projection
        self._projections[level] = projected
        self._timestamps[level] = time.time()
        
        # Add to combined state
        self._combined += projected * self._get_weight(level)
        self._version += 1
    
    def _get_weight(self, level: int) -> float:
        """
        Get the weight for a layer's contribution based on recency.
        
        More recent updates have higher weight.
        """
        if level not in self._timestamps:
            return 1.0
        
        age = time.time() - self._timestamps[level]
        return self.decay ** age
    
    def query(self, from_level: int) -> np.ndarray:
        """
        Query the holographic state from a specific layer's perspective.
        
        Each layer sees the combined state projected through its own
        inverse projection, giving a layer-specific view of the whole.
        
        Args:
            from_level: The meta-level querying the state
            
        Returns:
            State vector from the querying layer's perspective
        """
        if self._combined is None or np.allclose(self._combined, 0):
            return np.zeros(self.dim)
        
        # Inverse projection (transpose for orthogonal-ish matrices)
        if from_level in self._projection_matrices:
            proj_T = self._projection_matrices[from_level].T
            return proj_T @ self._combined
        
        return self._combined.copy()
    
    def query_layer(self, target_level: int, from_level: int) -> np.ndarray:
        """
        Reconstruct a specific layer's state from the holographic encoding.
        
        This is the key holographic property: any layer can approximately
        reconstruct any other layer's state.
        
        Args:
            target_level: The layer to reconstruct
            from_level: The layer doing the reconstruction
            
        Returns:
            Approximate reconstruction of target layer's state
        """
        if target_level in self._projections:
            # Direct reconstruction via inverse projection
            proj = self._projection_matrices[target_level]
            return np.linalg.lstsq(proj, self._projections[target_level], rcond=None)[0]
        
        # Fallback: use combined state
        return self.query(from_level)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire system from the holographic state.
        
        Returns:
            Dict with system-level metrics and state info
        """
        if self._combined is None:
            return {"status": "uninitialized", "n_layers": 0}
        
        return {
            "status": "active",
            "n_layers_reporting": len(self._projections),
            "version": self._version,
            "combined_norm": float(np.linalg.norm(self._combined)),
            "combined_mean": float(np.mean(self._combined)),
            "combined_std": float(np.std(self._combined)),
            "layer_norms": {
                level: float(np.linalg.norm(proj))
                for level, proj in self._projections.items()
            },
            "layer_ages": {
                level: time.time() - ts
                for level, ts in self._timestamps.items()
            },
            "coherence": self._compute_coherence(),
        }
    
    def _compute_coherence(self) -> float:
        """
        Compute coherence between layers in holographic space.
        
        High coherence = layers are in agreement
        Low coherence = layers have divergent states
        
        Returns:
            Coherence score 0-1
        """
        if len(self._projections) < 2:
            return 1.0
        
        projections = list(self._projections.values())
        norms = [np.linalg.norm(p) for p in projections]
        
        # Pairwise cosine similarities
        similarities = []
        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                if norms[i] > 1e-8 and norms[j] > 1e-8:
                    cos_sim = np.dot(projections[i], projections[j]) / (norms[i] * norms[j])
                    similarities.append(abs(cos_sim))
        
        if not similarities:
            return 1.0
        
        return float(np.mean(similarities))
    
    def get_hash(self) -> str:
        """Get a hash of the current holographic state for change detection."""
        if self._combined is None:
            return "0" * 16
        state_bytes = self._combined.tobytes()
        return hashlib.md5(state_bytes).hexdigest()[:16]
    
    def save(self) -> Dict[str, Any]:
        """Serialize holographic state for persistence."""
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "version": self._version,
            "projections": {
                str(k): v.tolist() for k, v in self._projections.items()
            },
            "timestamps": {str(k): v for k, v in self._timestamps.items()},
            "combined": self._combined.tolist() if self._combined is not None else None,
        }
    
    def load(self, data: Dict[str, Any]):
        """Restore holographic state from serialized data."""
        self.dim = data.get("dim", self.dim)
        self.n_layers = data.get("n_layers", self.n_layers)
        self._version = data.get("version", 0)
        
        if "projections" in data:
            self._projections = {
                int(k): np.array(v) for k, v in data["projections"].items()
            }
        if "timestamps" in data:
            self._timestamps = {int(k): v for k, v in data["timestamps"].items()}
        if data.get("combined") is not None:
            self._combined = np.array(data["combined"])
