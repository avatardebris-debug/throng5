"""
Feature Expansion Utilities

Provides feature expansion methods for richer state representations:
- RBF (Radial Basis Functions) for continuous states
- Polynomial features for smooth functions
- Tile coding for fast discrete approximation
"""

import numpy as np
from typing import List, Tuple, Optional


class RBFFeatures:
    """
    Radial Basis Function feature expansion.
    
    Creates localized features using Gaussian RBFs centered across state space.
    Each RBF activates strongly near its center, weakly far away.
    
    Improved version with:
    - Better sigma auto-computation (wider coverage)
    - Optional normalization (features sum to 1)
    - Bias term for baseline Q-value
    """
    
    def __init__(self, 
                 n_centers_per_dim: int = 10,
                 state_bounds: List[Tuple[float, float]] = None,
                 sigma: float = None,
                 normalize: bool = True,
                 add_bias: bool = True):
        """
        Initialize RBF feature expander.
        
        Args:
            n_centers_per_dim: Number of RBF centers per dimension (total = n^d)
            state_bounds: [(min, max), ...] for each state dimension
            sigma: RBF width (default: auto-computed for good overlap)
            normalize: If True, normalize features to sum to 1
            add_bias: If True, add constant bias feature
        """
        if state_bounds is None:
            raise ValueError("state_bounds required")
        
        self.state_bounds = np.array(state_bounds)
        self.n_dims = len(state_bounds)
        self.n_centers_per_dim = n_centers_per_dim
        self.normalize = normalize
        self.add_bias = add_bias
        
        # Create RBF centers uniformly across state space
        self.centers = self._create_centers()
        self.n_centers = len(self.centers)
        
        # Set RBF width (sigma) - key for good coverage
        if sigma is None:
            # Make RBFs overlap well: sigma = 1.5 * spacing
            state_range = self.state_bounds[:, 1] - self.state_bounds[:, 0]
            avg_spacing = np.mean(state_range / (n_centers_per_dim - 1))
            self.sigma = avg_spacing * 1.5
        else:
            self.sigma = sigma
    
    def _create_centers(self) -> np.ndarray:
        """Create RBF centers as a grid across state space."""
        grids = []
        for dim_bounds in self.state_bounds:
            grids.append(np.linspace(dim_bounds[0], dim_bounds[1], self.n_centers_per_dim))
        
        # Create meshgrid
        if self.n_dims == 1:
            return grids[0].reshape(-1, 1)
        elif self.n_dims == 2:
            xx, yy = np.meshgrid(grids[0], grids[1])
            return np.column_stack([xx.ravel(), yy.ravel()])
        else:
            mesh = np.meshgrid(*grids, indexing='ij')
            return np.column_stack([m.ravel() for m in mesh])
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """
        Transform state to RBF feature vector.
        
        Args:
            state: Raw state vector (n_dims,)
        
        Returns:
            RBF features (n_centers + 1 if bias)
        """
        state = np.array(state)
        
        # Compute squared distance to each center
        diff = self.centers - state
        sq_distances = np.sum(diff**2, axis=1)
        
        # Apply Gaussian RBF: exp(-d^2 / (2*sigma^2))
        features = np.exp(-sq_distances / (2 * self.sigma**2))
        
        # Normalize so features sum to 1 (like soft attention)
        if self.normalize and features.sum() > 1e-8:
            features = features / features.sum()
        
        # Add bias term
        if self.add_bias:
            features = np.concatenate([[1.0], features])
        
        return features
    
    def get_n_features(self) -> int:
        """Get number of output features."""
        return self.n_centers + (1 if self.add_bias else 0)


class PolynomialFeatures:
    """
    Polynomial feature expansion.
    
    Expands [x, y] → [1, x, y, x^2, xy, y^2, ...]
    """
    
    def __init__(self, degree: int = 2):
        """
        Initialize polynomial features.
        
        Args:
            degree: Maximum polynomial degree
        """
        self.degree = degree
    
    def transform(self, state: np.ndarray) -> np.ndarray:
        """Transform state to polynomial features."""
        state = np.array(state)
        features = [1.0]  # Bias term
        
        # Degree 1: original features
        features.extend(state)
        
        # Degree 2+: polynomial combinations
        if self.degree >= 2:
            n_dims = len(state)
            for i in range(n_dims):
                for j in range(i, n_dims):
                    features.append(state[i] * state[j])
        
        # Higher degrees (if needed)
        if self.degree >= 3:
            for i in range(len(state)):
                features.append(state[i] ** 3)
        
        return np.array(features)
    
    def get_n_features(self, n_input_dims: int) -> int:
        """Get number of output features."""
        if self.degree == 1:
            return 1 + n_input_dims
        elif self.degree == 2:
            return 1 + n_input_dims + (n_input_dims * (n_input_dims + 1)) // 2
        else:
            # Approximate for higher degrees
            return 1 + n_input_dims * (self.degree + 1)
