"""Utility functions for throng3."""

import numpy as np
from typing import Dict, Any
import json
import time


def compute_entropy(activations: np.ndarray) -> float:
    """Compute Shannon entropy of activation distribution."""
    if len(activations) == 0:
        return 0.0
    
    # Discretize activations into bins
    hist, _ = np.histogram(activations, bins=50, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return -float(np.sum(hist * np.log2(hist + 1e-10)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate mutual information between two variables."""
    if len(x) < 10 or len(y) < 10:
        return 0.0
    
    # Joint histogram
    H_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    H_xy = H_xy / (H_xy.sum() + 1e-10)
    
    # Marginals
    H_x = H_xy.sum(axis=1)
    H_y = H_xy.sum(axis=0)
    
    # MI = H(X) + H(Y) - H(X,Y)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if H_xy[i, j] > 1e-10 and H_x[i] > 1e-10 and H_y[j] > 1e-10:
                mi += H_xy[i, j] * np.log2(H_xy[i, j] / (H_x[i] * H_y[j]))
    
    return max(0, float(mi))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Compute participation ratio (effective dimensionality) from eigenvalues."""
    if len(eigenvalues) == 0 or np.sum(eigenvalues) < 1e-10:
        return 0.0
    p = eigenvalues / np.sum(eigenvalues)
    return 1.0 / np.sum(p ** 2)


class Timer:
    """Simple context manager timer."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self._start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self._start
    
    def __repr__(self):
        return f"Timer({self.name}: {self.elapsed*1000:.1f}ms)"


class MetricsLogger:
    """Collects and serializes metrics over time."""
    
    def __init__(self, max_history: int = 10000):
        self.history: Dict[str, list] = {}
        self.max_history = max_history
    
    def log(self, **kwargs):
        """Log one or more metrics."""
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            if len(self.history[key]) > self.max_history:
                self.history[key] = self.history[key][-self.max_history // 2:]
    
    def get(self, key: str, n: int = -1) -> list:
        """Get metric history."""
        vals = self.history.get(key, [])
        return vals[-n:] if n > 0 else vals
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        result = {}
        for key, values in self.history.items():
            if values and isinstance(values[0], (int, float)):
                arr = np.array(values)
                result[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'last': float(arr[-1]),
                    'n': len(arr),
                }
        return result
    
    def save_json(self, path: str):
        """Save metrics to JSON."""
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
