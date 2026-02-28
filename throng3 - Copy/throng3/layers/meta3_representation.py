"""
Meta^3: RepresentationOptimizer — Encoding Optimization

Optimizes HOW information is encoded in the neural substrate.
While Meta^0 processes data and Meta^1 adjusts weights,
Meta^3 asks: "Is this the best way to represent this information?"

Capabilities:
- Dimensionality analysis (too many/few dimensions?)
- Sparsity optimization (how sparse should activations be?)
- Decorrelation (reduce redundancy between neurons)
- Information-theoretic encoding (maximize mutual information)
- Compression ratio management (from throng2's compression work)
- Input/output encoding scheme selection
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class RepresentationConfig:
    """Configuration for representation optimization."""
    target_sparsity: float = 0.1      # Target fraction of active neurons
    target_decorrelation: float = 0.9  # Target decorrelation (1.0 = fully independent)
    compression_ratio: float = 10.0    # Target information compression
    dimensionality_check_interval: int = 50
    encoding_scheme: str = 'rate'      # 'rate', 'sparse', 'predictive', 'population'
    info_theoretic: bool = True        # Use mutual information metrics
    pca_dim: int = 32                  # PCA analysis dimensionality


class RepresentationOptimizer(MetaLayer):
    """
    Meta^3: Optimizes how information is encoded.
    
    Monitors the representational quality of Meta^0's activations
    and suggests changes to improve encoding efficiency.
    """
    
    def __init__(self, config: Optional[RepresentationConfig] = None, **kwargs):
        cfg = config or RepresentationConfig()
        super().__init__(level=3, name="RepresentationOptimizer", config=vars(cfg))
        self.repr_config = cfg
        
        # Activation statistics buffer
        self._activation_buffer: deque = deque(maxlen=200)
        self._output_buffer: deque = deque(maxlen=200)
        
        # Representational quality metrics
        self._sparsity_history: deque = deque(maxlen=500)
        self._correlation_history: deque = deque(maxlen=500)
        self._effective_dim_history: deque = deque(maxlen=500)
        self._mutual_info_history: deque = deque(maxlen=500)
        
        # Current encoding analysis
        self._current_sparsity = 0.0
        self._current_correlation = 0.0
        self._effective_dimensionality = 0
        self._current_compression = 1.0
        
        # Encoding scheme parameters
        self._encoding_params: Dict[str, float] = {
            'sparsity_penalty': 0.01,
            'decorrelation_strength': 0.1,
            'bottleneck_dim': cfg.pca_dim,
        }
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and optimize the representation.
        
        1. Collect activation patterns from Meta^0
        2. Compute representational quality metrics
        3. Suggest encoding changes if needed
        """
        self.process_inbox()
        
        # Get activations from context
        activations = context.get('activations', np.array([]))
        outputs = context.get('outputs', np.array([]))
        
        if len(activations) > 0:
            self._activation_buffer.append(activations.copy())
        if len(outputs) > 0:
            self._output_buffer.append(outputs.copy())
        
        # Analyze representation quality
        quality_metrics = self._analyze_representation()
        
        # Generate suggestions based on analysis
        suggestions = self._generate_encoding_suggestions(quality_metrics)
        
        # Send suggestions DOWN
        if suggestions:
            for target_level, suggestion in suggestions:
                self.signal(
                    direction=SignalDirection.DOWN,
                    signal_type=SignalType.SUGGESTION,
                    payload=suggestion,
                    target_level=target_level,
                    requires_response=True,
                )
        
        # Compute own metrics
        rep_quality = quality_metrics.get('overall_quality', 0.5)
        self.metrics.update(1.0 - rep_quality, rep_quality)
        
        # Signal UP
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'sparsity': self._current_sparsity,
                'correlation': self._current_correlation,
                'effective_dim': self._effective_dimensionality,
                'compression': self._current_compression,
                'encoding_scheme': self.repr_config.encoding_scheme,
                'quality': rep_quality,
            },
        )
        
        return {
            'quality_metrics': quality_metrics,
            'encoding_scheme': self.repr_config.encoding_scheme,
            'suggestions_sent': len(suggestions),
            'metrics': self.metrics,
        }
    
    def _analyze_representation(self) -> Dict[str, float]:
        """
        Analyze the quality of current representations.
        
        Returns dict of quality metrics.
        """
        if len(self._activation_buffer) < 10:
            return {'overall_quality': 0.5, 'data_insufficient': True}
        
        # Stack recent activations into matrix
        act_matrix = np.array(list(self._activation_buffer)[-50:])
        
        # 1. Sparsity analysis
        sparsity = self._compute_sparsity(act_matrix)
        self._current_sparsity = sparsity
        self._sparsity_history.append(sparsity)
        
        # 2. Correlation analysis (redundancy)
        correlation = self._compute_correlation(act_matrix)
        self._current_correlation = correlation
        self._correlation_history.append(correlation)
        
        # 3. Effective dimensionality
        eff_dim = self._compute_effective_dimensionality(act_matrix)
        self._effective_dimensionality = eff_dim
        self._effective_dim_history.append(eff_dim)
        
        # 4. Compression ratio
        compression = self._compute_compression_ratio(act_matrix)
        self._current_compression = compression
        
        # 5. Overall quality score
        sparsity_score = 1.0 - abs(sparsity - self.repr_config.target_sparsity)
        decorr_score = 1.0 - abs(correlation)  # Lower correlation = better
        dim_score = min(eff_dim / max(act_matrix.shape[1], 1), 1.0)
        
        overall = 0.4 * sparsity_score + 0.3 * decorr_score + 0.3 * dim_score
        
        return {
            'sparsity': sparsity,
            'sparsity_score': sparsity_score,
            'mean_correlation': correlation,
            'decorrelation_score': decorr_score,
            'effective_dimensionality': eff_dim,
            'dimensionality_score': dim_score,
            'compression_ratio': compression,
            'overall_quality': overall,
        }
    
    def _compute_sparsity(self, act_matrix: np.ndarray) -> float:
        """Compute average activation sparsity."""
        return float(np.mean(np.abs(act_matrix) < 0.01))
    
    def _compute_correlation(self, act_matrix: np.ndarray) -> float:
        """Compute mean pairwise correlation between neurons."""
        if act_matrix.shape[1] < 2:
            return 0.0
        
        # Subsample for efficiency
        n_neurons = act_matrix.shape[1]
        if n_neurons > 100:
            indices = np.random.choice(n_neurons, 100, replace=False)
            act_matrix = act_matrix[:, indices]
        
        # Compute correlation matrix
        std = np.std(act_matrix, axis=0)
        valid = std > 1e-8
        if np.sum(valid) < 2:
            return 0.0
        
        corr_matrix = np.corrcoef(act_matrix[:, valid].T)
        
        # Mean off-diagonal absolute correlation
        n = corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        mean_corr = float(np.mean(np.abs(corr_matrix[mask])))
        
        return mean_corr
    
    def _compute_effective_dimensionality(self, act_matrix: np.ndarray) -> int:
        """
        Compute effective dimensionality using explained variance.
        
        Uses the "participation ratio" from PCA eigenvalues.
        """
        if act_matrix.shape[0] < 5 or act_matrix.shape[1] < 2:
            return act_matrix.shape[1]
        
        # Center
        centered = act_matrix - np.mean(act_matrix, axis=0)
        
        # SVD (more numerically stable than eigendecomposition)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            eigenvalues = s ** 2 / (act_matrix.shape[0] - 1)
            
            # Participation ratio
            if np.sum(eigenvalues) < 1e-8:
                return 1
            
            p = eigenvalues / np.sum(eigenvalues)
            participation_ratio = 1.0 / np.sum(p ** 2)
            
            return max(1, int(participation_ratio))
        except np.linalg.LinAlgError:
            return act_matrix.shape[1]
    
    def _compute_compression_ratio(self, act_matrix: np.ndarray) -> float:
        """Estimate achievable compression ratio."""
        total_dim = act_matrix.shape[1]
        effective_dim = self._effective_dimensionality
        
        if effective_dim == 0:
            return 1.0
        
        return total_dim / effective_dim
    
    def _generate_encoding_suggestions(self, metrics: Dict) -> List[Tuple[int, Dict]]:
        """Generate suggestions to improve encoding."""
        suggestions = []
        
        if metrics.get('data_insufficient'):
            return suggestions
        
        # Sparsity too low → suggest stronger sparsity penalty
        sparsity = metrics.get('sparsity', 0.5)
        target = self.repr_config.target_sparsity
        
        if sparsity < target * 0.5:
            suggestions.append((0, {
                'threshold': self.config.get('threshold', 1.0) * 1.1,
            }))
        elif sparsity > target * 1.5:
            suggestions.append((0, {
                'threshold': self.config.get('threshold', 1.0) * 0.9,
            }))
        
        # High correlation → suggest decorrelation in weights
        correlation = metrics.get('mean_correlation', 0.0)
        if correlation > 0.5:
            # Suggest anti-Hebbian learning to decorrelate
            suggestions.append((1, {
                'hebbian_params': {
                    'decay': min(0.01, self._encoding_params['decorrelation_strength']),
                },
            }))
        
        # Low effective dimensionality → suggest expanding representation
        eff_dim = metrics.get('effective_dimensionality', 100)
        total_dim = len(self._activation_buffer[-1]) if self._activation_buffer else 100
        
        if eff_dim < total_dim * 0.1:
            # Many neurons are redundant — signal UP for architecture changes
            self.signal(
                direction=SignalDirection.UP,
                signal_type=SignalType.RESTRUCTURE,
                payload={
                    'reason': f'Low effective dimensionality ({eff_dim}/{total_dim})',
                    'suggestion': 'reduce_neurons',
                    'target_dim': eff_dim * 2,
                },
            )
        
        return suggestions
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state for Meta^3."""
        return np.array([
            self._current_sparsity,
            self._current_correlation,
            float(self._effective_dimensionality),
            self._current_compression,
            float(self.repr_config.encoding_scheme == 'rate'),
            float(self.repr_config.encoding_scheme == 'sparse'),
            float(self.repr_config.encoding_scheme == 'predictive'),
            float(self.repr_config.encoding_scheme == 'population'),
            self.metrics.loss,
            self.metrics.accuracy,
            self.metrics.stability,
            self._encoding_params.get('sparsity_penalty', 0.01),
            self._encoding_params.get('decorrelation_strength', 0.1),
            self._encoding_params.get('bottleneck_dim', 32),
        ])
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply encoding suggestions from Meta^4/Meta^5."""
        applied = False
        
        if 'encoding_scheme' in suggestion:
            scheme = suggestion['encoding_scheme']
            if scheme in ('rate', 'sparse', 'predictive', 'population'):
                self.repr_config.encoding_scheme = scheme
                applied = True
        
        if 'target_sparsity' in suggestion:
            self.repr_config.target_sparsity = np.clip(
                suggestion['target_sparsity'], 0.01, 0.99
            )
            applied = True
        
        if 'compression_ratio' in suggestion:
            self.repr_config.compression_ratio = max(1.0, suggestion['compression_ratio'])
            applied = True
        
        if 'encoding_params' in suggestion:
            self._encoding_params.update(suggestion['encoding_params'])
            applied = True
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate encoding suggestions."""
        score = 0.5
        reasons = []
        
        if 'encoding_scheme' in suggestion:
            # Accept if current scheme isn't performing well
            if self.metrics.accuracy < 0.4:
                score = 0.8
                reasons.append("Poor encoding quality, willing to change scheme")
            else:
                score = 0.3
                reasons.append("Current scheme performing adequately")
        
        if 'target_sparsity' in suggestion:
            target = suggestion['target_sparsity']
            if 0.01 <= target <= 0.99:
                score = max(score, 0.6)
                reasons.append(f"Valid sparsity target: {target}")
        
        return score, "; ".join(reasons) if reasons else "No criteria"
    
    def _self_optimize_weights(self):
        """Adjust encoding parameters based on recent metrics."""
        if not self._sparsity_history:
            return
        
        # Auto-tune sparsity penalty
        recent_sparsity = np.mean(list(self._sparsity_history)[-20:])
        target = self.repr_config.target_sparsity
        
        if recent_sparsity < target:
            self._encoding_params['sparsity_penalty'] *= 1.01
        else:
            self._encoding_params['sparsity_penalty'] *= 0.99
    
    def _self_optimize_synapses(self):
        """Adjust decorrelation strength."""
        if not self._correlation_history:
            return
        
        recent_corr = np.mean(list(self._correlation_history)[-20:])
        if recent_corr > self.repr_config.target_decorrelation:
            self._encoding_params['decorrelation_strength'] *= 1.02
        else:
            self._encoding_params['decorrelation_strength'] *= 0.98
    
    def _self_optimize_neurons(self):
        """Monitor effective dimensionality trends."""
        pass
