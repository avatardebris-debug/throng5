"""
PerceptionHub — Owns visual and causal pattern extraction.

Throng5 role: Feeds data to Amygdala (danger detection) and Risk Evaluator.
Wraps VisualPatternExtractor + CausalDiscovery + FailureProfiler + PredictionErrorTracker
into a single perception interface.
"""

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from throng4.meta_policy.visual_patterns import VisualPatternExtractor, VisualPatterns
from throng4.meta_policy.causal_discovery import CausalDiscovery, ActionEffect
from throng4.meta_policy.failure_profiler import FailureProfiler, FailureMode, FailureAnalysis
from throng4.meta_policy.prediction_error_tracker import (
    PredictionErrorTracker, PredictionErrorType, PredictionError
)


class PerceptionHub:
    """
    Central perception module — extracts patterns from raw state/transition data.
    
    In Throng5, this feeds:
      - Amygdala (danger pattern detection)
      - Risk Evaluator (position assessment)
      - Prefrontal Cortex (LLM prompt building)
    """
    
    MIN_STATES_FOR_VISUAL = 50
    MIN_TRANSITIONS_FOR_CAUSAL = 100
    FAILURE_REWARD_THRESHOLD = 0.0  # Rewards <= this are considered failures
    
    def __init__(self):
        self.visual_extractor = VisualPatternExtractor()
        self.causal_discovery = CausalDiscovery()
        self.failure_profiler = FailureProfiler()
        self.prediction_error_tracker = PredictionErrorTracker()
        
        # Raw data buffers
        self.recent_states: deque = deque(maxlen=1000)
        self.recent_transitions: List[Dict] = []
        
        # Extracted patterns (updated periodically)
        self.visual_patterns: Optional[VisualPatterns] = None
        self.causal_effects: Optional[Dict[int, ActionEffect]] = None
        
        # Failure tracking
        self.failure_analyses: List[FailureAnalysis] = []
        
        # Prediction error tracking
        self.prediction_errors: List[PredictionError] = []
    
    def record(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray):
        """Record a single step for pattern extraction and failure categorization."""
        self.recent_states.append(state)
        self.causal_discovery.record_transition(state, action, reward, next_state)
        self.recent_transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
        })
        
        # Categorize if this looks like a failure
        if self._is_failure(reward):
            # Get latest prediction error for this step (if available)
            pe_magnitude = None
            if self.prediction_error_tracker.recent_errors:
                latest_pe = self.prediction_error_tracker.recent_errors[-1]
                pe_magnitude = latest_pe.error
            
            analysis = self.failure_profiler.categorize_failure(
                state, action, reward, next_state,
                self.recent_transitions[-10:],  # Last 10 for context
                prediction_error=pe_magnitude
            )
            self.failure_analyses.append(analysis)
    
    def _is_failure(self, reward: float) -> bool:
        """Heuristic: is this a failure worth categorizing?"""
        return reward <= self.FAILURE_REWARD_THRESHOLD
    
    def update_patterns(self):
        """Extract/refresh visual and causal patterns from buffered data."""
        if len(self.recent_states) > self.MIN_STATES_FOR_VISUAL:
            self.visual_patterns = self.visual_extractor.extract_patterns(
                list(self.recent_states)
            )
        
        if len(self.recent_transitions) > self.MIN_TRANSITIONS_FOR_CAUSAL:
            self.causal_effects = self.causal_discovery.discover_action_effects(
                self.recent_transitions[-500:]
            )
    
    def get_causal_summary(self) -> str:
        """Get human-readable summary of causal effects."""
        if self.causal_effects:
            return self.causal_discovery.get_summary(self.causal_effects)
        return ""
    
    def get_failure_summary(self) -> str:
        """Get human-readable summary of failure categorization."""
        return self.failure_profiler.summary()
    
    def get_dominant_failure_mode(self) -> Optional[FailureMode]:
        """Get the most common failure mode."""
        return self.failure_profiler.get_dominant_failure_mode()
    
    def record_prediction_error(
        self,
        error_type: PredictionErrorType,
        predicted: float,
        actual: float,
        context: Optional[Dict] = None
    ):
        """Record a prediction error for surprise/anomaly detection."""
        self.prediction_error_tracker.record_error(
            error_type, predicted, actual, context
        )
    
    def get_prediction_error_summary(self) -> str:
        """Get human-readable summary of prediction errors."""
        return self.prediction_error_tracker.summary()
    
    def get_surprise_level(self) -> float:
        """Get surprise level (0-1) based on recent prediction errors."""
        return self.prediction_error_tracker.get_surprise_level()
    
    def get_anomaly_score(self) -> float:
        """Get anomaly score (0-1) for unusual prediction errors."""
        return self.prediction_error_tracker.get_anomaly_score()
    
    def reset(self):
        """Reset for a new environment."""
        self.recent_states.clear()
        self.recent_transitions.clear()
        self.visual_patterns = None
        self.causal_effects = None
        self.failure_profiler.reset()
        self.failure_analyses.clear()
        self.prediction_error_tracker.reset()
        self.prediction_errors.clear()

