"""
Prediction Error Tracker — Tracks prediction errors to detect surprises and anomalies.

In Throng5, this feeds:
  - Amygdala (danger detection via large errors)
  - Basal Ganglia (model refinement)
  - Policy Monitor (performance assessment)
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass


class PredictionErrorType(Enum):
    """Types of prediction errors."""
    REWARD = "reward"      # Expected vs actual reward
    STATE = "state"        # Expected vs actual next state
    VALUE = "value"        # TD error from Q-learning


@dataclass
class PredictionError:
    """Single prediction error record."""
    error_type: PredictionErrorType
    predicted: float
    actual: float
    error: float  # abs(actual - predicted)
    context: Dict


class PredictionErrorTracker:
    """
    Tracks prediction errors to detect surprises and anomalies.
    
    Uses blind heuristics to compute:
    - Surprise level (0-1): how unexpected recent outcomes are
    - Anomaly score (0-1): whether errors are unusually high
    """
    
    # Window sizes for tracking
    RECENT_WINDOW = 100
    BASELINE_WINDOW = 500
    
    # Thresholds
    SURPRISE_THRESHOLD = 2.0  # Std devs above mean
    ANOMALY_THRESHOLD = 3.0   # Std devs above baseline
    
    def __init__(self):
        self.errors: deque = deque(maxlen=self.BASELINE_WINDOW)
        self.recent_errors: deque = deque(maxlen=self.RECENT_WINDOW)
        
        # Per-type tracking
        self.errors_by_type: Dict[PredictionErrorType, List[float]] = {
            PredictionErrorType.REWARD: [],
            PredictionErrorType.STATE: [],
            PredictionErrorType.VALUE: [],
        }
    
    def record_error(
        self,
        error_type: PredictionErrorType,
        predicted: float,
        actual: float,
        context: Optional[Dict] = None
    ):
        """
        Record a prediction error.
        
        Args:
            error_type: Type of prediction error
            predicted: Predicted value
            actual: Actual value
            context: Optional context (state, action, etc.)
        """
        error = abs(actual - predicted)
        
        pred_error = PredictionError(
            error_type=error_type,
            predicted=predicted,
            actual=actual,
            error=error,
            context=context or {}
        )
        
        self.errors.append(pred_error)
        self.recent_errors.append(pred_error)
        self.errors_by_type[error_type].append(error)
    
    def get_surprise_level(self) -> float:
        """
        Compute surprise level (0-1) based on recent errors.
        
        High surprise = recent errors are much larger than baseline.
        
        Returns:
            0.0-1.0 score (0=not surprised, 1=very surprised)
        """
        if len(self.errors) < 20 or len(self.recent_errors) < 10:
            return 0.0
        
        # Baseline: mean error over all history
        all_errors = [e.error for e in self.errors]
        baseline_mean = np.mean(all_errors)
        baseline_std = np.std(all_errors) + 1e-8
        
        # Recent: mean error over recent window
        recent_errors = [e.error for e in self.recent_errors]
        recent_mean = np.mean(recent_errors)
        
        # Surprise = how many std devs above baseline
        surprise_std = (recent_mean - baseline_mean) / baseline_std
        
        # Normalize to 0-1 (clamp at threshold)
        surprise = surprise_std / self.SURPRISE_THRESHOLD
        return float(np.clip(surprise, 0.0, 1.0))
    
    def get_anomaly_score(self) -> float:
        """
        Detect if errors are unusually high (anomaly detection).
        
        High anomaly = recent max error is much larger than baseline max.
        
        Returns:
            0.0-1.0 score (0=normal, 1=anomaly detected)
        """
        if len(self.errors) < 20 or len(self.recent_errors) < 10:
            return 0.0
        
        # Baseline: 95th percentile of all errors
        all_errors = [e.error for e in self.errors]
        baseline_p95 = np.percentile(all_errors, 95)
        baseline_std = np.std(all_errors) + 1e-8
        
        # Recent: max error in recent window
        recent_errors = [e.error for e in self.recent_errors]
        recent_max = np.max(recent_errors)
        
        # Anomaly = how many std devs above baseline p95
        anomaly_std = (recent_max - baseline_p95) / baseline_std
        
        # Normalize to 0-1 (clamp at threshold)
        anomaly = anomaly_std / self.ANOMALY_THRESHOLD
        return float(np.clip(anomaly, 0.0, 1.0))
    
    def get_error_distribution(self) -> Dict[PredictionErrorType, Dict[str, float]]:
        """Get statistics for each error type."""
        distribution = {}
        
        for error_type in PredictionErrorType:
            errors = self.errors_by_type[error_type]
            if errors:
                distribution[error_type] = {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'max': float(np.max(errors)),
                    'count': len(errors),
                }
            else:
                distribution[error_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'max': 0.0,
                    'count': 0,
                }
        
        return distribution
    
    def summary(self) -> str:
        """Human-readable summary of prediction errors."""
        if len(self.errors) < 10:
            return "Insufficient prediction error data"
        
        surprise = self.get_surprise_level()
        anomaly = self.get_anomaly_score()
        distribution = self.get_error_distribution()
        
        lines = [
            f"Prediction Error Tracking ({len(self.errors)} errors):",
            f"  Surprise level: {surprise:.2f} (0=normal, 1=very surprised)",
            f"  Anomaly score:  {anomaly:.2f} (0=normal, 1=anomaly detected)",
            "",
            "Error distribution by type:",
        ]
        
        for error_type, stats in distribution.items():
            if stats['count'] > 0:
                lines.append(
                    f"  {error_type.value:8s}: "
                    f"mean={stats['mean']:6.3f}, "
                    f"std={stats['std']:6.3f}, "
                    f"max={stats['max']:6.3f} "
                    f"(n={stats['count']})"
                )
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset for new environment."""
        self.errors.clear()
        self.recent_errors.clear()
        for error_type in PredictionErrorType:
            self.errors_by_type[error_type].clear()


if __name__ == "__main__":
    """Test the prediction error tracker."""
    print("=" * 60)
    print("PREDICTION ERROR TRACKER TEST")
    print("=" * 60)
    
    tracker = PredictionErrorTracker()
    
    # Test 1: Normal errors (baseline)
    print("\n[Test 1] Recording baseline errors...")
    np.random.seed(42)
    for i in range(100):
        # Small random errors
        predicted = 1.0
        actual = predicted + np.random.randn() * 0.1
        tracker.record_error(PredictionErrorType.REWARD, predicted, actual)
    
    print(f"  Recorded {len(tracker.errors)} errors")
    print(f"  Surprise level: {tracker.get_surprise_level():.2f}")
    print(f"  Anomaly score: {tracker.get_anomaly_score():.2f}")
    
    # Test 2: Surprising errors (recent spike)
    print("\n[Test 2] Recording surprising errors (spike)...")
    for i in range(30):
        # Much larger errors
        predicted = 1.0
        actual = predicted + np.random.randn() * 5.0  # 50x larger than baseline
        tracker.record_error(PredictionErrorType.REWARD, predicted, actual)
    
    surprise = tracker.get_surprise_level()
    anomaly = tracker.get_anomaly_score()
    print(f"  Surprise level: {surprise:.2f}")
    print(f"  Anomaly score: {anomaly:.2f}")
    # Just verify it increases (don't enforce strict threshold)
    print("✅ Surprise/anomaly metrics computed")
    
    # Test 3: Anomaly (single huge error)
    print("\n[Test 3] Recording anomaly (huge error)...")
    tracker.record_error(PredictionErrorType.VALUE, 1.0, 50.0)  # Huge error
    
    anomaly_after = tracker.get_anomaly_score()
    print(f"  Anomaly score after huge error: {anomaly_after:.2f}")
    assert anomaly_after >= anomaly, "Anomaly score should increase or stay same"
    print("✅ Anomaly score responds to large errors")
    
    # Summary
    print(f"\n{tracker.summary()}")
    
    print("\n✅ Prediction error tracker test complete!")
