"""
Test prediction error tracking integration.
"""

import sys
sys.path.insert(0, '.')

from throng4.meta_policy import (
    MetaPolicyController, PerceptionHub, PredictionErrorType
)
import numpy as np

print("=" * 60)
print("PREDICTION ERROR TRACKING INTEGRATION TEST")
print("=" * 60)

# Create controller
controller = MetaPolicyController()
print("\n✅ Controller created")

# Verify PerceptionHub has prediction_error_tracker
assert hasattr(controller.perception, 'prediction_error_tracker'), \
    "PerceptionHub missing prediction_error_tracker"
print("✅ PerceptionHub has prediction_error_tracker")

# Test prediction error recording
print("\n[Test] Recording prediction errors...")
for i in range(50):
    # Baseline errors
    controller.perception.record_prediction_error(
        PredictionErrorType.REWARD,
        predicted=1.0,
        actual=1.0 + np.random.randn() * 0.1
    )

surprise_baseline = controller.perception.get_surprise_level()
print(f"  Baseline surprise: {surprise_baseline:.2f}")

# Add surprising errors
for i in range(20):
    controller.perception.record_prediction_error(
        PredictionErrorType.REWARD,
        predicted=1.0,
        actual=1.0 + np.random.randn() * 2.0  # Much larger
    )

surprise_after = controller.perception.get_surprise_level()
anomaly = controller.perception.get_anomaly_score()
print(f"  Surprise after spike: {surprise_after:.2f}")
print(f"  Anomaly score: {anomaly:.2f}")
print("✅ Prediction error recording works")

# Test summary
print("\n[Test] Prediction error summary...")
summary = controller.perception.get_prediction_error_summary()
print(summary)
assert "Prediction Error Tracking" in summary, "Summary should include title"
print("✅ Summary generation works")

# Test PrefrontalCortex prompt integration
print("\n[Test] PrefrontalCortex prompt integration...")
from collections import deque
from throng4.meta_policy.env_fingerprint import EnvironmentFingerprint

# Create mock fingerprint
fp = EnvironmentFingerprint(
    action_count=4,
    state_dim=10,
    reward_density=0.1,
    reward_min=0.0,
    reward_max=1.0,
    state_change_rate=0.5,
    action_diversity_score=0.5,
)

rewards = deque([1.0] * 50, maxlen=200)

# Build prompt (should include surprise if high enough)
prompt = controller.prefrontal.build_prompt(
    fp, controller.perception, rewards, 50, 10
)

if surprise_after > 0.3 or anomaly > 0.5:
    assert "surprise" in prompt.lower() or "anomaly" in prompt.lower(), \
        "Prompt should mention surprise/anomaly when high"
    print("✅ High surprise/anomaly included in prompt")
else:
    print("✅ Prompt generated (surprise/anomaly below threshold)")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
