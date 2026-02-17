"""
Test failure categorization integration.
"""

import sys
sys.path.insert(0, '.')

from throng4.meta_policy import MetaPolicyController, FailureMode
import numpy as np

print("=" * 60)
print("FAILURE CATEGORIZATION INTEGRATION TEST")
print("=" * 60)

# Create controller
controller = MetaPolicyController()
print("\n✅ Controller created")

# Verify PerceptionHub has failure_profiler
assert hasattr(controller.perception, 'failure_profiler'), "PerceptionHub missing failure_profiler"
print("✅ PerceptionHub has failure_profiler")

# Verify PolicyMonitor.check_retirement has dominant_failure_mode param
import inspect
sig = inspect.signature(controller.policy_monitor.check_retirement)
assert 'dominant_failure_mode' in sig.parameters, "PolicyMonitor.check_retirement missing dominant_failure_mode param"
print("✅ PolicyMonitor.check_retirement has dominant_failure_mode param")

# Test failure recording
print("\n[Test] Recording failures...")
for i in range(10):
    state = np.random.randn(10)
    next_state = state + np.random.randn(10) * 0.01  # Small change (mechanical)
    controller.perception.record(state, 0, -1.0, next_state)

print(f"  Recorded {len(controller.perception.failure_analyses)} failures")
assert len(controller.perception.failure_analyses) == 10, "Should have 10 failures"
print("✅ Failure recording works")

# Test failure categorization
print("\n[Test] Failure categorization...")
print(controller.perception.get_failure_summary())
dominant = controller.perception.get_dominant_failure_mode()
print(f"  Dominant mode: {dominant.value if dominant else 'None'}")
assert dominant is not None, "Should have a dominant failure mode"
print("✅ Failure categorization works")

# Test retirement logic with failure modes
print("\n[Test] Retirement logic with failure modes...")
from throng4.meta_policy.policy_tree import PolicyNode
from throng4.meta_policy.env_fingerprint import EnvironmentFingerprint

# Create mock policy
fp = EnvironmentFingerprint(
    action_count=4,
    state_dim=10,
    reward_density=0.1,
    reward_min=0.0,
    reward_max=1.0,
    state_change_rate=0.5,
    action_diversity_score=0.5,
)
policy = controller.policy_tree.create_root(fp)

# Test strategic failure → retire on declining
should_retire = controller.policy_monitor.check_retirement(
    policy, 'declining', 'strategic'
)
assert should_retire, "Should retire on strategic + declining"
print("✅ Strategic failures → retire on declining")

# Test temporal failure → don't retire on declining
should_retire = controller.policy_monitor.check_retirement(
    policy, 'declining', 'temporal'
)
assert not should_retire, "Should NOT retire on temporal + declining"
print("✅ Temporal failures → patient on declining")

# Test critical always retires
should_retire = controller.policy_monitor.check_retirement(
    policy, 'critical', 'temporal'
)
assert should_retire, "Should retire on critical regardless of failure mode"
print("✅ Critical risk always retires")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
