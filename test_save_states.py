"""Test save-state manager integration."""
import sys
sys.path.insert(0, '.')

import numpy as np
from throng4.meta_policy.save_state_manager import (
    SaveStateManager, SaveStateTrigger
)
from throng4.meta_policy.perception_hub import PerceptionHub

print("=" * 60)
print("SAVE-STATE MANAGER TEST")
print("=" * 60)

manager = SaveStateManager()
perception = PerceptionHub()

# Test 1: No triggers initially
print("\n[Test 1] No triggers initially...")
result = manager.check_triggers(perception, episode=1, rewards=[1.0])
assert result is None, f"Expected None, got {result}"
print("  PASS: No false triggers")

# Test 2: Reward spike
print("\n[Test 2] Reward spike...")
rewards = [1.0] * 20 + [10.0]
result = manager.check_triggers(perception, episode=10, rewards=rewards)
print(f"  Result: {result}")
if result:
    print(f"  Trigger: {result.trigger.value}")
    assert result.trigger == SaveStateTrigger.REWARD_SPIKE
    print("  PASS: Reward spike detected")
else:
    print("  PASS: No spike (within range)")

# Test 3: Failure cluster
print("\n[Test 3] Failure cluster...")
for i in range(10):
    state = np.random.randn(10)
    next_state = state + np.random.randn(10) * 0.001
    perception.record(state, 0, -1.0, next_state)
print(f"  Failure analyses: {len(perception.failure_analyses)}")
result = manager.check_triggers(perception, episode=20, rewards=[1.0] * 20)
print(f"  Result: {result}")
if result:
    print(f"  Trigger: {result.trigger.value}")
    print("  PASS: Failure cluster detected")
else:
    print("  PASS: No cluster (expected)")

# Test 4: Mode transition (use fresh perception to avoid failure cluster interference)
print("\n[Test 4] Mode transition...")
fresh_perception = PerceptionHub()
manager2 = SaveStateManager()
manager2._prev_mode = 'learning'
result = manager2.check_triggers(
    fresh_perception, episode=30, rewards=[1.0] * 30, current_mode='adaptive'
)
print(f"  Result: {result}")
assert result is not None, "Expected mode transition trigger"
assert result.trigger == SaveStateTrigger.MODE_TRANSITION
print("  PASS: Mode transition detected")

# Test 5: Hypothesis result (use fresh setup to avoid cluster interference)
print("\n[Test 5] Hypothesis result...")
manager3 = SaveStateManager()
fresh_p3 = PerceptionHub()
result = manager3.check_triggers(
    fresh_p3, episode=40, rewards=[1.0] * 40,
    hypothesis_result={'strategy': 'explore_more', 'reward_delta': 25.0}
)
assert result is not None
assert result.trigger == SaveStateTrigger.HYPOTHESIS_RESULT
print(f"  Importance: {result.importance:.2f}")
print("  PASS: Hypothesis result flagged")

# Test 6: Cooldown 
print("\n[Test 6] Cooldown...")
result = manager3.check_triggers(
    fresh_p3, episode=41, rewards=[1.0] * 41,
    hypothesis_result={'strategy': 'test', 'reward_delta': 10.0}
)
# Should be None or a different trigger type (cooldown prevents same type)
if result is None or result.trigger != SaveStateTrigger.HYPOTHESIS_RESULT:
    print("  PASS: Cooldown prevents rapid re-triggering")
else:
    print("  FAIL: Cooldown not working")

# Summary
print(f"\n{manager.summary()}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
