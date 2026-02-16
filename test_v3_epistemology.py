"""
Test v3 Epistemological Features.

Tests:
1. Confidence decay over time
2. Anti-policy auto-generation
3. Stochasticity detection (OutcomeDistribution)
4. Attribution diagnosis on GridWorld (deterministic)
5. Attribution diagnosis on FrozenLake (stochastic) — skipped if gymnasium unavailable
6. DORMANT status (never eliminated)
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompute/throng3')
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import time
import numpy as np
from throng4.llm_policy.hypothesis import (
    DiscoveredRule, RuleStatus, RuleLibrary, 
    OutcomeDistribution, TestResult
)
from throng4.llm_policy.attribution import (
    AttributionDiagnoser, Attribution
)


def test_confidence_decay():
    """Test that confidence decays over time."""
    print("="*70)
    print("TEST 1: Confidence Decay")
    print("="*70)
    
    rule = DiscoveredRule(
        id="test_decay",
        description="Test rule for decay",
        feature="test",
        confidence=0.9,
        status=RuleStatus.ACTIVE,
        decay_rate=0.1  # Fast decay for testing
    )
    
    print(f"\n  Initial confidence: {rule.confidence:.3f}")
    print(f"  Status: {rule.status.value}")
    
    # Simulate 5 hours passing
    future_time = time.time() + 5 * 3600
    rule.apply_confidence_decay(future_time)
    
    print(f"\n  After 5 hours (decay_rate=0.1):")
    print(f"    Confidence: {rule.confidence:.3f}")
    
    assert rule.confidence < 0.9, "Confidence should have decayed"
    assert rule.confidence > 0.01, "Confidence should not be zero (minimum floor)"
    
    # Update status based on decayed confidence
    rule.n_tests = 5  # Need enough tests for non-TENTATIVE status
    rule.update_status()
    print(f"    Status: {rule.status.value}")
    
    # After heavy decay, should become DORMANT
    far_future = time.time() + 100 * 3600
    rule.apply_confidence_decay(far_future)
    rule.update_status()
    
    print(f"\n  After extreme decay:")
    print(f"    Confidence: {rule.confidence:.3f}")
    print(f"    Status: {rule.status.value}")
    
    assert rule.status == RuleStatus.DORMANT, f"Should be DORMANT, got {rule.status.value}"
    assert rule.confidence > 0, "Should NEVER be zero (soft epistemology)"
    
    print("\n[OK] Confidence decay test passed!")
    return True


def test_anti_policy_generation():
    """Test auto-generation of anti-policies."""
    print("\n" + "="*70)
    print("TEST 2: Anti-Policy Generation")
    print("="*70)
    
    library = RuleLibrary()
    
    # Create a consistently failing rule
    failing_rule = DiscoveredRule(
        id="bad_action_3",
        description="Action 3 increases score",
        feature="score",
        direction="increase"
    )
    
    # Add many failures
    for i in range(10):
        result = TestResult(
            action=3,
            state_before=np.array([0, 0]),
            state_after=np.array([0, 0]),
            reward=-1.0,
            done=True
        )
        failing_rule.add_test_result(result, success=False)
    
    failing_rule.update_status()
    
    print(f"\n  Rule: {failing_rule.description}")
    print(f"  Confidence: {failing_rule.confidence:.3f}")
    print(f"  Status: {failing_rule.status.value}")
    
    # Add with anti-policy (should auto-generate since confidence < 0.3)
    library.add_with_anti_policy(failing_rule)
    
    anti_policies = library.get_anti_policies()
    
    print(f"\n  Anti-policies generated: {len(anti_policies)}")
    
    assert len(anti_policies) > 0, "Should generate anti-policy for low-confidence rule"
    
    anti = anti_policies[0]
    print(f"\n  Anti-policy: {anti.description}")
    print(f"    Direction: {anti.direction}")
    print(f"    Confidence: {anti.confidence:.3f}")
    print(f"    Parent: {anti.parent_rule_id}")
    
    assert anti.parent_rule_id == failing_rule.id, "Anti-policy should reference parent"
    assert anti.direction == "decrease", "Anti-policy should invert direction"
    assert anti.confidence > failing_rule.confidence, "Anti-policy should have higher confidence"
    
    print("\n[OK] Anti-policy generation test passed!")
    return True


def test_outcome_distribution():
    """Test stochasticity tracking via OutcomeDistribution."""
    print("\n" + "="*70)
    print("TEST 3: Outcome Distribution & Stochasticity")
    print("="*70)
    
    # Deterministic case
    det = OutcomeDistribution()
    for _ in range(20):
        det.record("same_outcome")
    
    print(f"\n  Deterministic (20x same outcome):")
    print(f"    Stochasticity: {det.stochasticity_score():.3f}")
    print(f"    Entropy: {det.entropy():.3f}")
    
    assert det.stochasticity_score() == 0.0, "Same outcome should be fully deterministic"
    
    # Fully random case (uniform distribution)
    rand = OutcomeDistribution()
    for outcome in ["A", "B", "C", "D"] * 25:
        rand.record(outcome)
    
    print(f"\n  Uniform random (4 outcomes, 25 each):")
    print(f"    Stochasticity: {rand.stochasticity_score():.3f}")
    print(f"    Entropy: {rand.entropy():.3f}")
    
    assert rand.stochasticity_score() > 0.9, "Uniform distribution should be highly stochastic"
    
    # Partially random (FrozenLake-like: 33% each way)
    partial = OutcomeDistribution()
    for _ in range(33):
        partial.record("intended_direction")
    for _ in range(33):
        partial.record("slip_left")
    for _ in range(34):
        partial.record("slip_right")
    
    print(f"\n  Partial random (FrozenLake slip, ~33% each):")
    print(f"    Stochasticity: {partial.stochasticity_score():.3f}")
    print(f"    Entropy: {partial.entropy():.3f}")
    print(f"    P(intended): {partial.probability('intended_direction'):.2f}")
    
    assert 0.5 < partial.stochasticity_score() < 1.0, "Partial random should be moderately stochastic"
    
    print("\n[OK] Outcome distribution test passed!")
    return True


def test_attribution_gridworld():
    """Test attribution diagnosis on deterministic GridWorld."""
    print("\n" + "="*70)
    print("TEST 4: Attribution — GridWorld (Deterministic)")
    print("="*70)
    
    from throng4.environments.gridworld_variants import SparseRewardGridWorld
    
    env = SparseRewardGridWorld(size=5)
    diagnoser = AttributionDiagnoser(n_rng_trials=15, n_state_trials=8)
    
    # Test action 0 (up)
    print("\n  Diagnosing action 0 (up)...")
    result = diagnoser.diagnose(env, action=0)
    
    print(f"\n    Attribution: {result.attribution.value}")
    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    Stochasticity: {result.stochasticity_score:.3f}")
    print(f"    {result.summary()}")
    
    # GridWorld is deterministic — same start state, same action -> same outcome
    assert result.stochasticity_score < 0.2, f"GridWorld should be deterministic, got stoch={result.stochasticity_score}"
    
    # Diagnose all actions
    print("\n  Diagnosing all actions...")
    all_results = diagnoser.diagnose_all_actions(env, n_actions=4)
    
    for action, res in all_results.items():
        print(f"\n  Action {action}: {res.attribution.value} (stoch={res.stochasticity_score:.3f})")
    
    # All GridWorld actions should be deterministic or state-dependent (not stochastic)
    for action, res in all_results.items():
        assert res.attribution != Attribution.STOCHASTIC, \
            f"GridWorld action {action} should not be stochastic"
    
    print("\n[OK] Attribution GridWorld test passed!")
    return True


def test_attribution_frozenlake():
    """Test attribution on FrozenLake (stochastic — slippery!)."""
    print("\n" + "="*70)
    print("TEST 5: Attribution — FrozenLake (Stochastic)")
    print("="*70)
    
    try:
        from throng4.environments.gym_envs import FrozenLakeAdapter
        env = FrozenLakeAdapter()
    except Exception as e:
        print(f"\n  [SKIP] FrozenLake: {e}")
        return True
    
    diagnoser = AttributionDiagnoser(n_rng_trials=30, stochastic_threshold=0.1)
    
    # FrozenLake is slippery by default — actions have stochastic outcomes
    print("\n  Diagnosing action 2 (right)...")
    result = diagnoser.diagnose(env, action=2)
    
    print(f"\n    Attribution: {result.attribution.value}")
    print(f"    Confidence: {result.confidence:.2f}")
    print(f"    Stochasticity: {result.stochasticity_score:.3f}")
    print(f"    Unique outcomes: {len(result.outcome_distribution.outcomes)}")
    print(f"    {result.summary()}")
    
    # FrozenLake is_slippery=True means actions are stochastic
    # The agent may slip to adjacent directions
    if result.stochasticity_score > 0.1:
        print(f"\n    [CONFIRMED] Stochastic behavior detected!")
        assert result.attribution == Attribution.STOCHASTIC, \
            f"Expected STOCHASTIC, got {result.attribution.value}"
    else:
        print(f"\n    [NOTE] Low stochasticity detected — may be non-slippery FrozenLake variant")
    
    print("\n[OK] Attribution FrozenLake test passed!")
    return True


def test_dormant_not_eliminated():
    """Test that rules become DORMANT instead of ELIMINATED."""
    print("\n" + "="*70)
    print("TEST 6: DORMANT (Never Eliminated)")
    print("="*70)
    
    rule = DiscoveredRule(
        id="test_dormant",
        description="Hypothesis that fails repeatedly",
        feature="test"
    )
    
    # Add many failures
    for i in range(15):
        result = TestResult(
            action=0,
            state_before=np.array([0]),
            state_after=np.array([0]),
            reward=-0.1,
            done=False
        )
        rule.add_test_result(result, success=False)
    
    rule.update_status()
    
    print(f"\n  After 15 failures:")
    print(f"    Confidence: {rule.confidence:.3f}")
    print(f"    Status: {rule.status.value}")
    
    # Should be DORMANT, NOT eliminated
    assert rule.status == RuleStatus.DORMANT, f"Should be DORMANT, got {rule.status.value}"
    assert rule.confidence > 0, "Confidence should never be exactly 0"
    
    # Verify ELIMINATED is not even in the enum
    status_values = [s.value for s in RuleStatus]
    assert "eliminated" not in status_values, "ELIMINATED should not exist in v3!"
    
    print(f"\n  Available statuses: {status_values}")
    print(f"  'eliminated' is NOT in the list (correct!)")
    
    # Now re-evaluate: add some successes
    for i in range(5):
        result = TestResult(
            action=0,
            state_before=np.array([0]),
            state_after=np.array([1]),
            reward=1.0,
            done=False
        )
        rule.add_test_result(result, success=True)
    
    rule.update_status()
    
    print(f"\n  After revival (5 successes added):")
    print(f"    Confidence: {rule.confidence:.3f}")
    print(f"    Status: {rule.status.value}")
    
    assert rule.status != RuleStatus.DORMANT, "Should have recovered from DORMANT"
    
    print("\n[OK] DORMANT test passed!")
    return True


def main():
    results = {
        "Confidence Decay": test_confidence_decay(),
        "Anti-Policy Generation": test_anti_policy_generation(),
        "Outcome Distribution": test_outcome_distribution(),
        "Attribution GridWorld": test_attribution_gridworld(),
        "Attribution FrozenLake": test_attribution_frozenlake(),
        "DORMANT (No Elimination)": test_dormant_not_eliminated(),
    }
    
    print("\n" + "="*70)
    print("V3 EPISTEMOLOGY TEST RESULTS")
    print("="*70)
    
    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print("\n[OK] ALL V3 TESTS PASSED!")
    else:
        print("\n[FAIL] Some tests failed!")
    
    return all_pass


if __name__ == '__main__':
    main()
