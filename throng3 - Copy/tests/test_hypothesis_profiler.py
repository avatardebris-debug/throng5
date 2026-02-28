"""
Tests for Phase 7: Hypothesis Profiling & Dreamer-as-Teacher.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_hypothesis_profile():
    print("=" * 60)
    print("TEST 1: HypothesisProfile")
    print("=" * 60)

    from throng4.basal_ganglia.hypothesis_profiler import HypothesisProfile

    profile = HypothesisProfile(
        hypothesis_id=0, hypothesis_name="jump", state_dim=16
    )

    # Simulate: "jump" works well in high states, poorly in low states
    for i in range(50):
        state = np.random.randn(16).astype(np.float32)
        if state.mean() > 0:
            # High state = jump works well → rank 0 (best)
            profile.record(state, reward=5.0, rank=0, total_hypotheses=3, step=i)
        else:
            # Low state = jump works poorly → rank 2 (worst)
            profile.record(state, reward=-3.0, rank=2, total_hypotheses=3, step=i)

    print(f"  {profile.summary()}")
    print(f"  Win rate: {profile.win_rate:.1%}")
    print(f"  Specialization: {profile.specialization_score:.2f}")

    # Should activate more in "high" states
    high_state = np.ones(16, dtype=np.float32) * 0.5
    low_state = np.ones(16, dtype=np.float32) * -0.5
    act_high = profile.should_activate(high_state)
    act_low = profile.should_activate(low_state)
    print(f"  Activation (high state): {act_high:.2f}")
    print(f"  Activation (low state): {act_low:.2f}")

    assert profile.total_evaluations == 50
    assert profile.specialization_score > 0.1, "Should show specialization"
    assert act_high > act_low, "Should prefer high states (where it works)"

    print("[PASS] HypothesisProfile works!\n")
    return profile


def test_options_library():
    print("=" * 60)
    print("TEST 2: OptionsLibrary")
    print("=" * 60)

    from throng4.basal_ganglia.hypothesis_profiler import (
        OptionsLibrary, HypothesisProfile, OptionStatus,
    )

    library = OptionsLibrary(state_dim=16, n_actions=4)

    # Create a specialized profile
    profile = HypothesisProfile(
        hypothesis_id=0, hypothesis_name="dodge", state_dim=16
    )
    for i in range(30):
        state = np.random.randn(16).astype(np.float32)
        if i % 3 == 0:
            profile.record(state, reward=3.0, rank=0, total_hypotheses=3, step=i)
        else:
            profile.record(state, reward=-1.0, rank=2, total_hypotheses=3, step=i)

    # Try to discover option
    option = library.discover_option(profile, current_step=30)
    if option:
        print(f"  Discovered: {option.summary()}")
    else:
        print("  Not specialized enough to discover (expected for random data)")

    # Create a strongly specialized profile
    profile2 = HypothesisProfile(
        hypothesis_id=1, hypothesis_name="climb", state_dim=16
    )
    for i in range(30):
        state = np.ones(16, dtype=np.float32) * (1 if i % 2 == 0 else -1)
        if i % 2 == 0:
            profile2.record(state, reward=10.0, rank=0, total_hypotheses=3, step=i)
        else:
            profile2.record(state, reward=-5.0, rank=2, total_hypotheses=3, step=i)

    option2 = library.discover_option(profile2, current_step=60)
    if option2:
        print(f"  Discovered: {option2.summary()}")

        # Test activation
        good_state = np.ones(16, dtype=np.float32)
        activation = option2.should_initiate(good_state)
        print(f"  Activation on good state: {activation:.2f}")

        # Record outcomes
        for _ in range(15):
            library.record_outcome(option2.option_id, reward=2.0)
        print(f"  After 15 uses: {option2.summary()}")
        assert option2.times_used == 15
        assert option2.success_rate > 0.8

    print(f"  {library.summary()}")
    print("[PASS] OptionsLibrary works!\n")


def test_dreamer_teacher():
    print("=" * 60)
    print("TEST 3: DreamerTeacher")
    print("=" * 60)

    from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher
    from throng4.basal_ganglia.dreamer_engine import DreamResult

    teacher = DreamerTeacher(n_actions=4, state_dim=16)

    # Simulate 50 dream cycles
    for step in range(50):
        state = np.random.randn(16).astype(np.float32)

        # Create mock dream results
        results = []
        for h in range(3):
            trajectory = [np.random.randint(4) for _ in range(10)]
            rewards = [float(np.random.randn()) for _ in range(10)]
            total = sum(rewards)
            results.append(DreamResult(
                hypothesis_id=h,
                hypothesis_name=["jump", "dodge", "climb"][h],
                predicted_rewards=rewards,
                total_predicted_reward=total,
                avg_predicted_reward=total / 10,
                worst_step_reward=min(rewards),
                best_step_reward=max(rewards),
                confidence=0.5,
                simulation_time_ms=1.0,
                trajectory=trajectory,
            ))

        # Sort like DreamerEngine does
        results.sort(key=lambda r: r.total_predicted_reward, reverse=True)

        # Process
        signals = teacher.process_dream_results(results, state, step)

        # Record that policy sometimes follows, sometimes doesn't
        if signals:
            best_action = signals[0].action_recommended
            if step % 3 == 0:
                # Policy follows dreamer
                teacher.record_policy_action(state, best_action, best_action, 1.0)
            else:
                # Policy ignores dreamer
                teacher.record_policy_action(state, (best_action + 1) % 4, best_action, 0.5)

    print(f"  {teacher.summary()}")
    print(f"  Dreamer reliance: {teacher.dreamer_reliance:.1%}")
    print(f"  Recommended interval: {teacher.recommended_dream_interval}")
    print(f"  Dreamer needed: {teacher.dreamer_is_needed}")

    # Verify profiles were created
    assert len(teacher.profiles) == 3, "Should have 3 hypothesis profiles"
    for p in teacher.profiles.values():
        assert p.total_evaluations > 0

    assert teacher._total_signals_generated > 0

    print("[PASS] DreamerTeacher works!\n")


def test_dreamer_reliance_decreases():
    print("=" * 60)
    print("TEST 4: Dreamer Reliance Decreases Over Time")
    print("=" * 60)

    from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher
    from throng4.basal_ganglia.dreamer_engine import DreamResult

    teacher = DreamerTeacher(n_actions=4, state_dim=16)

    # Phase 1: Policy follows dreamer (high reliance)
    for step in range(100):
        state = np.random.randn(16).astype(np.float32)
        results = [DreamResult(
            hypothesis_id=0, hypothesis_name="h0",
            predicted_rewards=[1.0]*10, total_predicted_reward=10.0,
            avg_predicted_reward=1.0, worst_step_reward=1.0, best_step_reward=1.0,
            confidence=0.5, simulation_time_ms=1.0, trajectory=[0]*10,
        )]
        teacher.process_dream_results(results, state, step)
        teacher.record_policy_action(state, 0, 0, 1.0)  # Always follows

    reliance_early = teacher.dreamer_reliance
    print(f"  Early reliance (policy follows): {reliance_early:.1%}")

    # Phase 2: Policy stops following (autonomous)
    for step in range(100, 300):
        state = np.random.randn(16).astype(np.float32)
        results = [DreamResult(
            hypothesis_id=0, hypothesis_name="h0",
            predicted_rewards=[1.0]*10, total_predicted_reward=10.0,
            avg_predicted_reward=1.0, worst_step_reward=1.0, best_step_reward=1.0,
            confidence=0.5, simulation_time_ms=1.0, trajectory=[0]*10,
        )]
        teacher.process_dream_results(results, state, step)
        teacher.record_policy_action(state, 2, 0, 1.0)  # Ignores dreamer

    reliance_late = teacher.dreamer_reliance
    print(f"  Late reliance (policy autonomous): {reliance_late:.1%}")
    print(f"  Interval changed: {teacher.recommended_dream_interval}")

    assert reliance_late < reliance_early, "Reliance should decrease"
    assert teacher.recommended_dream_interval > 1, "Should back off"

    print("[PASS] Reliance decreases correctly!\n")


def test_controller_integration():
    print("=" * 60)
    print("TEST 5: Controller Integration")
    print("=" * 60)

    from throng4.meta_policy.meta_policy_controller import MetaPolicyController

    controller = MetaPolicyController()

    assert hasattr(controller, 'dreamer_teacher'), "Missing dreamer_teacher"
    print("  Controller has dreamer_teacher")

    status = controller._get_meta_status()
    assert 'dreamer_reliance' in status
    assert 'dreamer_needed' in status
    assert 'active_options' in status
    assert 'dream_interval' in status
    print(f"  Meta status: reliance={status['dreamer_reliance']:.1%}, "
          f"needed={status['dreamer_needed']}, "
          f"options={status['active_options']}, "
          f"interval={status['dream_interval']}")

    print("[PASS] Controller integration works!\n")


def test_full_package_imports():
    print("=" * 60)
    print("TEST 6: Full Package Imports")
    print("=" * 60)

    from throng4.basal_ganglia import (
        DreamerEngine, Amygdala, CompressedStateEncoder,
        DreamResult, DangerSignal,
        DreamerTeacher, HypothesisProfile, OptionsLibrary,
        BehavioralOption, TeachingSignal,
    )
    print("  All 10 exports accessible")
    print("[PASS] Package imports work!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 7 TEST SUITE — Hypothesis Profiling & Dreamer-as-Teacher")
    print("=" * 60 + "\n")

    test_hypothesis_profile()
    test_options_library()
    test_dreamer_teacher()
    test_dreamer_reliance_decreases()
    test_controller_integration()
    test_full_package_imports()

    print("=" * 60)
    print("ALL 6 TESTS PASSED")
    print("=" * 60)
