"""
Tests for throng4.basal_ganglia module.
Bridge Step 4 verification.
"""

import numpy as np
import sys
import os

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_compressed_state():
    print("=" * 60)
    print("TEST 1: CompressedStateEncoder")
    print("=" * 60)

    from throng4.basal_ganglia.compressed_state import (
        CompressedStateEncoder, EncodingMode,
    )

    # Binary grid encoding (Tetris-like)
    enc = CompressedStateEncoder(
        mode=EncodingMode.BINARY_GRID, grid_shape=(10, 6)
    )
    board = np.random.rand(60)
    compressed = enc.encode(board)
    print(f"  Binary grid: {board.shape} -> {compressed.data.shape}")
    print(f"  Compression ratio: {compressed.compression_ratio:.1f}x")
    decoded = enc.decode(compressed)
    assert decoded.size > 0, "Decoded should not be empty"

    # Quantized encoding (generic)
    enc2 = CompressedStateEncoder(
        mode=EncodingMode.QUANTIZED, n_quantize_levels=4
    )
    obs = np.random.randn(128)
    comp2 = enc2.encode(obs)
    dec2 = enc2.decode(comp2)
    err = np.mean(np.abs(obs - dec2.flatten()))
    print(f"  Quantized: {obs.shape} -> {comp2.data.shape}, error={err:.4f}")
    assert err < 1.0, f"Quantization error too high: {err}"

    # Downsampled encoding (Atari-like)
    enc3 = CompressedStateEncoder(
        mode=EncodingMode.DOWNSAMPLED, downsample_shape=(8, 8)
    )
    frame = np.random.rand(84, 84)
    comp3 = enc3.encode(frame)
    print(f"  Downsampled: {frame.shape} -> {comp3.data.shape}")
    print(f"  Compression ratio: {comp3.compression_ratio:.1f}x")
    assert comp3.data.shape == (8, 8), "Downsample shape wrong"
    assert comp3.compression_ratio > 100, "Should compress 84x84 significantly"

    # Calibration
    cal = enc2.calibrate(np.random.randn(20, 128))
    print(f"  Calibration: mean_err={cal['mean_error']:.4f}")
    assert "mean_error" in cal
    assert "max_error" in cal

    print("[PASS] CompressedStateEncoder works!\n")


def test_dreamer_engine():
    print("=" * 60)
    print("TEST 2: DreamerEngine")
    print("=" * 60)

    from throng4.basal_ganglia.dreamer_engine import DreamerEngine

    dreamer = DreamerEngine(
        n_hypotheses=3, network_size="micro",
        state_size=32, n_actions=4
    )
    print(f"  DreamerEngine created: {dreamer.network_size.value}")
    assert not dreamer.is_calibrated, "Should not be calibrated yet"

    # Train world model
    for i in range(100):
        s = np.random.randn(32).astype(np.float32)
        a = np.random.randint(4)
        s2 = s + np.random.randn(32) * 0.1
        r = float(np.random.randn())
        dreamer.learn(s, a, s2, r)

    print(f"  Calibrated: {dreamer.is_calibrated}")
    assert dreamer.is_calibrated, "Should be calibrated after 100 steps"

    # Run dream
    hypotheses = dreamer.create_default_hypotheses(4)
    state = np.random.randn(32).astype(np.float32)
    results = dreamer.dream(state, hypotheses, n_steps=10)

    print(f"  Dream results ({len(results)} hypotheses):")
    for r in results:
        print(f"    {r.summary()}")
    print(f"  Avg dream time: {dreamer.avg_dream_time_ms:.1f}ms")

    assert len(results) == 3, "Should have 3 hypothesis results"
    assert all(len(r.predicted_rewards) == 10 for r in results)
    assert all(len(r.trajectory) == 10 for r in results)
    
    # Speed check
    assert dreamer.avg_dream_time_ms < 100, (
        f"Dream too slow: {dreamer.avg_dream_time_ms:.1f}ms"
    )

    print("[PASS] DreamerEngine works!\n")
    return results


def test_amygdala(dream_results):
    print("=" * 60)
    print("TEST 3: Amygdala")
    print("=" * 60)

    from throng4.basal_ganglia.amygdala import Amygdala
    from throng4.basal_ganglia.dreamer_engine import DreamResult

    amygdala = Amygdala()

    # Test with mixed results
    danger = amygdala.assess_danger(dream_results, current_step=1)
    print(f"  Mixed results: {danger.summary()}")
    print(f"  Should override: {amygdala.should_override(danger, 1)}")

    # Test with all-negative results
    bad_results = [
        DreamResult(
            hypothesis_id=i,
            hypothesis_name=f"bad_{i}",
            predicted_rewards=[-2.0] * 10,
            total_predicted_reward=-20.0,
            avg_predicted_reward=-2.0,
            worst_step_reward=-2.0,
            best_step_reward=-2.0,
            confidence=0.5,
            simulation_time_ms=1.0,
            trajectory=[0] * 10,
        )
        for i in range(3)
    ]
    danger2 = amygdala.assess_danger(bad_results, current_step=20)
    print(f"  All-negative: {danger2.summary()}")
    print(f"  Should override: {amygdala.should_override(danger2, 20)}")
    print(f"  Action: {danger2.recommended_action.value}")

    assert danger2.all_hypotheses_negative, "Should detect all negative"
    assert danger2.should_override, "Should recommend override"
    assert danger2.n_catastrophic == 3, "All hypotheses are catastrophic"

    # Test with empty results
    safe = amygdala.assess_danger([], current_step=30)
    assert not safe.should_override, "Empty results should be safe"

    # Test cooldown
    amygdala.record_override(20)
    should = amygdala.should_override(danger2, 25)
    print(f"  Override during cooldown (step 25 after override@20): {should}")
    assert not should, "Cooldown should prevent override"

    # Test after cooldown
    should2 = amygdala.should_override(danger2, 35)
    print(f"  Override after cooldown (step 35): {should2}")

    print("[PASS] Amygdala works!\n")


def test_package_imports():
    print("=" * 60)
    print("TEST 4: Package Import")
    print("=" * 60)

    from throng4.basal_ganglia import (
        DreamerEngine, Amygdala, CompressedStateEncoder,
        DreamResult, DangerSignal,
    )
    print("  All exports OK")
    print("[PASS] Package imports work!\n")


def test_controller_integration():
    print("=" * 60)
    print("TEST 5: MetaPolicyController Integration")
    print("=" * 60)

    from throng4.meta_policy.meta_policy_controller import MetaPolicyController

    controller = MetaPolicyController()
    
    # Verify basal ganglia is initialized
    assert hasattr(controller, "basal_ganglia"), "Missing basal_ganglia"
    assert hasattr(controller, "amygdala"), "Missing amygdala"
    print("  Controller has basal_ganglia and amygdala")
    
    # Test meta status includes new fields
    status = controller._get_meta_status()
    assert "dreamer_calibrated" in status, "Missing dreamer_calibrated in status"
    assert "amygdala_alertness" in status, "Missing amygdala_alertness in status"
    print(f"  Meta status: dreamer_calibrated={status['dreamer_calibrated']}, "
          f"amygdala_alertness={status['amygdala_alertness']}")

    print("[PASS] Controller integration works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BRIDGE STEP 4 — BASAL GANGLIA TEST SUITE")
    print("=" * 60 + "\n")

    test_compressed_state()
    dream_results = test_dreamer_engine()
    test_amygdala(dream_results)
    test_package_imports()
    test_controller_integration()

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
