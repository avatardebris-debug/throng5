"""
test_blind_protocol.py — Fast unit tests for the blind hypothesis protocol.

Tests:
  1. Blindness leak detection — ensures no game strings appear in prompts/logs
  2. Generality value validation — rejects invalid generality values
  3. Confidence range validation — rejects out-of-range confidence
  4. Mask integrity — assert_mask_binary catches non-binary mask values
  5. Registry persistence — labels survive module reload simulation
  6. Anonymization stability — same game_id always gets same label

Run with:
    python -m pytest test_blind_protocol.py -v
or:
    python test_blind_protocol.py
"""

import json
import sys
import tempfile
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Blindness leak detection
# ---------------------------------------------------------------------------

def test_blindness_leak_clean():
    """Abstract log with no game strings should pass cleanly."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    clean = "Step 001 | act: 2 | agent:(0.54,0.95) target:(0.76,0.70) reward:0.0"
    leaks = OfflineGenerator.check_blindness_leak(clean)
    assert leaks == [], f"Expected no leaks, got: {leaks}"
    print("  PASS: clean log has no leaks")


def test_blindness_leak_detected():
    """Log containing 'Breakout' or 'ALE/' must be caught."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    leaky_log = "Step 001 | Action: Right | Paddle_X: 086, Ball: (195, 179) | Lives: 3"
    leaks = OfflineGenerator.check_blindness_leak(leaky_log)
    assert len(leaks) > 0, "Expected leak detection but found none"
    print(f"  PASS: detected leaks: {leaks}")


def test_blindness_leak_raises_with_label():
    """check_blindness_leak with label= should raise RuntimeError on leaky input."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    leaky = "This is from ALE/Breakout-v5"
    try:
        OfflineGenerator.check_blindness_leak(leaky, label="test_prompt")
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Blindness leak" in str(e)
        print(f"  PASS: RuntimeError raised correctly: {str(e)[:60]}…")


# ---------------------------------------------------------------------------
# 2. Generality value validation
# ---------------------------------------------------------------------------

def test_generality_valid_values():
    """Valid generality values should produce no errors."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    gen = OfflineGenerator.__new__(OfflineGenerator)
    for val in ("universal", "class", "instance"):
        h = {"id": "x", "description": "d", "object": "agent", "feature": "agent_x",
             "direction": "maximize", "confidence": 0.8, "trigger": "t", "generality": val}
        errors = gen._validate_hypotheses([h])
        assert errors == [], f"Expected no errors for generality={val!r}, got: {errors}"
    print("  PASS: all valid generality values accepted")


def test_generality_invalid_value():
    """Invalid generality value should produce a validation error."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    gen = OfflineGenerator.__new__(OfflineGenerator)
    h = {"id": "x", "description": "d", "object": "agent", "feature": "agent_x",
         "direction": "maximize", "confidence": 0.8, "trigger": "t", "generality": "maybe"}
    errors = gen._validate_hypotheses([h])
    assert any("generality" in e for e in errors), f"Expected generality error, got: {errors}"
    print(f"  PASS: invalid generality caught: {errors[0][:70]}")


def test_confidence_out_of_range():
    """Confidence > 1.0 or < 0.0 should produce a validation error."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    gen = OfflineGenerator.__new__(OfflineGenerator)
    h = {"id": "x", "description": "d", "object": "agent", "feature": "agent_x",
         "direction": "maximize", "confidence": 1.5, "trigger": "t", "generality": "universal"}
    errors = gen._validate_hypotheses([h])
    assert any("confidence" in e for e in errors), f"Expected confidence error, got: {errors}"
    print(f"  PASS: out-of-range confidence caught: {errors[0][:70]}")


def test_missing_generality_key():
    """Missing generality key should produce a missing-key error (not a value error)."""
    from throng4.llm_policy.offline_generator import OfflineGenerator
    gen = OfflineGenerator.__new__(OfflineGenerator)
    h = {"id": "x", "description": "d", "object": "agent", "feature": "agent_x",
         "direction": "maximize", "confidence": 0.8, "trigger": "t"}  # no generality
    errors = gen._validate_hypotheses([h])
    assert any("missing keys" in e for e in errors), f"Expected missing key error, got: {errors}"
    print(f"  PASS: missing generality key caught: {errors[0][:70]}")


# ---------------------------------------------------------------------------
# 3. Mask integrity assertion
# ---------------------------------------------------------------------------

def test_mask_binary_clean():
    """A properly constructed to_vector() should pass mask integrity check."""
    from throng4.learning.abstract_features import (
        AbstractFeature, assert_mask_binary, make_ext, empty_core
    )
    core = empty_core()
    ext, mask = make_ext([0.5, 0.3, 0.8])
    af = AbstractFeature(core=core, ext=ext, ext_mask=mask)
    vec = af.to_vector()
    assert_mask_binary(vec, label="clean_vector")  # should not raise
    print("  PASS: clean mask passes binary assertion")


def test_mask_binary_drift_detected():
    """Artificially corrupted mask should be caught."""
    from throng4.learning.abstract_features import (
        AbstractFeature, assert_mask_binary, make_ext, empty_core, CORE_SIZE, EXT_MAX
    )
    core = empty_core()
    ext, mask = make_ext([0.5])
    af = AbstractFeature(core=core, ext=ext, ext_mask=mask)
    vec = af.to_vector().copy()
    vec[CORE_SIZE + EXT_MAX + 3] = 0.42  # corrupt a mask slot
    try:
        assert_mask_binary(vec, label="drifted_mask")
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "non-binary values" in str(e)
        print(f"  PASS: mask drift detected: {str(e)[:70]}…")


# ---------------------------------------------------------------------------
# 4. Registry persistence
# ---------------------------------------------------------------------------

def test_registry_persistence():
    """Labels should be stable: same game_id always gets same label across calls."""
    from throng4.llm_policy.offline_generator import _get_blind_label, _GAME_LABELS
    label_a = _get_blind_label("ALE/Breakout-v5")
    label_a2 = _get_blind_label("ALE/Breakout-v5")
    assert label_a == label_a2, f"Label changed: {label_a} → {label_a2}"
    assert label_a.startswith("Environment-"), f"Unexpected format: {label_a}"
    print(f"  PASS: stable label: ALE/Breakout-v5 → {label_a}")


def test_anonymization_uniqueness():
    """Different game_ids should get different labels."""
    from throng4.llm_policy.offline_generator import _get_blind_label
    labels = [_get_blind_label(g) for g in
              ["ALE/Breakout-v5", "ALE/Pong-v5", "ALE/Freeway-v5"]]
    assert len(set(labels)) == len(labels), f"Label collision: {labels}"
    print(f"  PASS: unique labels: {labels}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_blindness_leak_clean,
        test_blindness_leak_detected,
        test_blindness_leak_raises_with_label,
        test_generality_valid_values,
        test_generality_invalid_value,
        test_confidence_out_of_range,
        test_missing_generality_key,
        test_mask_binary_clean,
        test_mask_binary_drift_detected,
        test_registry_persistence,
        test_anonymization_uniqueness,
    ]
    failed = 0
    for t in tests:
        name = t.__name__
        print(f"\n{name}")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
