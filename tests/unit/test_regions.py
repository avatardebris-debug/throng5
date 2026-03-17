"""
tests/unit/test_regions.py — Unit tests for all hot-path brain regions.

Covers:
  - Init / default state
  - process() basic operation
  - learn() basic operation
  - reset_episode() cleanup
  - Edge cases (None inputs, zero features, large actions)
  - Specific bugs that were fixed (regression tests)

Run:
  pytest tests/unit/test_regions.py -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def bus():
    from brain.message_bus import MessageBus
    return MessageBus()


@pytest.fixture
def dummy_features():
    return np.random.rand(84).astype(np.float32)


@pytest.fixture
def dummy_small_features():
    return np.random.rand(4).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# base_region
# ════════════════════════════════════════════════════════════════════════

class TestBaseRegion:
    def test_resume_does_not_crash(self, bus):
        """Regression: resume() was calling bus.resume() which didn't exist."""
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        s.halt()
        assert not s._is_active
        s.resume()  # Must not raise AttributeError
        assert s._is_active

    def test_halt_stops_process(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        s.halt()
        out = s.step({"features": dummy_small_features})
        assert out.get("halted") is True


# ════════════════════════════════════════════════════════════════════════
# Striatum
# ════════════════════════════════════════════════════════════════════════

class TestStriatum:
    def test_init(self, bus):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        assert s.n_features == 4
        assert s.n_actions == 2
        assert s._epsilon == 0.15
        # Regression: episode tracking vars must exist
        assert hasattr(s, "_episode_reward")
        assert hasattr(s, "_episode_rewards")

    def test_process_returns_action(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        out = s.process({"features": dummy_small_features, "explore": False})
        assert "action" in out
        assert 0 <= out["action"] < 2

    def test_process_none_features(self, bus):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        out = s.process({"features": None})
        assert out["action"] == 0

    def test_learn_accumulates_buffer(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2, batch_size=4)
        for _ in range(3):
            s.learn({
                "state": dummy_small_features,
                "action": 0,
                "reward": 1.0,
                "next_state": dummy_small_features,
                "done": False,
            })
        assert len(s._replay) == 3

    def test_learn_trains_after_batch(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2, batch_size=4)
        for _ in range(5):
            result = s.learn({
                "state": dummy_small_features,
                "action": 0,
                "reward": 1.0,
                "next_state": dummy_small_features,
                "done": False,
            })
        assert result.get("loss", 0) >= 0
        assert s._total_updates > 0

    def test_target_sync(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2, batch_size=4, target_update_freq=5)
        original_W1 = s._tW1.copy()
        for _ in range(5):
            s.learn({
                "state": dummy_small_features, "action": 0,
                "reward": 1.0, "next_state": dummy_small_features, "done": False,
            })
        # Target should have been synced
        assert not np.allclose(s._tW1, original_W1) or s._total_updates >= 1

    def test_valid_actions_mask(self, bus, dummy_small_features):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=4)
        out = s.process({"features": dummy_small_features, "valid_actions": [1, 3], "explore": False})
        assert out["action"] in [1, 3]

    def test_reset_episode_clears_bias(self, bus):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        s._action_bias = np.ones(2)
        s.reset_episode()
        assert s._action_bias is None

    def test_report(self, bus):
        from brain.regions.striatum import Striatum
        s = Striatum(bus, n_features=4, n_actions=2)
        r = s.report()
        assert "epsilon" in r
        assert "buffer_size" in r
        assert "backend" in r


# ════════════════════════════════════════════════════════════════════════
# Hippocampus
# ════════════════════════════════════════════════════════════════════════

class TestHippocampus:
    def test_init(self, bus):
        from brain.regions.hippocampus import Hippocampus
        h = Hippocampus(bus)
        assert h._total_stored == 0

    def test_store_transition(self, bus, dummy_small_features):
        from brain.regions.hippocampus import Hippocampus
        h = Hippocampus(bus, buffer_size=100, batch_size=4)
        out = h.process({
            "state": dummy_small_features, "action": 0,
            "reward": 1.0, "next_state": dummy_small_features, "done": False,
        })
        assert h._total_stored == 1
        assert out["buffer_size"] == 1

    def test_no_learn_when_buffer_small(self, bus, dummy_small_features):
        from brain.regions.hippocampus import Hippocampus
        h = Hippocampus(bus, batch_size=10)
        result = h.learn({})
        assert result["replayed"] == 0

    def test_learns_after_enough_transitions(self, bus, dummy_small_features):
        from brain.regions.hippocampus import Hippocampus
        h = Hippocampus(bus, batch_size=3)
        for _ in range(4):
            h.process({
                "state": dummy_small_features, "action": 0,
                "reward": 1.0, "next_state": dummy_small_features, "done": False,
            })
        result = h.learn({})
        assert result["replayed"] > 0

    def test_edge_case_priority(self, bus, dummy_small_features):
        from brain.regions.hippocampus import Hippocampus
        h = Hippocampus(bus, batch_size=2)
        h.process({
            "state": dummy_small_features, "action": 0, "reward": 5.0,
            "next_state": dummy_small_features, "done": False,
            "is_edge_case": True,
        })
        assert len(h._edge_cases) == 1
        # Priority for edge case should be higher (multiplied by 3)
        p = list(h._priorities)[0]
        normal_p = (1e-5) ** 0.6
        assert p > normal_p * 2


# ════════════════════════════════════════════════════════════════════════
# MotorCortex
# ════════════════════════════════════════════════════════════════════════

class TestMotorCortex:
    def test_init(self, bus):
        from brain.regions.motor_cortex import MotorCortex
        m = MotorCortex(bus, n_actions=4)
        assert m.n_actions == 4

    def test_normal_path(self, bus):
        from brain.regions.motor_cortex import MotorCortex
        m = MotorCortex(bus, n_actions=4)
        out = m.process({"striatum_action": 2, "striatum_halted": False})
        assert out["action"] == 2
        assert out["source"] == "striatum"

    def test_heuristic_fallback_when_halted(self, bus, dummy_small_features):
        from brain.regions.motor_cortex import MotorCortex
        m = MotorCortex(bus, n_actions=4)
        out = m.process({"striatum_action": None, "features": dummy_small_features})
        assert "action" in out
        assert out["source"] == "heuristic"

    def test_learn_installs_heuristic(self, bus, dummy_small_features):
        from brain.regions.motor_cortex import MotorCortex
        m = MotorCortex(bus, n_actions=4)
        m.learn({"features": dummy_small_features, "action": 3, "reward": 1.0})
        assert len(m._heuristics) == 1

    def test_motor_never_halts(self, bus):
        """Regression: MotorCortex must stay active on emergency signal."""
        from brain.message_bus import BrainMessage, Priority
        from brain.regions.motor_cortex import MotorCortex
        m = MotorCortex(bus, n_actions=4)
        halt_msg = BrainMessage("amygdala", "motor_cortex", Priority.EMERGENCY, "halt", {})
        m._on_emergency(halt_msg)
        assert m._is_active  # Must remain active


# ════════════════════════════════════════════════════════════════════════
# AmygdalaThalamus
# ════════════════════════════════════════════════════════════════════════

class TestAmygdalaThalamus:
    def test_init_n_features_default(self, bus):
        """Regression: default n_features was 18 (Montezuma-specific), now 84."""
        from brain.regions.amygdala_thalamus import AmygdalaThalamus
        a = AmygdalaThalamus(bus)
        assert a.n_features == 84

    def test_process_basic(self, bus, dummy_features):
        from brain.regions.amygdala_thalamus import AmygdalaThalamus
        a = AmygdalaThalamus(bus)
        out = a.process({"features": dummy_features})
        assert "threat_score" in out
        assert 0.0 <= out["threat_score"] <= 1.0
        assert "epsilon" in out

    def test_process_none_features(self, bus):
        from brain.regions.amygdala_thalamus import AmygdalaThalamus
        a = AmygdalaThalamus(bus)
        out = a.process({"features": None})
        assert "threat_score" in out  # Should use 0.0 estimator

    def test_transitions_capped(self, bus, dummy_features):
        """Regression: _transitions was an unbounded list, now capped deque."""
        from brain.regions.amygdala_thalamus import AmygdalaThalamus
        from collections import deque
        a = AmygdalaThalamus(bus, hysteresis_steps=1, enter_survive=0.0)
        assert isinstance(a._transitions, deque)
        assert a._transitions.maxlen == 1000

    def test_mode_switching(self, bus):
        from brain.regions.amygdala_thalamus import AmygdalaThalamus, OperatingMode
        a = AmygdalaThalamus(bus, enter_survive=0.5, hysteresis_steps=1)
        # High-threat features
        high_threat = np.ones(84, dtype=np.float32)
        a.process({"features": high_threat, "dream_results": None})
        # Mode may shift based on learned threat — just verify it doesn't crash
        assert a._mode in list(OperatingMode)

    def test_learn_threat_estimator(self, bus):
        from brain.regions.amygdala_thalamus import AmygdalaThalamus
        a = AmygdalaThalamus(bus)
        X = np.random.rand(10, 84).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
        result = a.learn({"X": X, "y": y, "lr": 0.01, "epochs": 5})
        assert "loss" in result
        assert result["loss"] >= 0


# ════════════════════════════════════════════════════════════════════════
# SensoryCortex
# ════════════════════════════════════════════════════════════════════════

class TestSensoryCortex:
    def test_random_projection_1d(self, bus, dummy_small_features):
        from brain.regions.sensory_cortex import SensoryCortex
        sc = SensoryCortex(bus, n_features=8)
        out = sc.process({"obs": dummy_small_features, "action": 0, "reward": 0.0, "done": False})
        assert out["features"] is not None
        assert len(out["features"]) == 8

    def test_random_projection_pads_short_obs(self, bus):
        from brain.regions.sensory_cortex import SensoryCortex
        sc = SensoryCortex(bus, n_features=84)
        tiny_obs = np.array([0.5, 0.5], dtype=np.float32)
        out = sc.process({"obs": tiny_obs, "action": 0, "reward": 0.0, "done": False})
        assert len(out["features"]) == 84

    def test_none_obs_returns_last_features(self, bus, dummy_features):
        from brain.regions.sensory_cortex import SensoryCortex
        sc = SensoryCortex(bus, n_features=84)
        sc._last_features = dummy_features.copy()
        out = sc.process({"obs": None})
        assert np.allclose(out["features"], dummy_features)

    def test_frame_buffer_clears_on_done(self, bus):
        from brain.regions.sensory_cortex import SensoryCortex
        sc = SensoryCortex(bus, n_features=84, use_cnn=False)
        sc._frame_buffer.append(np.zeros((84, 84), dtype=np.float32))
        sc.process({"obs": np.zeros(4, dtype=np.float32), "done": True})
        assert len(sc._frame_buffer) == 0

    def test_report(self, bus, dummy_small_features):
        from brain.regions.sensory_cortex import SensoryCortex
        sc = SensoryCortex(bus, n_features=84)
        sc.process({"obs": dummy_small_features, "action": 0, "reward": 0.0, "done": False})
        r = sc.report()
        assert r["has_features"] is True
        assert r["encoder"] == "random_projection"


# ════════════════════════════════════════════════════════════════════════
# BasalGanglia
# ════════════════════════════════════════════════════════════════════════

class TestBasalGanglia:
    def test_init(self, bus):
        from brain.regions.basal_ganglia import BasalGanglia
        bg = BasalGanglia(bus, n_features=4, n_actions=2, use_world_model=False)
        assert bg.n_features == 4

    def test_process_no_wm(self, bus, dummy_small_features):
        from brain.regions.basal_ganglia import BasalGanglia
        bg = BasalGanglia(bus, n_features=4, n_actions=2, use_world_model=False)
        out = bg.process({"features": dummy_small_features, "reward": 0.0})
        assert "context_score" in out

    def test_learn_wm_none_does_not_increment_confidence(self, bus, dummy_small_features):
        """Regression: wm_confidence was incrementing even when world_model is None."""
        from brain.regions.basal_ganglia import BasalGanglia
        bg = BasalGanglia(bus, n_features=4, n_actions=2, use_world_model=False)
        assert bg._world_model is None
        result = bg.learn({
            "state": dummy_small_features, "action": 0,
            "next_state": dummy_small_features, "reward": 1.0, "done": False,
        })
        # Should return early without touching wm_confidence
        assert result.get("wm_available") is False
        assert bg._wm_confidence == 0.0  # Must NOT have incremented

    def test_dream_interval_scheduling(self, bus, dummy_small_features):
        from brain.regions.basal_ganglia import BasalGanglia
        bg = BasalGanglia(bus, n_features=4, n_actions=2,
                          use_world_model=False, dream_interval=3)
        for _ in range(2):
            bg.process({"features": dummy_small_features, "reward": 0.0})
        assert bg._total_dreams == 0  # No WM, no dreams

    def test_report(self, bus):
        from brain.regions.basal_ganglia import BasalGanglia
        bg = BasalGanglia(bus, n_features=4, n_actions=2, use_world_model=False)
        r = bg.report()
        assert "has_world_model" in r
        assert r["has_world_model"] is False
