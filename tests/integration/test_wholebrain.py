"""
tests/integration/test_wholebrain.py — End-to-end integration test for WholeBrain.

Verifies:
  - WholeBrain initializes without errors (all subsystems)
  - 100 steps on random obs runs without crash
  - report() and get_diagnostic_info() return correct structure
  - Episode boundary (done=True) resets cleanly
  - Bare mode (all subsystems disabled) works the same
  - world_model dream_all_actions optimization correctness

Run:
  pytest tests/integration/test_wholebrain.py -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")


@pytest.fixture
def obs4():
    return np.random.rand(4).astype(np.float32)


@pytest.fixture
def obs84():
    return np.random.rand(84).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# WholeBrain lifecycle
# ════════════════════════════════════════════════════════════════════════

class TestWholeBrainInit:
    def test_bare_init(self):
        """Bare brain (all extras off) should init without errors."""
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(
            n_features=4, n_actions=2, enable_logging=False,
            enabled_systems={
                "world_model": False, "dreams": False, "causal_model": False,
                "skill_library": False, "attribution": False,
                "stage_classifier": False, "counterfactual": False,
                "hippocampus_store": True, "threat_gating": True, "probe_runner": False,
            },
        )
        assert brain.n_features == 4
        assert brain.n_actions == 2
        assert len(brain._regions) == 7

    def test_full_init_no_crash(self):
        """Full brain should init (some subsystems may fail gracefully with init_errors)."""
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False)
        # Any init errors should be captured, not raised
        assert isinstance(brain._init_errors, dict)


class TestWholeBrainStep:
    def test_100_steps_no_crash(self, obs84):
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        action = 0
        for i in range(100):
            done = (i == 99)
            result = brain.step(obs84, prev_action=action, reward=float(i % 5), done=done)
            assert "action" in result
            assert 0 <= result["action"] < 4
            action = result["action"]

    def test_episode_reset_clears_state(self, obs84):
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=2, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        # Run to end of episode
        for _ in range(10):
            brain.step(obs84, 0, 1.0, False)
        ep_reward_before = brain._episode_reward
        brain.step(obs84, 0, 5.0, done=True)  # Episode ends
        # After done, episode reward should be reset
        assert brain._episode_count == 1
        assert brain._episode_reward == 0.0

    def test_step_returns_required_keys(self, obs84):
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        result = brain.step(obs84, 0, 0.0, False)
        for key in ("action", "threat_score", "operating_mode", "epsilon"):
            assert key in result, f"Missing key: {key}"


class TestWholeBrainReport:
    def test_report_no_crash(self, obs84):
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        brain.step(obs84, 0, 0.0, False)
        r = brain.report()
        assert "sensory_cortex" in r
        assert "striatum" in r
        assert "amygdala_thalamus" in r

    def test_report_no_purged_attrs(self, obs84):
        """Regression: report() must not reference purged subsystems."""
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        brain.step(obs84, 0, 0.0, False)
        r = brain.report()
        for purged in ("rehearsal", "surprise_tracker", "entropy_monitor"):
            assert purged not in r, f"Purged subsystem '{purged}' still in report()"

    def test_diagnostic_no_purged_attrs(self, obs84):
        """Regression: get_diagnostic_info() must not reference purged subsystems."""
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        d = brain.get_diagnostic_info()
        active = d["active_subsystems"]
        for purged in ("curiosity", "meta_controller", "rehearsal",
                       "surprise_tracker", "entropy_monitor"):
            assert purged not in active, f"Purged attr '{purged}' still in diagnostic"

    def test_stubs_return_safely(self):
        from brain.orchestrator import WholeBrain
        brain = WholeBrain(n_features=4, n_actions=2, enable_logging=False,
                           enabled_systems={
                               "world_model": False, "dreams": False, "causal_model": False,
                               "skill_library": False, "attribution": False,
                               "stage_classifier": False, "counterfactual": False,
                           })
        assert brain.rehearse()["status"] == "not_available"
        assert brain.request_plateau_review() is None
        assert brain.dream()["status"] == "not_available"  # dream_loop disabled


# ════════════════════════════════════════════════════════════════════════
# WorldModel
# ════════════════════════════════════════════════════════════════════════

class TestWorldModel:
    def test_store_and_train(self):
        from brain.learning.world_model import WorldModel, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        wm = WorldModel(n_features=8, n_actions=2, batch_size=4)
        f = np.random.rand(8).astype(np.float32)
        for _ in range(5):
            wm.store_transition(f, 0, f + 0.1, 1.0)
        result = wm.train_step()
        assert "wm_loss" in result

    def test_not_ready_before_training(self):
        from brain.learning.world_model import WorldModel
        wm = WorldModel(n_features=8, n_actions=2)
        assert not wm.is_ready

    def test_dream_all_actions_shape(self):
        from brain.learning.world_model import WorldModel, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        wm = WorldModel(n_features=8, n_actions=4, batch_size=4)
        f = np.random.rand(8).astype(np.float32)
        for _ in range(12):
            wm.store_transition(f, np.random.randint(4), f + np.random.randn(8)*0.1, 1.0)
        for _ in range(10):
            wm.train_step()
        vals = wm.dream_all_actions(f, depth=3)
        assert vals.shape == (4,), f"Expected (4,), got {vals.shape}"

    def test_dream_all_actions_optimization_correctness(self):
        """Regression: optimized version must return same shape as n_actions."""
        from brain.learning.world_model import WorldModel, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        wm = WorldModel(n_features=4, n_actions=3, batch_size=4)
        f = np.zeros(4, dtype=np.float32)
        for _ in range(12):
            wm.store_transition(f, 0, f, 0.0)
        for _ in range(10):
            wm.train_step()
        vals = wm.dream_all_actions(f, depth=2)
        assert len(vals) == 3
        assert not np.any(np.isnan(vals))

    def test_checkpoint_round_trip(self):
        from brain.learning.world_model import WorldModel, TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        wm = WorldModel(n_features=4, n_actions=2, batch_size=4)
        f = np.random.rand(4).astype(np.float32)
        for _ in range(5):
            wm.store_transition(f, 0, f, 1.0)
        state = wm.save_state()
        wm2 = WorldModel(n_features=4, n_actions=2, batch_size=4)
        wm2.load_state(state)
        assert wm2._total_updates == wm._total_updates


# ════════════════════════════════════════════════════════════════════════
# CausalModel
# ════════════════════════════════════════════════════════════════════════

class TestCausalModel:
    def test_observe_and_predict(self):
        from brain.planning.causal_model import CausalModel
        cm = CausalModel()
        f = np.random.rand(8).astype(np.float32)
        for _ in range(6):
            cm.observe(f, action=1, features_after=f + 0.5, reward=-1.0, is_dead_end=True)
        # Action 1 at this state should now be flagged dangerous
        assert cm.is_action_dangerous(f, 1)

    def test_safe_actions_fallback(self):
        from brain.planning.causal_model import CausalModel
        cm = CausalModel()
        f = np.zeros(8, dtype=np.float32)
        # No observations yet — all actions should be safe
        safe = cm.get_safe_actions(f, n_actions=4)
        assert len(safe) == 4

    def test_precondition_tracking(self):
        from brain.planning.causal_model import CausalModel
        cm = CausalModel()
        cm.add_precondition(goal_hash=42, required_state_hash=7)
        assert 7 in cm.get_preconditions(42)

    def test_report(self):
        from brain.planning.causal_model import CausalModel
        cm = CausalModel()
        r = cm.report()
        assert "total_observations" in r
        assert "dangerous_effects" in r
