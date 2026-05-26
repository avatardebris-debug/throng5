"""IntentionalController reuses brain.planner when wired (no duplicate graphs)."""

import numpy as np
from brain.orchestrator import WholeBrain
from brain.games.montezuma.runner.intentional import IntentionalController
from brain.planning.object_graph import ObjectGraph


def test_controller_shares_brain_planner():
    brain = WholeBrain(n_features=128, n_actions=18, session_name="ic_planner_test")
    og = ObjectGraph()
    ic = IntentionalController(brain, og, n_actions=18)

    assert brain.planner is not None, "wire_subsystems should attach planner"
    assert ic.subgoal_planner is brain.planner
    assert ic.landmark_graph is brain.planner.graph
    assert ic.causal_model is brain._causal_model
    assert ic._delegate_causal_observe is True
    assert brain._dead_end_detector is not None
    assert brain.planner.dead_end_detector is brain._dead_end_detector

    ram = np.random.randint(0, 256, 128, dtype=np.uint8)
    features = ram.astype(np.float32) / 255.0
    gs = {
        "player_x": 10, "player_y": 100, "room": 0,
        "lives": 3, "score": 0, "skull_x": 50, "skull_y": 100,
        "items_mask": 0,
        "items": {"key": {"x": 15, "y": 200, "collected": False}},
        "enemies": {"skull": {"x": 50, "y": 100}},
        "has_key": False,
    }
    ic.on_episode_start(gs)
    brain.step(features, prev_action=0, reward=0.0, done=False)
    out = ic.step(gs, features, brain_action=1, reward=0.0, done=False)
    assert "action" in out
    brain.close()
    print("[PASS] IntentionalController shares brain planner stack")


def test_planner_trap_observe_no_attribute_error():
    """Planner transition with trap check must not fail when dead_end is wired."""
    brain = WholeBrain(
        n_features=128, n_actions=18, session_name="trap_observe_test",
        game_mode="puzzle",
    )
    assert brain.planner is not None
    assert brain.planner.dead_end_detector is not None

    f0 = np.random.randn(128).astype(np.float32)
    f1 = f0 + 0.01
    # Positive reward path exercises is_trap branch in observe_transition.
    brain.planner.observe_transition(f0, 1, f1, reward=1.0, done=False)
    brain.step(f1, prev_action=1, reward=0.0, done=False)
    brain.close()
    print("[PASS] planner observe_transition with wired dead_end_detector")


if __name__ == "__main__":
    test_controller_shares_brain_planner()
    test_planner_trap_observe_no_attribute_error()
