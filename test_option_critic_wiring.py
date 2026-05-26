"""Verify Option-Critic and SR wiring on WholeBrain (torch path)."""

from __future__ import annotations

import numpy as np
from brain.orchestrator import WholeBrain


def test_oc_trains_and_sr_wired():
    brain = WholeBrain(
        n_features=84,
        n_actions=18,
        session_name="oc_wiring_test",
        enable_logging=False,
        use_torch=True,
        enabled_systems={
            "causal_model": False,
            "dreams": False,
            "counterfactual": False,
            "rehearsal_loop": False,
            "probe_runner": False,
        },
    )
    assert brain.hippocampus._sr is not None, "SR should be enabled on hippocampus"
    assert brain.striatum._sr_matrix is brain.hippocampus._sr, "SR wired to striatum"
    assert brain.striatum.option_critic is not None, "OC should be enabled with use_torch"

    rng = np.random.RandomState(0)
    for i in range(120):
        obs = rng.randn(84).astype(np.float32)
        prev = i % 18
        result = brain.step(obs, prev_action=prev, reward=0.1, done=False)
        assert "action" in result

    oc_updates = brain.striatum.option_critic._total_updates
    assert oc_updates > 0, f"OC should receive updates during learn(), got {oc_updates}"
    print(f"[PASS] OC updates={oc_updates}, sr_wired=True")
    brain.close()


if __name__ == "__main__":
    test_oc_trains_and_sr_wired()
    print("ALL OC WIRING TESTS PASSED")
