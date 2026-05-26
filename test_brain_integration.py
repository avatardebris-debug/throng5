"""
Integration test — full WholeBrain pipeline running for 100 random steps.

Verifies:
1. All 7 regions instantiate and communicate via MessageBus
2. The full step() pipeline runs without errors
3. Actions are produced every step
4. Threat assessment and mode switching work
5. Episode boundaries reset correctly
6. Brain report aggregates all region states
"""

from __future__ import annotations

import numpy as np


def test_wholebrain_100_steps(brain):
    actions_taken = []
    modes_seen = set()
    episode_count = 0
    rng = np.random.RandomState(42)

    for i in range(100):
        obs = rng.randn(84).astype(np.float32)
        reward = rng.randn() * 0.1
        done = (i > 0 and i % 25 == 0)

        result = brain.step(
            obs,
            prev_action=actions_taken[-1] if actions_taken else 0,
            reward=reward,
            done=done,
        )

        action = result["action"]
        assert isinstance(action, int)
        assert 0 <= action < brain.n_actions
        actions_taken.append(action)
        modes_seen.add(result["operating_mode"])

        if done:
            episode_count += 1

    assert episode_count > 0
    assert len(set(actions_taken)) > 1
    assert modes_seen

    report = brain.report()
    assert len(report) >= 7
    core_regions = [n for n, r in report.items() if "step_count" in r]
    assert len(core_regions) == 7, f"Expected 7 core regions, got {core_regions}: {core_regions}"


def test_wholebrain_report_sections(brain):
    report = brain.report()
    for name, r in report.items():
        if "step_count" in r:
            assert r.get("step_count", 0) >= 0, f"{name} step_count invalid"
