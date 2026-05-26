"""Root pytest fixtures for throng5 brain integration tests."""

from __future__ import annotations

import pytest

from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()

# Script-style modules that execute side effects at import time.
collect_ignore = ["test_tetra_dialogue.py"]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring external services (OpenClaw gateway, etc.)",
    )
    config.addinivalue_line(
        "markers",
        "slow: longer-running integration/pretrain checks",
    )


@pytest.fixture
def brain():
    """WholeBrain with default wiring (planner enabled when causal_model on)."""
    from brain.orchestrator import WholeBrain

    b = WholeBrain(n_features=84, n_actions=4, enable_logging=False)
    yield b
    b.close()
