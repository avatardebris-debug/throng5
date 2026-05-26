"""
Live integration test for OpenClaw Bridge.
Tests real-time messaging to Tetra via the gateway.

Skipped by default when throng4 or gateway is unavailable.
Run with gateway live: pytest test_openclaw_bridge.py -m integration
"""
from __future__ import annotations

import pytest
from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()

try:
    from throng4.llm_policy.openclaw_bridge import OpenClawBridge
except ImportError:
    OpenClawBridge = None  # type: ignore[misc, assignment]

pytestmark = pytest.mark.integration

requires_openclaw = pytest.mark.skipif(
    OpenClawBridge is None,
    reason="throng4.llm_policy.openclaw_bridge not on PYTHONPATH",
)


@pytest.fixture
def require_gateway():
    """Skip live gateway tests when OpenClaw is down (checked per test, not at import)."""
    if OpenClawBridge is None:
        pytest.skip("throng4.llm_policy.openclaw_bridge not on PYTHONPATH")
    try:
        ok = OpenClawBridge(game="test").check_gateway()
    except Exception:
        ok = False
    if not ok:
        pytest.skip("OpenClaw gateway not reachable")


@requires_openclaw
def test_gateway_health(require_gateway):
    """Test 1: Can we reach the gateway?"""
    bridge = OpenClawBridge(game="test")
    assert bridge.check_gateway(), "Gateway is not running"


@requires_openclaw
def test_ping(require_gateway):
    """Test 2: Send a ping, get a response."""
    bridge = OpenClawBridge(game="test")
    response = bridge.query(
        "BRIDGE_TEST: Reply with exactly BRIDGE_OK to confirm connectivity."
    )
    assert response.success, f"Query failed: {response.error}"
    assert len(response.raw) > 0, "Empty response"


@requires_openclaw
def test_observation(require_gateway):
    """Test 3: Send a game observation, get hypothesis back."""
    bridge = OpenClawBridge(game="FrozenLake_4x4")
    response = bridge.send_observation(
        episode=1,
        observation=(
            "Action 2 (RIGHT) from state [1,1] leads to state [1,2] deterministically. "
            "But action 2 from state [2,3] sometimes leads to [2,3] (same position) or [3,3]. "
            "Success rate: 60% across 10 trials."
        ),
        context={
            "state_dims": [4, 4],
            "action": 2,
            "action_name": "RIGHT",
            "success_rate": 0.6,
            "n_trials": 10,
            "from_state": [2, 3],
            "expected_state": [2, 4],
            "actual_states": {"[2,4]": 6, "[2,3]": 3, "[3,3]": 1},
        },
    )
    assert response.success, f"Observation failed: {response.error}"


@requires_openclaw
def test_memory_write(tmp_path, monkeypatch):
    """Test 4: Memory file write (no gateway required)."""
    if OpenClawBridge is None:
        pytest.skip("throng4.llm_policy.openclaw_bridge not on PYTHONPATH")

    from datetime import datetime

    bridge = OpenClawBridge(game="TestGame")
    monkeypatch.setattr(bridge, "memory_dir", tmp_path)

    bridge._write_daily_memory("Test entry from bridge integration test")

    today = datetime.now().strftime("%Y-%m-%d")
    memory_file = tmp_path / f"{today}.md"
    assert memory_file.exists(), "Memory file not created"
    assert "Test entry from bridge integration test" in memory_file.read_text()


@requires_openclaw
def test_session_summary(require_gateway):
    """Test 5: Print bridge session summary."""
    bridge = OpenClawBridge(game="FrozenLake_4x4")
    bridge.query("Session summary test — acknowledge with one word.")
    summary = bridge.get_summary()
    assert "OpenClaw Bridge" in summary
    assert bridge.total_sent >= 1
