"""
Live integration test for OpenClaw Bridge.
Tests real-time messaging to Tetra via the gateway.

Run with gateway live: python test_openclaw_bridge.py
"""
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.llm_policy.openclaw_bridge import OpenClawBridge


def test_gateway_health():
    """Test 1: Can we reach the gateway?"""
    print("=" * 60)
    print("TEST 1: Gateway Health Check")
    print("=" * 60)
    
    bridge = OpenClawBridge(game="test")
    ok = bridge.check_gateway()
    print(f"  Gateway alive: {ok}")
    assert ok, "Gateway is not running!"
    print("  ✅ PASS\n")
    return True


def test_ping():
    """Test 2: Send a ping, get a response."""
    print("=" * 60)
    print("TEST 2: Ping (simple message round-trip)")
    print("=" * 60)
    
    bridge = OpenClawBridge(game="test")
    response = bridge.query("BRIDGE_TEST: Reply with exactly BRIDGE_OK to confirm connectivity.")
    
    print(f"  Success: {response.success}")
    print(f"  Response: {response.raw[:300]}")
    
    assert response.success, f"Query failed: {response.error}"
    assert len(response.raw) > 0, "Empty response"
    print("  ✅ PASS\n")
    return True


def test_observation():
    """Test 3: Send a game observation, get hypothesis back."""
    print("=" * 60)
    print("TEST 3: Real-time Observation")
    print("=" * 60)
    
    bridge = OpenClawBridge(game="FrozenLake_4x4")
    response = bridge.send_observation(
        episode=1,
        observation="Action 2 (RIGHT) from state [1,1] leads to state [1,2] deterministically. "
                    "But action 2 from state [2,3] sometimes leads to [2,3] (same position) or [3,3]. "
                    "Success rate: 60% across 10 trials.",
        context={
            "state_dims": [4, 4],
            "action": 2,
            "action_name": "RIGHT",
            "success_rate": 0.6,
            "n_trials": 10,
            "from_state": [2, 3],
            "expected_state": [2, 4],
            "actual_states": {"[2,4]": 6, "[2,3]": 3, "[3,3]": 1}
        }
    )
    
    print(f"  Success: {response.success}")
    print(f"  Hypotheses: {len(response.hypotheses)}")
    print(f"  Response preview: {response.raw[:400]}")
    
    assert response.success, f"Observation failed: {response.error}"
    print("  ✅ PASS\n")
    return True


def test_memory_write():
    """Test 4: Check that memory files get written."""
    print("=" * 60)
    print("TEST 4: Memory File Write")
    print("=" * 60)
    
    from datetime import datetime
    from pathlib import Path
    
    bridge = OpenClawBridge(game="TestGame")
    
    # Write a daily memory entry
    bridge._write_daily_memory("Test entry from bridge integration test")
    
    today = datetime.now().strftime("%Y-%m-%d")
    memory_file = bridge.memory_dir / f"{today}.md"
    
    print(f"  Memory file: {memory_file}")
    print(f"  Exists: {memory_file.exists()}")
    
    assert memory_file.exists(), "Memory file not created"
    
    content = memory_file.read_text()
    assert "Test entry from bridge integration test" in content
    print(f"  Content size: {len(content)} chars")
    print("  ✅ PASS\n")
    return True


def test_session_summary():
    """Test 5: Print bridge session summary."""
    print("=" * 60)
    print("TEST 5: Session Summary")
    print("=" * 60)
    
    bridge = OpenClawBridge(game="FrozenLake_4x4")
    # Do a quick query to populate stats
    bridge.query("Session summary test — acknowledge with one word.")
    
    summary = bridge.get_summary()
    print(summary)
    
    assert "OpenClaw Bridge" in summary
    assert bridge.total_sent >= 1
    print("  ✅ PASS\n")
    return True


if __name__ == "__main__":
    print("\n🧩 OpenClaw Bridge Live Integration Tests\n")
    
    results = {}
    tests = [
        ("Gateway Health", test_gateway_health),
        ("Ping", test_ping),
        ("Observation", test_observation),
        ("Memory Write", test_memory_write),
        ("Session Summary", test_session_summary),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ❌ FAIL: {e}\n")
            results[name] = False
    
    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🧩 Bridge is LIVE! Ready for real-time observation flow.\n")
    else:
        print("\n  ⚠️ Some tests failed. Check gateway status.\n")
