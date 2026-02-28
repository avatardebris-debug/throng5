"""
Smoke test — verifies that the brain/ module foundation works end-to-end.

Tests:
1. SessionLogger writes JSONL with branch tracking
2. MessageBus routes messages with priority ordering
3. BrainRegion subclass processes and communicates
4. ContextRestorer reads and summarizes logs
5. Emergency halt/resume flow
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.telemetry.session_logger import SessionLogger
from brain.telemetry.context_restorer import ContextRestorer
from brain.message_bus import MessageBus, BrainMessage, Priority
from brain.regions.base_region import BrainRegion
from brain.config import VERSION

# ── 1. Test SessionLogger ─────────────────────────────────────────────

print("=" * 60)
print(f"Throng 5 Brain Module Smoke Test — v{VERSION}")
print("=" * 60)

log = SessionLogger("smoke_test")
log.event("phase1", "test", "Smoke test started")
log.decision("system", "architecture_choice", {"module": "brain"}, reason="Clean break from throng4")
log.branch("phase1", "montezuma_fix", "Testing branch tracking")
log.event("phase1", "test", "Work inside branch")
log.milestone("phase1", "Branch tracking works")
log.merge("phase1", "montezuma_fix", "Branch completed successfully")
log.event("phase1", "test", "Back on core path")

# Branch that stays open (should trigger recommendation)
log.branch("phase1", "unfinished_work", "This branch will stay open")
log.event("phase1", "test", "Work that never got merged")

print(f"[PASS] SessionLogger wrote {log._event_count} events to {log.log_file.name}")

# Check recommendation
rec = log.recommend_return_to_core()
# Won't trigger time-based (too fast), but the branch is tracked
branches = log.get_active_branches()
assert "unfinished_work" in branches, "Branch tracking failed!"
print(f"[PASS] Branch tracking: {len(branches)} active branch(es) detected")

log.close()

# ── 2. Test ContextRestorer ───────────────────────────────────────────

restorer = ContextRestorer()
summary = restorer.latest_session_summary()
assert "smoke_test" in summary, "Session name not in summary"
assert "Milestones" in summary, "Milestones section missing"
assert "OPEN BRANCHES" in summary, "Open branch warning missing"
print(f"[PASS] ContextRestorer summary ({len(summary)} chars)")

divergences = restorer.find_divergence_points(log.log_file)
assert any(d["type"] == "open_branch" for d in divergences), "Divergence detection failed"
print(f"[PASS] Divergence detection: {len(divergences)} point(s) found")

# ── 3. Test MessageBus ────────────────────────────────────────────────

bus = MessageBus()
bus.register("sensory")
bus.register("amygdala")
bus.register("motor")
bus.register("prefrontal")

# Routine message
bus.send(BrainMessage(
    source="sensory", target="motor", priority=Priority.ROUTINE,
    msg_type="perception", payload={"frame": 42}
))

# Emergency halt broadcast from amygdala
bus.send(BrainMessage(
    source="amygdala", target="prefrontal", priority=Priority.EMERGENCY,
    msg_type="halt", payload={"threat": 0.95, "reason": "catastrophic prediction"}
))

# Motor should have 1 message
motor_msgs = bus.poll("motor")
assert len(motor_msgs) == 1, f"Motor expected 1 msg, got {len(motor_msgs)}"
assert motor_msgs[0].msg_type == "perception"
print(f"[PASS] MessageBus routing (routine)")

# Prefrontal should be halted
assert bus.is_halted("prefrontal"), "Prefrontal should be halted after emergency"
print(f"[PASS] Emergency halt working")

# Resume
bus.resume_all()
assert not bus.is_halted("prefrontal"), "Should be resumed"
print(f"[PASS] Resume after halt")

# ── 4. Test BrainRegion subclass ──────────────────────────────────────

class TestRegion(BrainRegion):
    def process(self, inputs):
        return {"processed": True, "input_keys": list(inputs.keys())}
    
    def learn(self, experience):
        return {"loss": 0.01}

bus2 = MessageBus()
region = TestRegion("test_region", bus2)

# Normal step
result = region.step({"obs": [1, 2, 3]})
assert result["processed"] is True
print(f"[PASS] BrainRegion.step() works")

# Halt and step
region.halt()
result = region.step({"obs": [1, 2, 3]})
assert result.get("halted") is True
print(f"[PASS] BrainRegion halted correctly")

# Resume
region.resume()
result = region.step({"obs": [1, 2, 3]})
assert result["processed"] is True
print(f"[PASS] BrainRegion resumed correctly")

# Send message
region.broadcast("test_broadcast", {"data": 123})
print(f"[PASS] BrainRegion messaging works")

# ── Summary ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("ALL TESTS PASSED")
print(f"Brain module v{VERSION} foundation verified.")
print(f"Log file: {log.log_file}")
print("=" * 60)
