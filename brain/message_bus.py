"""
message_bus.py — Cross-region communication for Brain architecture.

All brain regions communicate through BrainMessages routed by the MessageBus.

Message priority levels:
    0 = routine (background processing, replay data)
    1 = urgent (learner swap, mode change)
    2 = emergency (amygdala HALT — suspends slow path immediately)

Paths:
    Fast path (subconscious):  Sensory → Basal Ganglia → Motor Cortex
    Slow path (conscious):     Sensory → all regions → Prefrontal → Striatum
    Emergency override:        Amygdala broadcasts priority=2 → suspends slow path
    Overnight:                 Hippocampus replays → all regions learn
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional


class Priority(IntEnum):
    """Message priority levels."""
    ROUTINE = 0
    URGENT = 1
    EMERGENCY = 2  # Amygdala override — halt higher functions


# ────────────────────────────────────────────────────────────────────────────
# FastSlotBus — zero-allocation hot-path messaging
# ────────────────────────────────────────────────────────────────────────────

class FastSlotBus:
    """
    Zero-allocation inter-region signalling for the hot path.

    Instead of creating a BrainMessage dataclass + time.time() + payload dict
    on every send, regions write into pre-allocated slots (plain Python dicts
    reused across steps). Readers call read_slot() which returns the same dict
    object — no copy, no GC pressure.

    Emergency and broadcast messages still go through the full MessageBus.

    Usage:
        fast_bus = FastSlotBus()
        fast_bus.register_slot("amygdala", "threat_score", 0.0)
        fast_bus.write("amygdala", threat_score=0.7, operating_mode="execute")
        score = fast_bus.read("amygdala", "threat_score")  # 0.7, no alloc
    """

    __slots__ = ('_slots', '_halted')

    def __init__(self):
        # _slots[source_name] = dict of key->value (mutable, reused each step)
        self._slots: Dict[str, Dict[str, Any]] = {}
        self._halted: set = set()

    def register(self, region_name: str, defaults: Dict[str, Any]) -> None:
        """Register a region with its default slot values."""
        self._slots[region_name] = dict(defaults)

    def write(self, region_name: str, **kwargs) -> None:
        """
        Update a region's slot values in-place.

        Zero allocation when region is pre-registered: just updates existing dict.
        """
        slot = self._slots.get(region_name)
        if slot is None:
            self._slots[region_name] = kwargs
        else:
            slot.update(kwargs)

    def read(self, region_name: str, key: str, default: Any = None) -> Any:
        """Read a single value from a region's slot. No allocation."""
        slot = self._slots.get(region_name)
        if slot is None:
            return default
        return slot.get(key, default)

    def read_slot(self, region_name: str) -> Dict[str, Any]:
        """Return the entire slot dict for a region (same object, not a copy)."""
        return self._slots.get(region_name, {})

    def is_halted(self, region_name: str) -> bool:
        return region_name in self._halted

    def halt(self, region_name: str) -> None:
        self._halted.add(region_name)

    def resume(self, region_name: str) -> None:
        self._halted.discard(region_name)

    def resume_all(self) -> None:
        self._halted.clear()



@dataclass
class BrainMessage:
    """
    A message passed between brain regions.

    Attributes:
        source:     Region name that created this message
        target:     Target region name, or None for broadcast
        priority:   0=routine, 1=urgent, 2=emergency
        msg_type:   Semantic type (perception, threat, action_request, replay, strategy, halt, etc.)
        payload:    Region-specific data dict
        timestamp:  Auto-set to creation time
        msg_id:     Auto-incrementing ID for ordering
    """
    source: str
    target: Optional[str]
    priority: Priority
    msg_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    msg_id: int = 0

    # Class-level counter for unique IDs
    _next_id: int = 0

    def __post_init__(self):
        BrainMessage._next_id += 1
        self.msg_id = BrainMessage._next_id

    def is_broadcast(self) -> bool:
        return self.target is None

    def is_emergency(self) -> bool:
        return self.priority == Priority.EMERGENCY


class MessageBus:
    """
    Routes BrainMessages between brain regions.

    Each region registers with the bus and receives messages via callbacks
    or by polling its inbox.

    Features:
    - Priority ordering: EMERGENCY messages processed first
    - Broadcast: target=None delivers to all registered regions
    - Emergency halt: priority=2 clears non-emergency queues for target
    - Message history for telemetry/debugging
    """

    def __init__(self, history_size: int = 500, enable_history: bool = False):
        self._inboxes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._registered_regions: set = set()
        self._enable_history = enable_history
        self._history: deque = deque(maxlen=history_size) if enable_history else deque(maxlen=0)
        self._stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"sent": 0, "received": 0})
        self._halted_regions: set = set()  # Regions currently halted by amygdala

    def register(self, region_name: str, callback: Optional[Callable] = None) -> None:
        """Register a brain region with the bus."""
        self._registered_regions.add(region_name)
        if callback:
            self._callbacks[region_name].append(callback)

    def unregister(self, region_name: str) -> None:
        """Remove a region from the bus."""
        self._registered_regions.discard(region_name)
        self._callbacks.pop(region_name, None)
        self._inboxes.pop(region_name, None)

    def send(self, message: BrainMessage) -> None:
        """
        Send a message to the bus for routing.

        Emergency messages (priority=2) are processed immediately.
        Other messages are queued in the target's inbox.
        """
        if self._enable_history:
            self._history.append(message)
        self._stats[message.source]["sent"] += 1

        if message.is_broadcast():
            targets = self._registered_regions - {message.source}
        elif message.target in self._registered_regions:
            targets = {message.target}
        else:
            # Unknown target — drop silently
            return

        for target in targets:
            # Emergency override: if this IS an emergency halt, mark the target
            if message.is_emergency() and message.msg_type == "halt":
                self._halted_regions.add(target)

            # If target is halted and this isn't an emergency, queue but deprioritize
            if target in self._halted_regions and not message.is_emergency():
                if message.msg_type != "resume":
                    continue  # Drop non-emergency messages to halted regions

            self._inboxes[target].append(message)
            self._stats[target]["received"] += 1

            # Fire callbacks immediately for emergency messages
            if message.is_emergency():
                for cb in self._callbacks.get(target, []):
                    try:
                        cb(message)
                    except Exception:
                        pass  # Never crash on callback errors

    def resume(self, region_name: str) -> None:
        """Resume a halted region (called when amygdala clears the threat)."""
        self._halted_regions.discard(region_name)

    def resume_all(self) -> None:
        """Resume all halted regions."""
        self._halted_regions.clear()

    def poll(self, region_name: str, max_messages: int = 10) -> List[BrainMessage]:
        """
        Poll the inbox for a region. Returns up to max_messages, priority-sorted.

        Emergency messages are always returned first.
        """
        inbox = self._inboxes.get(region_name, deque())
        if not inbox:
            return []

        # Drain up to max_messages, prioritized
        messages = []
        remaining = deque()

        while inbox and len(messages) < max_messages:
            msg = inbox.popleft()
            messages.append(msg)

        # Sort by priority (highest first), then by timestamp
        messages.sort(key=lambda m: (-m.priority, m.timestamp))

        return messages

    def is_halted(self, region_name: str) -> bool:
        """Check if a region is currently halted by emergency override."""
        return region_name in self._halted_regions

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get message statistics per region."""
        return dict(self._stats)

    def get_recent_messages(self, n: int = 20) -> List[BrainMessage]:
        """Get the N most recent messages for debugging."""
        return list(self._history)[-n:]

    def __repr__(self) -> str:
        return (
            f"MessageBus(regions={len(self._registered_regions)}, "
            f"halted={self._halted_regions or 'none'}, "
            f"history={len(self._history)})"
        )
