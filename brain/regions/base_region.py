"""
base_region.py — Abstract base class for all brain regions.

Every brain region implements this interface so the message bus and
orchestrator can interact with them uniformly.

Inspired by throng35/regions/region_base.py and throng35/regions/region_protocol.py
(preserved in throng4_new/throng35/ for reference).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from brain.message_bus import BrainMessage, MessageBus, Priority


class BrainRegion(ABC):
    """
    Abstract brain region.

    Every region has:
    - A name (e.g., "sensory_cortex", "amygdala", "prefrontal")
    - A reference to the shared MessageBus
    - process() — handle one tick of input
    - learn() — update internal models from experience
    - report() — produce a status summary for telemetry
    """

    def __init__(self, name: str, bus: MessageBus):
        self.name = name
        self.bus = bus
        self.bus.register(name, callback=self._on_emergency)
        self._step_count = 0
        self._is_active = True

    # ── Core interface ────────────────────────────────────────────────

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one tick of input and return outputs.

        Args:
            inputs: Region-specific input data

        Returns:
            Region-specific output data (may include actions, predictions, etc.)
        """
        ...

    @abstractmethod
    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Update internal models from experience.

        Args:
            experience: Training data (transitions, replay, dream data)

        Returns:
            Training metrics dict
        """
        ...

    def report(self) -> Dict[str, Any]:
        """Produce a status summary for telemetry and cross-region communication."""
        return {
            "name": self.name,
            "step_count": self._step_count,
            "is_active": self._is_active,
        }

    # ── Messaging ─────────────────────────────────────────────────────

    def send(
        self,
        target: Optional[str],
        msg_type: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.ROUTINE,
    ) -> None:
        """Send a message to another region (or broadcast if target=None)."""
        msg = BrainMessage(
            source=self.name,
            target=target,
            priority=priority,
            msg_type=msg_type,
            payload=payload,
        )
        self.bus.send(msg)

    def receive(self, max_messages: int = 10) -> List[BrainMessage]:
        """Poll the inbox for messages addressed to this region."""
        return self.bus.poll(self.name, max_messages=max_messages)

    def broadcast(self, msg_type: str, payload: Dict[str, Any], priority: Priority = Priority.ROUTINE) -> None:
        """Broadcast a message to all other regions."""
        self.send(None, msg_type, payload, priority)

    # ── Emergency handling ────────────────────────────────────────────

    def _on_emergency(self, message: BrainMessage) -> None:
        """
        Handle an emergency message (e.g., amygdala HALT).

        Default: deactivate this region. Override for custom behavior.
        """
        if message.msg_type == "halt":
            self._is_active = False
        elif message.msg_type == "resume":
            self._is_active = True

    def halt(self) -> None:
        """Deactivate this region (called by amygdala override)."""
        self._is_active = False

    def resume(self) -> None:
        """Reactivate this region."""
        self._is_active = True
        self.bus.resume(self.name)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Called at the start of each episode. Override for state cleanup."""
        pass

    def step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified step: process inputs, increment counter, return outputs.

        Skips processing if region is halted (returns empty dict).
        """
        self._step_count += 1

        if not self._is_active:
            return {"halted": True, "region": self.name}

        return self.process(inputs)

    def __repr__(self) -> str:
        status = "active" if self._is_active else "HALTED"
        return f"{self.__class__.__name__}(name={self.name}, steps={self._step_count}, {status})"
