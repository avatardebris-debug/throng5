"""
Signal Protocol for Meta^N Cross-Scale Communication

Signals carry information between MetaLayers:
- UP: From lower meta-level to higher (data → abstraction)
- DOWN: From higher meta-level to lower (guidance → implementation)
- LATERAL: Between same meta-level (coordination)

Each signal includes:
- source_level: Which meta-layer sent it
- target_level: Which meta-layer should receive it (or None for broadcast)
- signal_type: What kind of information
- payload: The actual data
- priority: How urgent
- requires_response: Whether sender expects accept/reject
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import time
import uuid
import numpy as np


class SignalDirection(Enum):
    """Direction of signal flow in the meta-hierarchy."""
    UP = auto()       # Lower → Higher (data flows up)
    DOWN = auto()     # Higher → Lower (guidance flows down)
    LATERAL = auto()  # Same level (coordination)
    BROADCAST = auto()  # To all layers


class SignalType(Enum):
    """Types of information carried by signals."""
    # Data signals
    STATE_UPDATE = auto()       # Current state snapshot
    GRADIENT = auto()           # Error/loss gradient information
    PERFORMANCE = auto()        # Performance metrics
    
    # Control signals
    SUGGESTION = auto()         # Suggested parameter change
    COMMAND = auto()            # Direct parameter override
    QUERY = auto()              # Request for information
    
    # Meta signals
    ACCEPT = auto()             # Accept a suggestion
    REJECT = auto()             # Reject a suggestion (with reason)
    NEGOTIATE = auto()          # Counter-proposal
    
    # Learning signals
    REWARD = auto()             # Reward/punishment signal
    PREDICTION_ERROR = auto()   # Prediction error (surprise)
    ELIGIBILITY = auto()        # Eligibility trace update
    
    # Structural signals
    GROW = auto()               # Request to add capacity
    PRUNE = auto()              # Request to remove capacity
    RESTRUCTURE = auto()        # Request architectural change
    
    # Holographic signals
    SNAPSHOT_REQUEST = auto()   # Request state snapshot
    SNAPSHOT_RESPONSE = auto()  # State snapshot data


class SignalPriority(Enum):
    """Priority levels for signal processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Signal:
    """
    A signal between MetaLayers in the fractal stack.
    
    Immutable message passing with accept/reject protocol.
    """
    source_level: int                          # Meta-level of sender
    direction: SignalDirection                  # UP, DOWN, LATERAL, BROADCAST
    signal_type: SignalType                     # What kind of signal
    payload: Dict[str, Any] = field(default_factory=dict)  # Data
    target_level: Optional[int] = None         # Specific target (None = auto-route)
    priority: SignalPriority = SignalPriority.NORMAL
    requires_response: bool = False            # Expects ACCEPT/REJECT back
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None            # For response tracking
    ttl: int = 10                              # Time-to-live (max hops)
    
    def create_response(self, accepted: bool, reason: str = "",
                       counter_payload: Optional[Dict] = None) -> 'Signal':
        """Create an ACCEPT or REJECT response to this signal."""
        return Signal(
            source_level=self.target_level if self.target_level is not None else -1,
            direction=SignalDirection.UP if self.direction == SignalDirection.DOWN else SignalDirection.DOWN,
            signal_type=SignalType.ACCEPT if accepted else SignalType.REJECT,
            payload={
                "reason": reason,
                "original_type": self.signal_type.name,
                "counter": counter_payload or {},
            },
            target_level=self.source_level,
            priority=self.priority,
            requires_response=False,
            parent_id=self.signal_id,
        )
    
    def create_negotiation(self, counter_payload: Dict) -> 'Signal':
        """Create a NEGOTIATE counter-proposal."""
        return Signal(
            source_level=self.target_level if self.target_level is not None else -1,
            direction=SignalDirection.UP if self.direction == SignalDirection.DOWN else SignalDirection.DOWN,
            signal_type=SignalType.NEGOTIATE,
            payload={
                "original_type": self.signal_type.name,
                "counter": counter_payload,
            },
            target_level=self.source_level,
            priority=self.priority,
            requires_response=True,
            parent_id=self.signal_id,
        )
    
    def decrement_ttl(self) -> 'Signal':
        """Return copy with decremented TTL."""
        return Signal(
            source_level=self.source_level,
            direction=self.direction,
            signal_type=self.signal_type,
            payload=self.payload,
            target_level=self.target_level,
            priority=self.priority,
            requires_response=self.requires_response,
            signal_id=self.signal_id,
            timestamp=self.timestamp,
            parent_id=self.parent_id,
            ttl=self.ttl - 1,
        )
    
    @property
    def is_alive(self) -> bool:
        """Check if signal still has TTL remaining."""
        return self.ttl > 0
    
    def __repr__(self) -> str:
        return (f"Signal({self.signal_id}: {self.signal_type.name} "
                f"{self.direction.name} L{self.source_level}→"
                f"{'L'+str(self.target_level) if self.target_level is not None else '*'} "
                f"p={self.priority.name})")


@dataclass
class SignalBundle:
    """
    A collection of signals to be processed together.
    
    Used for batch processing and holographic state assembly.
    """
    signals: list = field(default_factory=list)
    
    def add(self, signal: Signal):
        """Add a signal to the bundle."""
        self.signals.append(signal)
    
    def by_type(self, signal_type: SignalType) -> list:
        """Filter signals by type."""
        return [s for s in self.signals if s.signal_type == signal_type]
    
    def by_direction(self, direction: SignalDirection) -> list:
        """Filter signals by direction."""
        return [s for s in self.signals if s.direction == direction]
    
    def by_priority(self, min_priority: SignalPriority = SignalPriority.NORMAL) -> list:
        """Filter signals by minimum priority."""
        return [s for s in self.signals if s.priority.value >= min_priority.value]
    
    def sorted_by_priority(self) -> list:
        """Return signals sorted by priority (highest first)."""
        return sorted(self.signals, key=lambda s: s.priority.value, reverse=True)
    
    def __len__(self) -> int:
        return len(self.signals)
    
    def __iter__(self):
        return iter(self.signals)
