"""
MetaLayer — Abstract Base Class for all Meta^N Layers

Every layer in the fractal stack inherits from MetaLayer.
Each layer has three sub-levels of self-optimization:
  1. Weight-level: Individual parameter tuning
  2. Synapse-level: Connection structure optimization
  3. Neuron-level: Unit activation/creation/deletion

The accept/reject protocol allows each layer to evaluate
suggestions from other layers before applying them.

Key methods:
  - optimize(context)           → Run one optimization step
  - signal(direction, data)     → Send signal to other layers
  - receive(signal)             → Process incoming signal
  - accept_reject(signal)       → Evaluate and accept/reject suggestion
  - snapshot()                  → Return holographic state snapshot
  - self_optimize()             → Internal self-optimization across all sub-levels
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time

from throng3.core.signal import (
    Signal, SignalBundle, SignalDirection, SignalType, SignalPriority
)


@dataclass
class LayerMetrics:
    """Performance metrics tracked by each layer."""
    loss: float = float('inf')
    accuracy: float = 0.0
    complexity: float = 0.0          # Structural complexity (params, connections)
    efficiency: float = 0.0          # Performance / complexity ratio
    stability: float = 1.0           # How stable over recent history
    improvement_rate: float = 0.0    # Rate of improvement
    n_parameters: int = 0
    n_active_connections: int = 0
    step_count: int = 0
    last_update: float = field(default_factory=time.time)
    
    def update(self, loss: float, accuracy: float):
        """Update metrics with new measurement."""
        # Compute improvement rate (exponential moving average)
        if self.loss != float('inf'):
            improvement = (self.loss - loss) / max(abs(self.loss), 1e-8)
            self.improvement_rate = 0.9 * self.improvement_rate + 0.1 * improvement
        
        # Compute stability (low variance = high stability)
        if self.loss != float('inf'):
            change = abs(loss - self.loss) / max(abs(self.loss), 1e-8)
            self.stability = 0.95 * self.stability + 0.05 * (1.0 - min(change, 1.0))
        
        self.loss = loss
        self.accuracy = accuracy
        self.efficiency = accuracy / max(self.complexity, 1e-8)
        self.step_count += 1
        self.last_update = time.time()


@dataclass
class AcceptRejectDecision:
    """Result of evaluating a signal for accept/reject."""
    accepted: bool
    confidence: float                # 0-1 how confident in decision
    reason: str = ""
    counter_proposal: Optional[Dict] = None  # If rejected, alternative suggestion
    

class MetaLayer(ABC):
    """
    Abstract base class for all Meta^N layers.
    
    Each layer is a self-contained optimization unit that:
    1. Maintains its own state and metrics
    2. Optimizes at weight, synapse, and neuron levels
    3. Sends/receives signals to/from other layers
    4. Can accept or reject suggestions from other layers
    5. Provides holographic state snapshots
    """
    
    def __init__(self, level: int, name: str, config: Optional[Dict] = None):
        """
        Initialize a MetaLayer.
        
        Args:
            level: Meta-level in the hierarchy (0=substrate, 1=synapse, etc.)
            name: Human-readable name
            config: Layer-specific configuration
        """
        self.level = level
        self.name = name
        self.config = config or {}
        
        # Metrics tracking
        self.metrics = LayerMetrics()
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Signal management
        self.inbox: deque = deque(maxlen=1000)      # Incoming signals
        self.outbox: List[Signal] = []               # Outgoing signals
        self.pending_responses: Dict[str, Signal] = {}  # Awaiting accept/reject
        
        # Self-optimization state
        self._optimization_step = 0
        self._learning_rate = self.config.get('learning_rate', 0.01)
        self._momentum = self.config.get('momentum', 0.9)
        
        # Accept/reject thresholds
        self._accept_threshold = self.config.get('accept_threshold', 0.5)
        self._reject_cost_limit = self.config.get('reject_cost_limit', 0.1)
        
        # Holographic encoding
        self._state_hash = 0
        self._holographic_dim = self.config.get('holographic_dim', 64)
    
    # ================================================================
    # ABSTRACT METHODS — Must be implemented by each concrete layer
    # ================================================================
    
    @abstractmethod
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one optimization step.
        
        Args:
            context: Information from the environment and other layers
            
        Returns:
            Results dict with at minimum 'loss' and 'metrics'
        """
        pass
    
    @abstractmethod
    def _compute_state_vector(self) -> np.ndarray:
        """
        Compute a compressed state vector for holographic encoding.
        
        Returns:
            1D numpy array representing the layer's current state
        """
        pass
    
    @abstractmethod
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """
        Apply a suggestion from another layer.
        
        Args:
            suggestion: Dict with suggested changes
            
        Returns:
            True if successfully applied
        """
        pass
    
    @abstractmethod
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluate a suggestion without applying it.
        
        Args:
            suggestion: Dict with suggested changes
            
        Returns:
            (score, reason) where score is 0-1 (1 = definitely accept)
        """
        pass
    
    @abstractmethod
    def _self_optimize_weights(self):
        """Optimize at the weight/parameter level."""
        pass
    
    @abstractmethod
    def _self_optimize_synapses(self):
        """Optimize at the connection/synapse level."""
        pass
    
    @abstractmethod
    def _self_optimize_neurons(self):
        """Optimize at the unit/neuron level (add/remove units)."""
        pass
    
    # ================================================================
    # SIGNAL METHODS — Sending and receiving
    # ================================================================
    
    def signal(self, direction: SignalDirection, signal_type: SignalType,
               payload: Dict[str, Any], target_level: Optional[int] = None,
               priority: SignalPriority = SignalPriority.NORMAL,
               requires_response: bool = False) -> Signal:
        """
        Create and queue a signal to send to other layers.
        
        Args:
            direction: UP, DOWN, LATERAL, or BROADCAST
            signal_type: Type of signal
            payload: Data to send
            target_level: Specific target (None = auto-route)
            priority: Signal priority
            requires_response: Whether to expect ACCEPT/REJECT
            
        Returns:
            The created Signal object
        """
        sig = Signal(
            source_level=self.level,
            direction=direction,
            signal_type=signal_type,
            payload=payload,
            target_level=target_level,
            priority=priority,
            requires_response=requires_response,
        )
        self.outbox.append(sig)
        
        if requires_response:
            self.pending_responses[sig.signal_id] = sig
        
        return sig
    
    def receive(self, signal: Signal) -> Optional[Signal]:
        """
        Process an incoming signal.
        
        For SUGGESTION signals, runs accept/reject protocol.
        For ACCEPT/REJECT signals, resolves pending responses.
        For other signals, adds to inbox for processing.
        
        Args:
            signal: Incoming signal
            
        Returns:
            Response signal if one was generated, else None
        """
        # Handle responses to our pending signals
        if signal.parent_id and signal.parent_id in self.pending_responses:
            del self.pending_responses[signal.parent_id]
            self.inbox.append(signal)
            return None
        
        # Handle suggestions with accept/reject
        if signal.signal_type == SignalType.SUGGESTION and signal.requires_response:
            decision = self.accept_reject(signal)
            response = signal.create_response(
                accepted=decision.accepted,
                reason=decision.reason,
                counter_payload=decision.counter_proposal,
            )
            
            if decision.accepted:
                self._apply_suggestion(signal.payload)
            
            self.outbox.append(response)
            return response
        
        # Handle negotiation
        if signal.signal_type == SignalType.NEGOTIATE:
            counter = signal.payload.get("counter", {})
            decision = self.accept_reject(signal)
            response = signal.create_response(
                accepted=decision.accepted,
                reason=decision.reason,
            )
            if decision.accepted:
                self._apply_suggestion(counter)
            self.outbox.append(response)
            return response
        
        # All other signals: queue for processing
        self.inbox.append(signal)
        return None
    
    def accept_reject(self, signal: Signal) -> AcceptRejectDecision:
        """
        Evaluate a signal and decide whether to accept or reject.
        
        Uses the layer's internal _evaluate_suggestion to score the proposal,
        then applies threshold logic.
        
        Args:
            signal: Signal to evaluate
            
        Returns:
            AcceptRejectDecision
        """
        suggestion = signal.payload
        score, reason = self._evaluate_suggestion(suggestion)
        
        # Accept if score above threshold
        if score >= self._accept_threshold:
            return AcceptRejectDecision(
                accepted=True,
                confidence=score,
                reason=f"Accepted (score={score:.3f}): {reason}",
            )
        
        # Check if we can negotiate
        if score >= self._accept_threshold * 0.5:
            # Partial acceptance — try to negotiate
            counter = self._generate_counter_proposal(suggestion, score)
            return AcceptRejectDecision(
                accepted=False,
                confidence=score,
                reason=f"Negotiate (score={score:.3f}): {reason}",
                counter_proposal=counter,
            )
        
        # Reject
        return AcceptRejectDecision(
            accepted=False,
            confidence=1.0 - score,
            reason=f"Rejected (score={score:.3f}): {reason}",
        )
    
    def _generate_counter_proposal(self, suggestion: Dict, score: float) -> Dict:
        """
        Generate a counter-proposal when partially accepting.
        
        Default: scale down the suggested changes proportionally.
        Override in subclasses for smarter negotiation.
        """
        counter = {}
        scale = score / self._accept_threshold  # 0.5 - 1.0
        
        for key, value in suggestion.items():
            if isinstance(value, (int, float)):
                counter[key] = value * scale
            elif isinstance(value, np.ndarray):
                counter[key] = value * scale
            else:
                counter[key] = value
        
        return counter
    
    def drain_outbox(self) -> List[Signal]:
        """Remove and return all pending outgoing signals."""
        signals = self.outbox.copy()
        self.outbox.clear()
        return signals
    
    def process_inbox(self):
        """
        Process all queued inbox signals.
        
        Called during optimize() to incorporate received information.
        """
        while self.inbox:
            signal = self.inbox.popleft()
            self._process_signal(signal)
    
    def _process_signal(self, signal: Signal):
        """
        Process a single non-suggestion signal.
        
        Override in subclasses for signal-type-specific handling.
        """
        if signal.signal_type == SignalType.REWARD:
            self._handle_reward(signal)
        elif signal.signal_type == SignalType.PREDICTION_ERROR:
            self._handle_prediction_error(signal)
        elif signal.signal_type == SignalType.STATE_UPDATE:
            self._handle_state_update(signal)
        elif signal.signal_type == SignalType.SNAPSHOT_REQUEST:
            self._handle_snapshot_request(signal)
        elif signal.signal_type == SignalType.PERFORMANCE:
            self._handle_performance_update(signal)
    
    def _handle_reward(self, signal: Signal):
        """Handle reward signal. Override in subclasses."""
        pass
    
    def _handle_prediction_error(self, signal: Signal):
        """Handle prediction error signal. Override in subclasses."""
        pass
    
    def _handle_state_update(self, signal: Signal):
        """Handle state update from another layer. Override in subclasses."""
        pass
    
    def _handle_snapshot_request(self, signal: Signal):
        """Handle snapshot request by sending our state."""
        snapshot = self.snapshot()
        self.signal(
            direction=SignalDirection.UP if signal.source_level > self.level else SignalDirection.DOWN,
            signal_type=SignalType.SNAPSHOT_RESPONSE,
            payload={"snapshot": snapshot, "level": self.level},
            target_level=signal.source_level,
        )
    
    def _handle_performance_update(self, signal: Signal):
        """Handle performance info from another layer. Override in subclasses."""
        pass
    
    # ================================================================
    # SELF-OPTIMIZATION — Three levels
    # ================================================================
    
    def self_optimize(self):
        """
        Run self-optimization across all three sub-levels.
        
        Order: weights → synapses → neurons
        (fine-grained before coarse-grained)
        """
        self._self_optimize_weights()
        self._self_optimize_synapses()
        self._self_optimize_neurons()
        self._optimization_step += 1
    
    # ================================================================
    # HOLOGRAPHIC STATE — Any slice contains info about the whole
    # ================================================================
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Create a holographic state snapshot.
        
        The snapshot contains enough information to reconstruct
        the essential behavior of this layer. Used for:
        - Cross-scale communication
        - Redundancy/recovery
        - Higher-level reasoning about the system
        
        Returns:
            Dict with state information
        """
        state_vector = self._compute_state_vector()
        
        # Holographic projection: compress to fixed-size representation
        # using random projection (preserves distances approximately)
        if len(state_vector) > self._holographic_dim:
            # Use deterministic projection seeded by level
            rng = np.random.RandomState(self.level * 42 + 7)
            projection = rng.randn(self._holographic_dim, len(state_vector))
            projection /= np.sqrt(self._holographic_dim)
            holographic = projection @ state_vector
        else:
            holographic = state_vector
        
        return {
            "level": self.level,
            "name": self.name,
            "metrics": {
                "loss": self.metrics.loss,
                "accuracy": self.metrics.accuracy,
                "efficiency": self.metrics.efficiency,
                "stability": self.metrics.stability,
                "improvement_rate": self.metrics.improvement_rate,
                "step_count": self.metrics.step_count,
                "n_parameters": self.metrics.n_parameters,
                "n_active_connections": self.metrics.n_active_connections,
            },
            "state_vector": holographic,
            "config": self.config.copy(),
            "optimization_step": self._optimization_step,
            "timestamp": time.time(),
        }
    
    def restore_from_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore layer state from a snapshot.
        
        Override in subclasses for full restoration.
        Base implementation restores metrics and config.
        """
        if "metrics" in snapshot:
            m = snapshot["metrics"]
            self.metrics.loss = m.get("loss", float('inf'))
            self.metrics.accuracy = m.get("accuracy", 0.0)
            self.metrics.efficiency = m.get("efficiency", 0.0)
            self.metrics.stability = m.get("stability", 1.0)
            self.metrics.step_count = m.get("step_count", 0)
        if "config" in snapshot:
            self.config.update(snapshot["config"])
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get layer status summary."""
        return {
            "level": self.level,
            "name": self.name,
            "metrics": {
                "loss": self.metrics.loss,
                "accuracy": self.metrics.accuracy,
                "efficiency": self.metrics.efficiency,
                "stability": self.metrics.stability,
            },
            "optimization_step": self._optimization_step,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
            "pending_responses": len(self.pending_responses),
        }
    
    def __repr__(self) -> str:
        return (f"MetaLayer(level={self.level}, name='{self.name}', "
                f"step={self._optimization_step}, "
                f"loss={self.metrics.loss:.4f})")
