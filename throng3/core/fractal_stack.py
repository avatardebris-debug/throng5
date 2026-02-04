"""
FractalStack — Composes MetaLayers into a Recursive Hierarchy

The FractalStack:
1. Manages the collection of MetaLayers (Meta^0 through Meta^N)
2. Routes signals between layers (UP, DOWN, LATERAL, BROADCAST)
3. Maintains the global holographic state
4. Orchestrates optimization across all layers
5. Handles the accept/reject negotiation protocol

Key design: Every layer can talk to every other layer, but signals
are routed through the stack to maintain ordering and priority.
"""

from typing import Dict, List, Any, Optional, Type
from collections import defaultdict, deque
import numpy as np
import time
import logging

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import (
    Signal, SignalBundle, SignalDirection, SignalType, SignalPriority
)
from throng3.core.holographic import HolographicState

logger = logging.getLogger(__name__)


class FractalStack:
    """
    The fractal stack composes MetaLayer instances into a meta^N hierarchy.
    
    Manages signal routing, holographic state, and orchestrated optimization.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize an empty FractalStack.
        
        Args:
            config: Stack-level configuration
        """
        self.config = config or {}
        
        # Layer registry (level -> MetaLayer)
        self.layers: Dict[int, MetaLayer] = {}
        
        # Signal routing
        self._signal_queue: deque = deque(maxlen=10000)
        self._signal_log: deque = deque(maxlen=5000)
        self._route_count = 0
        
        # Holographic state
        holographic_dim = self.config.get('holographic_dim', 128)
        self.holographic = HolographicState(dim=holographic_dim)
        
        # Optimization state
        self._step = 0
        self._optimization_order = self.config.get(
            'optimization_order', 'bottom_up'  # or 'top_down', 'parallel', 'adaptive'
        )
        
        # Performance tracking
        self._step_times: deque = deque(maxlen=100)
        self._signal_stats: Dict[str, int] = defaultdict(int)
    
    # ================================================================
    # LAYER MANAGEMENT
    # ================================================================
    
    def add_layer(self, layer: MetaLayer) -> 'FractalStack':
        """
        Add a MetaLayer to the stack.
        
        Args:
            layer: MetaLayer instance to add
            
        Returns:
            self (for chaining)
        """
        if layer.level in self.layers:
            raise ValueError(f"Layer at level {layer.level} already exists: "
                           f"{self.layers[layer.level].name}")
        
        self.layers[layer.level] = layer
        logger.info(f"Added layer: Meta^{layer.level} ({layer.name})")
        
        # Update holographic state capacity
        self.holographic.n_layers = max(self.holographic.n_layers, len(self.layers))
        
        return self
    
    def remove_layer(self, level: int) -> Optional[MetaLayer]:
        """Remove a layer from the stack."""
        return self.layers.pop(level, None)
    
    def get_layer(self, level: int) -> Optional[MetaLayer]:
        """Get a layer by its meta-level."""
        return self.layers.get(level)
    
    @property
    def n_layers(self) -> int:
        """Number of layers in the stack."""
        return len(self.layers)
    
    @property
    def levels(self) -> List[int]:
        """Sorted list of active meta-levels."""
        return sorted(self.layers.keys())
    
    # ================================================================
    # SIGNAL ROUTING
    # ================================================================
    
    def route_signals(self):
        """
        Route all pending signals between layers.
        
        This is the heart of cross-scale communication.
        Each layer's outbox is drained and signals are delivered
        to appropriate targets.
        """
        # Collect all outgoing signals from all layers
        all_signals = []
        for level, layer in self.layers.items():
            signals = layer.drain_outbox()
            all_signals.extend(signals)
        
        # Sort by priority (highest first)
        all_signals.sort(key=lambda s: s.priority.value, reverse=True)
        
        # Route each signal
        for signal in all_signals:
            if not signal.is_alive:
                continue
            
            targets = self._resolve_targets(signal)
            
            for target_level in targets:
                if target_level in self.layers:
                    # Deliver signal with decremented TTL
                    routed = signal.decrement_ttl()
                    response = self.layers[target_level].receive(routed)
                    
                    self._signal_stats[signal.signal_type.name] += 1
                    self._route_count += 1
                    self._signal_log.append({
                        "id": signal.signal_id,
                        "type": signal.signal_type.name,
                        "from": signal.source_level,
                        "to": target_level,
                        "direction": signal.direction.name,
                        "time": time.time(),
                    })
                    
                    # If target generated a response, queue it
                    if response:
                        self._signal_queue.append(response)
        
        # Process any response signals generated during routing
        while self._signal_queue:
            signal = self._signal_queue.popleft()
            if signal.target_level is not None and signal.target_level in self.layers:
                self.layers[signal.target_level].receive(signal)
    
    def _resolve_targets(self, signal: Signal) -> List[int]:
        """
        Determine which layers should receive a signal.
        
        Routing rules:
        - BROADCAST: all layers except source
        - UP: next higher level(s) from source
        - DOWN: next lower level(s) from source
        - LATERAL: same level (if multiple instances exist, or neighbors)
        - Specific target_level: just that layer
        """
        if signal.target_level is not None:
            return [signal.target_level]
        
        source = signal.source_level
        levels = self.levels
        
        if signal.direction == SignalDirection.BROADCAST:
            return [l for l in levels if l != source]
        
        elif signal.direction == SignalDirection.UP:
            # Send to next higher level
            higher = [l for l in levels if l > source]
            if higher:
                return [higher[0]]  # Immediate next level up
            return []
        
        elif signal.direction == SignalDirection.DOWN:
            # Send to next lower level
            lower = [l for l in levels if l < source]
            if lower:
                return [lower[-1]]  # Immediate next level down
            return []
        
        elif signal.direction == SignalDirection.LATERAL:
            # Same level neighbors (±1 for now)
            return [l for l in levels if l != source and abs(l - source) <= 1]
        
        return []
    
    def inject_signal(self, signal: Signal):
        """
        Inject an external signal into the stack.
        
        Used for external stimuli, rewards, LLM suggestions, etc.
        """
        targets = self._resolve_targets(signal)
        for target_level in targets:
            if target_level in self.layers:
                self.layers[target_level].receive(signal)
    
    # ================================================================
    # OPTIMIZATION ORCHESTRATION
    # ================================================================
    
    def step(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run one full optimization step across all layers.
        
        1. Process inbox signals in each layer
        2. Optimize each layer (in configured order)
        3. Collect outgoing signals
        4. Route signals between layers
        5. Update holographic state
        
        Args:
            context: External context (environment state, etc.)
            
        Returns:
            Step results with per-layer metrics
        """
        t0 = time.time()
        context = context or {}
        results = {}
        
        # Step 1: Process inbox for each layer
        for level in self.levels:
            self.layers[level].process_inbox()
        
        # Step 2: Optimize layers in configured order
        optimization_order = self._get_optimization_order()
        
        for level in optimization_order:
            layer = self.layers[level]
            
            # Enrich context with holographic view
            layer_context = {
                **context,
                "holographic_view": self.holographic.query(level),
                "system_summary": self.holographic.get_system_summary(),
                "step": self._step,
                "stack_levels": self.levels,
            }
            
            # Optimize
            result = layer.optimize(layer_context)
            results[level] = result
            
            # Self-optimize
            layer.self_optimize()
        
        # Step 3 & 4: Route signals
        self.route_signals()
        
        # Step 5: Update holographic state
        self._update_holographic()
        
        # Track timing
        dt = time.time() - t0
        self._step_times.append(dt)
        self._step += 1
        
        return {
            "step": self._step,
            "layer_results": results,
            "holographic": self.holographic.get_system_summary(),
            "signals_routed": self._route_count,
            "step_time": dt,
        }
    
    def _get_optimization_order(self) -> List[int]:
        """Get layer optimization order based on configuration."""
        levels = self.levels
        
        if self._optimization_order == 'bottom_up':
            return levels
        elif self._optimization_order == 'top_down':
            return list(reversed(levels))
        elif self._optimization_order == 'parallel':
            return levels  # Same as bottom_up but conceptually parallel
        elif self._optimization_order == 'adaptive':
            # Optimize layers with worst metrics first
            return sorted(levels, key=lambda l: self.layers[l].metrics.loss, reverse=True)
        else:
            return levels
    
    def _update_holographic(self):
        """Update the global holographic state from all layers."""
        for level, layer in self.layers.items():
            snapshot = layer.snapshot()
            state_vector = snapshot.get("state_vector", np.zeros(self.holographic.dim))
            if isinstance(state_vector, np.ndarray):
                self.holographic.update_layer(level, state_vector)
    
    # ================================================================
    # HOLOGRAPHIC QUERIES
    # ================================================================
    
    def get_holographic_view(self, from_level: int) -> np.ndarray:
        """Get the holographic state from a specific layer's perspective."""
        return self.holographic.query(from_level)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get complete system state summary."""
        return {
            "step": self._step,
            "n_layers": self.n_layers,
            "layers": {
                level: layer.get_status()
                for level, layer in self.layers.items()
            },
            "holographic": self.holographic.get_system_summary(),
            "signal_stats": dict(self._signal_stats),
            "total_signals_routed": self._route_count,
            "avg_step_time": (
                np.mean(list(self._step_times)) if self._step_times else 0
            ),
        }
    
    def get_all_snapshots(self) -> Dict[int, Dict[str, Any]]:
        """Get holographic snapshots from all layers."""
        return {
            level: layer.snapshot()
            for level, layer in self.layers.items()
        }
    
    # ================================================================
    # SAVE/LOAD
    # ================================================================
    
    def save_state(self) -> Dict[str, Any]:
        """Serialize the full stack state."""
        return {
            "config": self.config,
            "step": self._step,
            "layer_snapshots": self.get_all_snapshots(),
            "holographic": self.holographic.save(),
            "signal_stats": dict(self._signal_stats),
        }
    
    def __repr__(self) -> str:
        layers_str = ", ".join(
            f"Meta^{l}({self.layers[l].name})" for l in self.levels
        )
        return f"FractalStack(step={self._step}, layers=[{layers_str}])"
