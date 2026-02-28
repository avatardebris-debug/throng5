"""
Meta^N: LLM/Agent Interface — Reasoning About the Whole System

The highest meta-layer. Provides an interface for an external
LLM/agent to observe the entire fractal stack and make suggestions.

This is where human-like reasoning meets neural optimization:
- The LLM can read holographic state summaries
- The LLM can suggest architectural changes
- The LLM can set goals and priorities
- The LLM can explain what the system is doing (interpretability)

The key insight: the LLM doesn't need to understand every weight —
it reasons about the META-patterns (which learning rules are working,
whether exploration is sufficient, etc.)

API modes:
- Observer: Read-only view of system state
- Advisor: Can send suggestions (accept/reject protocol applies)
- Controller: Can override lower layers (emergency only)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import time

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class LLMConfig:
    """Configuration for LLM interface."""
    mode: str = 'advisor'            # 'observer', 'advisor', 'controller'
    observation_interval: int = 10   # Steps between observations
    suggestion_cooldown: int = 50    # Min steps between suggestions
    max_suggestions_per_step: int = 3
    auto_summarize: bool = True      # Generate human-readable summaries
    log_observations: bool = True
    callback: Optional[Callable] = None  # External LLM callback function


@dataclass
class Observation:
    """A system observation for the LLM."""
    timestamp: float
    step: int
    system_summary: Dict[str, Any]
    layer_states: Dict[int, Dict[str, Any]]
    holographic_coherence: float
    alerts: List[str]
    human_readable: str = ""


class LLMInterface(MetaLayer):
    """
    Meta^N: LLM-in-the-loop interface.
    
    Bridges the gap between neural optimization and symbolic reasoning.
    The LLM observes system state and can make high-level suggestions
    that propagate DOWN through the fractal stack.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None, **kwargs):
        cfg = config or LLMConfig()
        # Use level 99 to ensure it's always the highest
        super().__init__(level=99, name="LLMInterface", config=vars(cfg))
        self.llm_config = cfg
        
        # Observation log
        self.observations: deque = deque(maxlen=100)
        self.pending_observations: List[Observation] = []
        
        # Suggestion tracking
        self._suggestions_sent: deque = deque(maxlen=500)
        self._suggestions_accepted: int = 0
        self._suggestions_rejected: int = 0
        self._last_suggestion_step: int = -100
        
        # Alert system
        self._alerts: List[str] = []
        self._alert_thresholds = {
            'stagnation': 50,       # Steps without improvement
            'instability': 0.3,     # Max acceptable instability
            'low_coherence': 0.2,   # Min holographic coherence
            'exploration_collapse': 0.01,  # Min exploration rate
        }
        
        # External callback (for actual LLM integration)
        self._callback = cfg.callback
        
        # Internal reasoning state
        self._system_model: Dict[str, Any] = {}
        self._intervention_history: deque = deque(maxlen=100)
        self._stagnation_counter = 0
        self._last_best_loss = float('inf')
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe system state and optionally intervene.
        
        1. Collect observations from all layers
        2. Check for alerts (anomalies)
        3. Generate suggestions if needed
        4. Call external LLM if configured
        5. Send suggestions DOWN
        """
        self.process_inbox()
        
        system_summary = context.get('system_summary', {})
        holographic_view = context.get('holographic_view', np.zeros(10))
        
        # Build observation
        observation = self._build_observation(context)
        self.observations.append(observation)
        
        # Check alerts
        self._check_alerts(observation)
        
        # Update internal model
        self._update_system_model(observation)
        
        # Generate suggestions (if advisor/controller mode)
        suggestions = []
        if self.llm_config.mode in ('advisor', 'controller'):
            if (self._optimization_step - self._last_suggestion_step 
                >= self.llm_config.suggestion_cooldown):
                suggestions = self._generate_suggestions(observation)
                
                # Send suggestions DOWN
                for target_level, suggestion in suggestions[:self.llm_config.max_suggestions_per_step]:
                    self.signal(
                        direction=SignalDirection.DOWN,
                        signal_type=SignalType.SUGGESTION if self.llm_config.mode == 'advisor' 
                            else SignalType.COMMAND,
                        payload=suggestion,
                        target_level=target_level,
                        requires_response=True,
                        priority=SignalPriority.HIGH,
                    )
                    self._suggestions_sent.append({
                        'step': self._optimization_step,
                        'target': target_level,
                        'suggestion': suggestion,
                    })
                
                if suggestions:
                    self._last_suggestion_step = self._optimization_step
        
        # Call external LLM if configured
        if self._callback and self._optimization_step % self.llm_config.observation_interval == 0:
            try:
                llm_response = self._callback(observation)
                if llm_response:
                    self._process_llm_response(llm_response)
            except Exception:
                pass  # Don't let LLM errors crash the system
        
        # Metrics
        accept_rate = (
            self._suggestions_accepted / 
            max(self._suggestions_accepted + self._suggestions_rejected, 1)
        )
        self.metrics.update(1.0 - accept_rate, accept_rate)
        
        return {
            'observation': observation.human_readable if observation else "",
            'alerts': list(self._alerts),
            'suggestions_sent': len(suggestions),
            'accept_rate': accept_rate,
            'metrics': self.metrics,
        }
    
    def _build_observation(self, context: Dict) -> Observation:
        """Build a structured observation of the system."""
        system_summary = context.get('system_summary', {})
        layer_results = context.get('layer_results', {})
        
        # Collect layer states
        layer_states = {}
        for level, result in (layer_results.items() if isinstance(layer_results, dict) else []):
            if isinstance(result, dict):
                layer_states[level] = {
                    'loss': result.get('loss', None),
                    'accuracy': result.get('accuracy', None),
                }
        
        coherence = system_summary.get('coherence', 0.0)
        
        # Generate human-readable summary
        human_readable = self._generate_summary(system_summary, layer_states)
        
        return Observation(
            timestamp=time.time(),
            step=self._optimization_step,
            system_summary=system_summary,
            layer_states=layer_states,
            holographic_coherence=coherence,
            alerts=list(self._alerts),
            human_readable=human_readable,
        )
    
    def _generate_summary(self, summary: Dict, layer_states: Dict) -> str:
        """Generate a human-readable system summary."""
        lines = [
            f"=== System State (step {self._optimization_step}) ===",
            f"Layers active: {summary.get('n_layers_reporting', '?')}",
            f"Coherence: {summary.get('coherence', 0):.3f}",
        ]
        
        for level, state in sorted(layer_states.items()):
            loss = state.get('loss', '?')
            acc = state.get('accuracy', '?')
            if isinstance(loss, float):
                lines.append(f"  Meta^{level}: loss={loss:.4f}, acc={acc:.4f}")
            else:
                lines.append(f"  Meta^{level}: {state}")
        
        if self._alerts:
            lines.append(f"ALERTS: {', '.join(self._alerts)}")
        
        return "\n".join(lines)
    
    def _check_alerts(self, observation: Observation):
        """Check for system anomalies and raise alerts."""
        self._alerts.clear()
        
        # Stagnation check
        if observation.layer_states:
            losses = [s.get('loss', 1.0) for s in observation.layer_states.values()
                     if isinstance(s.get('loss'), (int, float))]
            if losses:
                avg_loss = np.mean(losses)
                if avg_loss < self._last_best_loss:
                    self._last_best_loss = avg_loss
                    self._stagnation_counter = 0
                else:
                    self._stagnation_counter += 1
                
                if self._stagnation_counter > self._alert_thresholds['stagnation']:
                    self._alerts.append(
                        f"STAGNATION: No improvement for {self._stagnation_counter} steps"
                    )
        
        # Coherence check
        if observation.holographic_coherence < self._alert_thresholds['low_coherence']:
            self._alerts.append(
                f"LOW_COHERENCE: {observation.holographic_coherence:.3f}"
            )
    
    def _update_system_model(self, observation: Observation):
        """Update internal model of system behavior."""
        self._system_model = {
            'step': observation.step,
            'coherence': observation.holographic_coherence,
            'n_layers': len(observation.layer_states),
            'stagnation': self._stagnation_counter,
            'alerts': list(self._alerts),
            'trend': 'improving' if self._stagnation_counter == 0 else (
                'stagnant' if self._stagnation_counter < 20 else 'declining'
            ),
        }
    
    def _generate_suggestions(self, observation: Observation) -> List[Tuple[int, Dict]]:
        """
        Generate intervention suggestions based on observations.
        
        Rule-based heuristics (would be replaced by actual LLM calls
        in production).
        """
        suggestions = []
        
        # Stagnation → increase exploration
        if self._stagnation_counter > self._alert_thresholds['stagnation']:
            suggestions.append((4, {
                'exploration_rate': 0.2,
            }))
            
            # Also try switching learning rule
            suggestions.append((2, {
                'selection_strategy': 'epsilon_greedy',
                'epsilon': 0.3,
            }))
        
        # Low coherence → request holographic sync
        if observation.holographic_coherence < self._alert_thresholds['low_coherence']:
            # Ask all layers to report state
            self.signal(
                direction=SignalDirection.BROADCAST,
                signal_type=SignalType.SNAPSHOT_REQUEST,
                payload={'reason': 'low_coherence'},
            )
        
        # Check for specific layer issues
        for level, state in observation.layer_states.items():
            loss = state.get('loss', 0)
            if isinstance(loss, (int, float)) and loss > 0.9:
                # Layer performing very poorly
                if level == 0:
                    # Reset exploration in Meta^0
                    suggestions.append((0, {
                        'threshold': 0.5,  # Lower threshold
                    }))
                elif level == 1:
                    # Try different learning rule
                    suggestions.append((1, {
                        'active_rule': 'both',
                    }))
        
        return suggestions
    
    def _process_llm_response(self, response: Dict):
        """Process response from external LLM."""
        if 'suggestions' in response:
            for sugg in response['suggestions']:
                target = sugg.get('target_level', 0)
                payload = sugg.get('payload', {})
                
                self.signal(
                    direction=SignalDirection.DOWN,
                    signal_type=SignalType.SUGGESTION,
                    payload=payload,
                    target_level=target,
                    requires_response=True,
                    priority=SignalPriority.HIGH,
                )
        
        if 'new_goals' in response:
            for goal in response['new_goals']:
                self.signal(
                    direction=SignalDirection.DOWN,
                    signal_type=SignalType.SUGGESTION,
                    payload={'new_goal': goal},
                    target_level=4,
                    requires_response=True,
                )
        
        if 'architecture' in response:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload={'force_architecture': response['architecture']},
                target_level=5,
                requires_response=True,
                priority=SignalPriority.HIGH,
            )
    
    # ================================================================
    # PUBLIC API for external agents
    # ================================================================
    
    def get_system_report(self) -> str:
        """Get a detailed human-readable system report."""
        if not self.observations:
            return "No observations yet."
        
        latest = self.observations[-1]
        report = [
            latest.human_readable,
            "",
            f"Suggestions sent: {len(self._suggestions_sent)}",
            f"Accepted: {self._suggestions_accepted}",
            f"Rejected: {self._suggestions_rejected}",
            f"Stagnation: {self._stagnation_counter} steps",
            f"System trend: {self._system_model.get('trend', 'unknown')}",
        ]
        return "\n".join(report)
    
    def submit_suggestion(self, target_level: int, payload: Dict):
        """Submit a suggestion from external agent."""
        self.signal(
            direction=SignalDirection.DOWN,
            signal_type=SignalType.SUGGESTION,
            payload=payload,
            target_level=target_level,
            requires_response=True,
            priority=SignalPriority.HIGH,
        )
    
    def get_observation_history(self, n: int = 10) -> List[Dict]:
        """Get recent observations as dicts."""
        recent = list(self.observations)[-n:]
        return [
            {
                'step': obs.step,
                'coherence': obs.holographic_coherence,
                'alerts': obs.alerts,
                'summary': obs.human_readable,
            }
            for obs in recent
        ]
    
    def _handle_reward(self, signal):
        """Track accept/reject responses."""
        pass
    
    def _process_signal(self, signal):
        """Override to track accept/reject responses."""
        super()._process_signal(signal)
        
        if signal.signal_type == SignalType.ACCEPT:
            self._suggestions_accepted += 1
        elif signal.signal_type == SignalType.REJECT:
            self._suggestions_rejected += 1
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state for Meta^N."""
        return np.array([
            float(self._suggestions_accepted),
            float(self._suggestions_rejected),
            float(self._stagnation_counter),
            self._system_model.get('coherence', 0.0),
            float(len(self._alerts)),
            self.metrics.loss,
            self.metrics.accuracy,
        ])
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Meta^N doesn't typically receive suggestions."""
        if 'mode' in suggestion:
            mode = suggestion['mode']
            if mode in ('observer', 'advisor', 'controller'):
                self.llm_config.mode = mode
                return True
        return False
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate suggestions directed at Meta^N."""
        return 0.3, "Meta^N is typically the top-level suggester"
    
    def _self_optimize_weights(self):
        """Adjust alert thresholds based on false alarm rate."""
        pass
    
    def _self_optimize_synapses(self):
        """Clean old observations."""
        pass
    
    def _self_optimize_neurons(self):
        """No structural changes."""
        pass
