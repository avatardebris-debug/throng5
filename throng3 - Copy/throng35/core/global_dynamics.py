"""
GlobalDynamicsOptimizer — Self-Optimizing Coordination for the Fractal Stack

The GlobalDynamicsOptimizer sits above the FractalStack and:
1. Assesses task complexity to determine needed meta-layers
2. Gates layers dynamically based on their contribution
3. Coordinates global optimization to prevent interference
4. Adapts in real-time as task demands change

Key insight: Higher meta-layers (3-5) are designed for complex tasks.
On simple tasks, they add overhead and noise. This optimizer learns
when to activate them.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class GlobalConfig:
    """Configuration for the GlobalDynamicsOptimizer."""
    
    # Complexity assessment
    complexity_window: int = 50  # Steps to assess complexity
    plateau_threshold: float = 0.01  # Improvement below this = plateau
    
    # Layer gating
    min_gate: float = 0.1  # Never fully disable (maintain exploration)
    max_gate: float = 1.0
    gate_adjustment_rate: float = 0.05  # How fast gates change
    warmup_steps: int = 20  # Steps before gating activates
    
    # Contribution tracking
    contribution_window: int = 30  # Steps to average contribution
    positive_threshold: float = 0.05  # Above this = helping
    negative_threshold: float = -0.02  # Below this = hurting
    
    # Update frequency
    update_interval: int = 10  # Update gates every N steps
    
    # Escalation
    escalate_on_plateau: bool = True
    plateau_escalation_gate: float = 0.5  # Gate value when escalating


class GlobalDynamicsOptimizer:
    """
    Global coordinator that tunes the entire fractal stack dynamics.
    
    The optimizer observes the stack's behavior and adjusts layer gates
    to maximize learning efficiency. It implements self-optimization
    at the system level.
    
    Usage:
        optimizer = GlobalDynamicsOptimizer(stack, GlobalConfig())
        
        for step in range(1000):
            result = optimizer.step(context)
            # result includes both stack results and global metrics
    """
    
    def __init__(self, stack: 'FractalStack', config: Optional[GlobalConfig] = None):
        """
        Initialize the global dynamics optimizer.
        
        Args:
            stack: The FractalStack to optimize
            config: Configuration options
        """
        self.stack = stack
        self.config = config or GlobalConfig()
        
        # Layer gates (level -> gate value 0-1)
        self._gates: Dict[int, float] = {}
        self._init_gates()
        
        # Contribution tracking (level -> list of contributions)
        self._contributions: Dict[int, deque] = {}
        for level in self.stack.levels:
            self._contributions[level] = deque(maxlen=self.config.contribution_window)
        
        # History buffers
        self._loss_history: deque = deque(maxlen=100)
        self._reward_history: deque = deque(maxlen=100)
        self._complexity_history: deque = deque(maxlen=50)
        
        # State
        self._step = 0
        self._complexity = 0.5  # Current estimated complexity
        self._last_loss = float('inf')
        self._plateau_count = 0
        
        # Metrics
        self._gate_history: Dict[int, List[float]] = {l: [] for l in self.stack.levels}
        self._contribution_scores: Dict[int, float] = {}
    
    def _init_gates(self):
        """Initialize layer gates with conservative values for higher layers."""
        for level in self.stack.levels:
            if level == 0:
                self._gates[level] = 1.0  # Meta^0 always fully active
            elif level == 1:
                self._gates[level] = 1.0  # Meta^1 synapse optimization usually helps
            elif level == 2:
                self._gates[level] = 0.7  # Meta^2 learning rule selection
            else:
                # Higher layers start with lower gates
                # They must prove they help before being fully activated
                self._gates[level] = max(
                    self.config.min_gate, 
                    0.5 - (level - 2) * 0.1
                )
    
    # ================================================================
    # COMPLEXITY ASSESSMENT
    # ================================================================
    
    def assess_complexity(self, context: Dict[str, Any]) -> float:
        """
        Estimate task complexity from observable signals.
        
        Factors:
        1. Input dimensionality (higher = more complex)
        2. Reward variance/sparsity (higher variance = more complex)
        3. Learning plateau detection (stuck = need higher layers)
        4. Loss trajectory (erratic = complex)
        
        Returns:
            0.0-1.0 complexity score
        """
        factors = []
        
        # 1. Input complexity
        input_data = context.get('input', np.zeros(1))
        if isinstance(input_data, np.ndarray):
            input_dim = input_data.shape[0] if input_data.ndim > 0 else 1
            # Normalize: 16 dims = 0.2, 64 dims = 0.5, 256+ dims = 1.0
            input_complexity = min(1.0, input_dim / 200)
            factors.append(input_complexity)
        
        # 2. Reward variance (sparse/variable rewards = complex)
        if len(self._reward_history) > 10:
            rewards = list(self._reward_history)[-20:]
            reward_variance = np.var(rewards) if rewards else 0
            # Also check sparsity (many zeros = complex)
            sparsity = 1.0 - (np.count_nonzero(rewards) / len(rewards))
            reward_complexity = (reward_variance + sparsity) / 2
            factors.append(min(1.0, reward_complexity))
        
        # 3. Learning plateau (stuck = need higher layers)
        plateau_factor = 0.0
        if len(self._loss_history) > 20:
            recent_losses = list(self._loss_history)[-20:]
            old_losses = list(self._loss_history)[-40:-20] if len(self._loss_history) > 40 else recent_losses
            
            recent_mean = np.mean(recent_losses)
            old_mean = np.mean(old_losses)
            
            improvement = (old_mean - recent_mean) / (old_mean + 1e-8)
            
            if improvement < self.config.plateau_threshold:
                plateau_factor = 0.8  # Plateau detected
                self._plateau_count += 1
            else:
                self._plateau_count = max(0, self._plateau_count - 1)
            
            factors.append(plateau_factor)
        
        # 4. Loss trajectory erraticness
        if len(self._loss_history) > 10:
            recent_losses = list(self._loss_history)[-10:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            cv = loss_std / (loss_mean + 1e-8)  # Coefficient of variation
            trajectory_complexity = min(1.0, cv)
            factors.append(trajectory_complexity)
        
        # Combine factors
        if factors:
            self._complexity = np.mean(factors)
        
        self._complexity_history.append(self._complexity)
        return self._complexity
    
    # ================================================================
    # CONTRIBUTION TRACKING
    # ================================================================
    
    def _update_contributions(self, result: Dict[str, Any]):
        """
        Track whether each layer is helping or hurting performance.
        
        For each layer, estimate its contribution by looking at:
        1. Accept rate of its suggestions (accepted = likely helpful)
        2. Loss change after its optimization step
        3. Signal-to-noise ratio of its outputs
        """
        layer_results = result.get('layer_results', {})
        
        # Get current loss from Meta^0
        current_loss = layer_results.get(0, {}).get('loss', self._last_loss)
        loss_change = self._last_loss - current_loss  # Positive = improvement
        
        for level in self.stack.levels:
            if level == 0:
                # Meta^0 contribution is always positive if loss improves
                contribution = loss_change / (self._last_loss + 1e-8)
                self._contributions[level].append(contribution)
                continue
            
            layer = self.stack.get_layer(level)
            if layer is None:
                continue
            
            # Get layer metrics
            metrics = layer.metrics
            
            # Factors for contribution estimate
            factors = []
            
            # 1. Accept rate (if layer's suggestions are accepted, it's helping)
            # This requires the layer to track accept/reject
            accept_rate = getattr(metrics, 'accept_rate', 0.5)
            factors.append(accept_rate - 0.5)  # Center around 0
            
            # 2. Did loss improve when this layer was active?
            # Crude estimate: attribute proportional improvement
            if loss_change > 0 and self._gates[level] > 0.5:
                factors.append(0.1)  # Small positive contribution
            elif loss_change < 0 and self._gates[level] > 0.5:
                factors.append(-0.1)  # Possible negative contribution
            
            # 3. Signal count (too many signals = noise)
            signals_sent = layer_results.get(level, {}).get('signals_sent', 0)
            if signals_sent > 20:
                factors.append(-0.05)  # Penalize excessive signaling
            
            contribution = np.mean(factors) if factors else 0.0
            self._contributions[level].append(contribution)
            
            # Update running score
            if len(self._contributions[level]) > 5:
                self._contribution_scores[level] = np.mean(
                    list(self._contributions[level])[-20:]
                )
        
        self._last_loss = current_loss
    
    # ================================================================
    # ADAPTIVE LAYER GATING
    # ================================================================
    
    def update_layer_gates(self):
        """
        Adjust layer gates based on contribution scores.
        
        Rules:
        1. Meta^0-1 always stay at 1.0 (core functionality)
        2. Higher layers adjust based on contribution
        3. Never fully disable (maintain exploration)
        4. Escalate on plateau detection
        """
        for level in self.stack.levels:
            if level <= 1:
                self._gates[level] = 1.0
                continue
            
            contributions = list(self._contributions.get(level, []))
            
            if len(contributions) < self.config.warmup_steps:
                # Not enough data yet - keep current gate
                continue
            
            avg_contribution = np.mean(contributions[-20:])
            current_gate = self._gates[level]
            
            # Adjust gate based on contribution
            if avg_contribution > self.config.positive_threshold:
                # Layer is helping - increase gate
                new_gate = current_gate + self.config.gate_adjustment_rate
            elif avg_contribution < self.config.negative_threshold:
                # Layer is hurting - decrease gate
                new_gate = current_gate - self.config.gate_adjustment_rate
            else:
                # Neutral - small decay toward conservative value
                neutral_target = 0.5 - (level - 2) * 0.1
                new_gate = current_gate + 0.01 * (neutral_target - current_gate)
            
            # Apply bounds
            new_gate = np.clip(new_gate, self.config.min_gate, self.config.max_gate)
            
            # Escalation on plateau
            if (self.config.escalate_on_plateau and 
                self._plateau_count > 5 and 
                new_gate < self.config.plateau_escalation_gate):
                new_gate = self.config.plateau_escalation_gate
                logger.debug(f"Escalating layer {level} gate to {new_gate} due to plateau")
            
            self._gates[level] = new_gate
            self._gate_history[level].append(new_gate)
        
        # Apply gates to stack
        self._apply_gates()
    
    def _apply_gates(self):
        """Apply current gate values to the stack."""
        for level, gate in self._gates.items():
            layer = self.stack.get_layer(level)
            if layer is not None:
                # Store gate in layer config for it to use
                layer.config['global_gate'] = gate
                
                # Optionally reduce signal frequency based on gate
                if hasattr(layer, 'signal_frequency'):
                    layer.signal_frequency = gate
    
    # ================================================================
    # MAIN STEP
    # ================================================================
    
    def step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one optimization step with global coordination.
        
        1. Assess task complexity
        2. Run stack step
        3. Update contributions
        4. Update layer gates (periodically)
        5. Return combined results
        
        Args:
            context: External context (input, target, reward, etc.)
            
        Returns:
            Dict with stack results plus global metrics
        """
        # 1. Assess complexity
        complexity = self.assess_complexity(context)
        
        # 2. Record reward
        reward = context.get('reward', 0.0)
        self._reward_history.append(reward)
        
        # 3. Run stack step
        stack_result = self.stack.step(context)
        
        # 4. Record loss
        layer_results = stack_result.get('layer_results', {})
        if 0 in layer_results:
            loss = layer_results[0].get('loss', float('inf'))
            self._loss_history.append(loss)
        
        # 5. Update contributions
        self._update_contributions(stack_result)
        
        # 6. Update gates (periodically)
        if self._step > self.config.warmup_steps:
            if self._step % self.config.update_interval == 0:
                self.update_layer_gates()
        
        self._step += 1
        
        # 7. Return combined results
        return {
            **stack_result,
            'global': {
                'step': self._step,
                'complexity': complexity,
                'gates': dict(self._gates),
                'contributions': dict(self._contribution_scores),
                'plateau_count': self._plateau_count,
            }
        }
    
    # ================================================================
    # QUERIES
    # ================================================================
    
    def get_active_layers(self) -> List[int]:
        """Get list of layers with gate > 0.5 (considered 'active')."""
        return [level for level, gate in self._gates.items() if gate > 0.5]
    
    def get_contribution_scores(self) -> Dict[int, float]:
        """Get current contribution scores for all layers."""
        return dict(self._contribution_scores)
    
    def get_gates(self) -> Dict[int, float]:
        """Get current gate values for all layers."""
        return dict(self._gates)
    
    def get_complexity(self) -> float:
        """Get current estimated task complexity."""
        return self._complexity
    
    def get_report(self) -> str:
        """Get a human-readable status report."""
        lines = [
            "=== Global Dynamics Optimizer ===",
            f"Step: {self._step}",
            f"Complexity: {self._complexity:.3f}",
            f"Plateau count: {self._plateau_count}",
            "",
            "Layer Gates:",
        ]
        
        for level in sorted(self._gates.keys()):
            gate = self._gates[level]
            contrib = self._contribution_scores.get(level, 0.0)
            status = "ACTIVE" if gate > 0.5 else "GATED"
            lines.append(f"  Meta^{level}: gate={gate:.2f}, contrib={contrib:+.3f} [{status}]")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        active = len(self.get_active_layers())
        total = len(self._gates)
        return f"GlobalDynamicsOptimizer(step={self._step}, active={active}/{total}, complexity={self._complexity:.2f})"
