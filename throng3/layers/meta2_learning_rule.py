"""
Meta^2: LearningRuleSelector — Chooses Which Learning Rule to Apply

Observes Meta^1's performance under different learning rules and
decides which rule (or combination) is best for the current situation.

This is "learning to learn" — it doesn't learn the task directly,
it learns which learning algorithm works best.

Strategies:
- Bandit-based: Treat rule selection as multi-armed bandit
- Context-dependent: Different rules for different input patterns
- Adaptive blending: Weighted combination of rules
- Performance-triggered switching: Change rules when performance plateaus
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class LearningRuleSelectorConfig:
    """Configuration for the learning rule selector."""
    available_rules: List[str] = field(default_factory=lambda: ['stdp', 'hebbian', 'both'])
    selection_strategy: str = 'ucb'   # 'ucb', 'epsilon_greedy', 'softmax', 'contextual'
    epsilon: float = 0.1             # For epsilon-greedy
    temperature: float = 1.0         # For softmax
    ucb_c: float = 2.0              # UCB exploration constant
    evaluation_window: int = 50      # Steps to evaluate a rule before considering switch
    switch_threshold: float = 0.05   # Minimum improvement to switch rules
    parameter_search: bool = True    # Also optimize rule parameters


class LearningRuleSelector(MetaLayer):
    """
    Meta^2: Learning rule selection and parameter optimization.
    
    Decides which learning rule Meta^1 should use, and tunes
    the parameters of that rule.
    """
    
    def __init__(self, config: Optional[LearningRuleSelectorConfig] = None, **kwargs):
        cfg = config or LearningRuleSelectorConfig()
        super().__init__(level=2, name="LearningRuleSelector", config=vars(cfg))
        self.selector_config = cfg
        
        # Rule performance tracking (multi-armed bandit)
        self.rules = cfg.available_rules
        self.n_rules = len(self.rules)
        
        # Bandit state
        self.rule_rewards: Dict[str, List[float]] = {r: [] for r in self.rules}
        self.rule_counts: Dict[str, int] = {r: 0 for r in self.rules}
        self.rule_values: Dict[str, float] = {r: 0.0 for r in self.rules}
        
        # Current selection
        self.current_rule = self.rules[0]
        self._steps_on_current_rule = 0
        
        # Context features for contextual bandit
        self._context_features: deque = deque(maxlen=1000)
        self._context_rule_pairs: List[Tuple[np.ndarray, str, float]] = []
        
        # Parameter optimization state
        self._parameter_perturbation: Dict[str, Dict[str, float]] = {}
        self._best_params: Dict[str, Dict[str, float]] = {}
        
        # Meta-metrics: how well is rule selection itself performing
        self._selection_history: deque = deque(maxlen=500)
        self._improvement_after_switch: deque = deque(maxlen=100)
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the best learning rule based on observed performance.
        
        1. Observe Meta^1's recent performance
        2. Update bandit estimates
        3. Select rule using chosen strategy
        4. Optionally perturb rule parameters
        5. Signal suggestion DOWN to Meta^1
        """
        self.process_inbox()
        
        # Get Meta^1's performance from context
        meta1_perf = context.get('meta1_performance', {})
        current_loss = meta1_perf.get('loss', self.metrics.loss)
        current_rpe = meta1_perf.get('rpe', 0.0)
        active_rule = meta1_perf.get('active_rule', self.current_rule)
        
        # Compute reward for current rule
        reward = self._compute_rule_reward(current_loss, current_rpe)
        
        # Update bandit
        self._update_bandit(active_rule, reward)
        self._steps_on_current_rule += 1
        
        # Select rule (only consider switching after evaluation window)
        if self._steps_on_current_rule >= self.selector_config.evaluation_window:
            new_rule = self._select_rule(context)
            
            if new_rule != self.current_rule:
                # Record the switch
                self._selection_history.append({
                    'from': self.current_rule,
                    'to': new_rule,
                    'step': self._optimization_step,
                    'reason': f"UCB selected {new_rule}",
                })
                
                self.current_rule = new_rule
                self._steps_on_current_rule = 0
                
                # Signal DOWN to Meta^1 to change rule
                self.signal(
                    direction=SignalDirection.DOWN,
                    signal_type=SignalType.SUGGESTION,
                    payload={'active_rule': new_rule},
                    target_level=1,
                    requires_response=True,
                )
        
        # Parameter optimization
        if self.selector_config.parameter_search:
            param_suggestion = self._optimize_parameters(context)
            if param_suggestion:
                self.signal(
                    direction=SignalDirection.DOWN,
                    signal_type=SignalType.SUGGESTION,
                    payload=param_suggestion,
                    target_level=1,
                    requires_response=True,
                )
        
        # Compute own metrics
        selection_quality = self._compute_selection_quality()
        self.metrics.update(1.0 - selection_quality, selection_quality)
        
        # Signal performance UP
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'current_rule': self.current_rule,
                'rule_values': dict(self.rule_values),
                'rule_counts': dict(self.rule_counts),
                'selection_quality': selection_quality,
            },
        )
        
        return {
            'current_rule': self.current_rule,
            'rule_values': dict(self.rule_values),
            'selection_quality': selection_quality,
            'metrics': self.metrics,
        }
    
    def _compute_rule_reward(self, loss: float, rpe: float) -> float:
        """
        Compute reward signal for the bandit from Meta^1's performance.
        
        Combines loss reduction with reward prediction error.
        """
        # Reward = improvement in loss + positive RPE bonus
        improvement = 0.0
        if self.current_rule in self.rule_rewards and self.rule_rewards[self.current_rule]:
            prev = self.rule_rewards[self.current_rule][-1]
            improvement = prev - loss  # Positive = better
        
        reward = 0.5 + improvement + 0.1 * max(rpe, 0)
        reward = np.clip(reward, 0, 1)
        return float(reward)
    
    def _update_bandit(self, rule: str, reward: float):
        """Update bandit estimates for a rule."""
        # Skip tracking for 'none' rule (e.g., when Meta^1 is in Q-learning only mode)
        if rule != 'none' and rule in self.rule_rewards:
            self.rule_rewards[rule].append(reward)
            self.rule_counts[rule] += 1
            
            # Incremental mean update
            n = self.rule_counts[rule]
            self.rule_values[rule] += (reward - self.rule_values[rule]) / n
        elif rule == 'none':
            # If 'none' rule is active, we still count steps but don't update values
            # This prevents division by zero if 'none' is the only rule ever active
            # and ensures total count for UCB is accurate.
            if rule not in self.rule_counts:
                self.rule_counts[rule] = 0
            self.rule_counts[rule] += 1
    
    def _select_rule(self, context: Dict) -> str:
        """Select a rule using the configured strategy."""
        strategy = self.selector_config.selection_strategy
        
        if strategy == 'ucb':
            return self._ucb_select()
        elif strategy == 'epsilon_greedy':
            return self._epsilon_greedy_select()
        elif strategy == 'softmax':
            return self._softmax_select()
        elif strategy == 'contextual':
            return self._contextual_select(context)
        
        return self.current_rule
    
    def _ucb_select(self) -> str:
        """Upper Confidence Bound selection."""
        total = sum(self.rule_counts.values())
        if total == 0:
            return np.random.choice(self.rules)
        
        ucb_values = {}
        for rule in self.rules:
            n = max(self.rule_counts[rule], 1)
            exploitation = self.rule_values[rule]
            exploration = self.selector_config.ucb_c * np.sqrt(np.log(total) / n)
            ucb_values[rule] = exploitation + exploration
        
        return max(ucb_values, key=ucb_values.get)
    
    def _epsilon_greedy_select(self) -> str:
        """Epsilon-greedy selection."""
        if np.random.random() < self.selector_config.epsilon:
            return np.random.choice(self.rules)
        return max(self.rule_values, key=self.rule_values.get)
    
    def _softmax_select(self) -> str:
        """Softmax/Boltzmann selection."""
        values = np.array([self.rule_values[r] for r in self.rules])
        temp = self.selector_config.temperature
        
        exp_values = np.exp((values - np.max(values)) / max(temp, 1e-8))
        probs = exp_values / np.sum(exp_values)
        
        return np.random.choice(self.rules, p=probs)
    
    def _contextual_select(self, context: Dict) -> str:
        """Context-dependent selection using simple feature matching."""
        # Extract context features
        holographic = context.get('holographic_view', np.zeros(10))
        if isinstance(holographic, np.ndarray) and len(holographic) > 0:
            features = holographic[:10]  # Use first 10 dims
        else:
            features = np.zeros(10)
        
        # If we have enough history, use nearest-neighbor
        if len(self._context_rule_pairs) >= 20:
            best_rule = self.current_rule
            best_score = -float('inf')
            
            for ctx_feat, rule, reward in self._context_rule_pairs[-100:]:
                similarity = np.dot(features, ctx_feat) / (
                    np.linalg.norm(features) * np.linalg.norm(ctx_feat) + 1e-8
                )
                score = similarity * reward
                if score > best_score:
                    best_score = score
                    best_rule = rule
            
            return best_rule
        
        # Fallback to UCB
        return self._ucb_select()
    
    def _optimize_parameters(self, context: Dict) -> Optional[Dict]:
        """
        Optimize the parameters of the current learning rule.
        
        Uses evolutionary strategy: perturb → evaluate → keep if better.
        """
        if self._optimization_step % 20 != 0:  # Don't perturb every step
            return None
        
        # Generate small random perturbation for current rule's params
        if self.current_rule == 'stdp':
            perturbation = {
                'stdp_params': {
                    'A_plus': np.random.randn() * 0.001,
                    'A_minus': np.random.randn() * 0.001,
                    'tau_plus': np.random.randn() * 0.001,
                    'tau_minus': np.random.randn() * 0.001,
                }
            }
        elif self.current_rule == 'hebbian':
            perturbation = {
                'hebbian_params': {
                    'learning_rate': np.random.randn() * 0.001,
                    'decay': np.random.randn() * 0.0001,
                }
            }
        else:
            perturbation = {
                'learning_rate': self.synapse_config_lr() + np.random.randn() * 0.001
            }
        
        return perturbation
    
    def synapse_config_lr(self) -> float:
        """Get current synapse learning rate estimate."""
        return self.config.get('learning_rate', 0.01)
    
    def _compute_selection_quality(self) -> float:
        """How well is rule selection performing overall."""
        if not self.rule_rewards:
            return 0.5
        
        # Quality = best rule's recent performance
        recent_window = 50
        best_recent = 0.0
        for rule, rewards in self.rule_rewards.items():
            if rewards:
                recent = rewards[-recent_window:]
                best_recent = max(best_recent, np.mean(recent))
        
        return float(best_recent)
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state for Meta^2."""
        values = [self.rule_values.get(r, 0.0) for r in self.rules]
        counts = [self.rule_counts.get(r, 0) / max(sum(self.rule_counts.values()), 1)
                  for r in self.rules]
        
        return np.array(values + counts + [
            self.metrics.loss,
            self.metrics.accuracy,
            self.metrics.stability,
            float(self._steps_on_current_rule),
            len(self._selection_history),
        ])
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestion from Meta^3 or higher."""
        applied = False
        
        if 'selection_strategy' in suggestion:
            new_strategy = suggestion['selection_strategy']
            if new_strategy in ('ucb', 'epsilon_greedy', 'softmax', 'contextual'):
                self.selector_config.selection_strategy = new_strategy
                applied = True
        
        if 'epsilon' in suggestion:
            self.selector_config.epsilon = np.clip(suggestion['epsilon'], 0, 1)
            applied = True
        
        if 'temperature' in suggestion:
            self.selector_config.temperature = max(suggestion['temperature'], 0.01)
            applied = True
        
        if 'ucb_c' in suggestion:
            self.selector_config.ucb_c = max(suggestion['ucb_c'], 0.1)
            applied = True
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate a suggestion."""
        score = 0.5
        reasons = []
        
        if 'selection_strategy' in suggestion:
            # Accept new strategies when current one isn't working well
            quality = self._compute_selection_quality()
            if quality < 0.4:
                score = 0.8
                reasons.append("Low quality, open to strategy change")
            else:
                score = 0.3
                reasons.append("Strategy working well, prefer to keep")
        
        return score, "; ".join(reasons) if reasons else "No criteria"
    
    def _self_optimize_weights(self):
        """Adapt exploration/exploitation balance."""
        # Reduce epsilon over time (more exploitation as we learn)
        if self.selector_config.selection_strategy == 'epsilon_greedy':
            self.selector_config.epsilon *= 0.999
            self.selector_config.epsilon = max(self.selector_config.epsilon, 0.01)
        
        # Reduce temperature (more confident selections)
        if self.selector_config.selection_strategy == 'softmax':
            self.selector_config.temperature *= 0.999
            self.selector_config.temperature = max(self.selector_config.temperature, 0.1)
    
    def _self_optimize_synapses(self):
        """Clean old context-rule pairs."""
        if len(self._context_rule_pairs) > 500:
            self._context_rule_pairs = self._context_rule_pairs[-200:]
    
    def _self_optimize_neurons(self):
        """No structural changes at Meta^2."""
        pass
