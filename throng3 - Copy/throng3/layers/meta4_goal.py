"""
Meta^4: GoalHierarchy — Multi-Timescale Reward Management

Manages the hierarchy of goals at different time scales:
- Short-term: Immediate rewards (next few steps)
- Medium-term: Episode-level rewards (task completion)
- Long-term: Lifetime rewards (general capability)

This layer decides how to balance exploration vs exploitation,
when to sacrifice short-term reward for long-term gain,
and how to decompose complex goals into sub-goals.

Inspired by throng2's dopamine system but elevated to meta-level.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class GoalConfig:
    """Configuration for goal hierarchy."""
    n_timescales: int = 3                    # short, medium, long
    discount_factors: List[float] = field(default_factory=lambda: [0.9, 0.99, 0.999])
    exploration_rate: float = 0.1            # Global exploration rate
    exploration_decay: float = 0.9999        # Exploration decay per step
    min_exploration: float = 0.01
    goal_switch_threshold: float = 0.1       # When to switch active goal
    intrinsic_reward_weight: float = 0.3     # Weight for curiosity/novelty
    extrinsic_reward_weight: float = 0.7     # Weight for external rewards


@dataclass
class Goal:
    """A goal at a specific timescale."""
    name: str
    timescale: int                    # 0=short, 1=medium, 2=long
    target_value: float = 1.0        # What success looks like
    current_value: float = 0.0       # Progress toward goal
    priority: float = 1.0            # Relative importance
    active: bool = True
    created_step: int = 0
    achieved: bool = False
    deadline: Optional[int] = None   # Step by which to achieve


class GoalHierarchy(MetaLayer):
    """
    Meta^4: Multi-timescale goal management.
    
    Balances short-term vs long-term rewards and manages
    exploration/exploitation tradeoff across the whole system.
    """
    
    def __init__(self, config: Optional[GoalConfig] = None, **kwargs):
        cfg = config or GoalConfig()
        super().__init__(level=4, name="GoalHierarchy", config=vars(cfg))
        self.goal_config = cfg
        
        # Goal registry
        self.goals: Dict[str, Goal] = {}
        self._init_default_goals()
        
        # Multi-timescale value tracking
        self.n_timescales = cfg.n_timescales
        self.value_estimates = [0.0] * cfg.n_timescales  # V(s) at each scale
        self.reward_accumulators = [deque(maxlen=1000) for _ in range(cfg.n_timescales)]
        
        # Exploration state
        self.exploration_rate = cfg.exploration_rate
        
        # Intrinsic motivation (curiosity)
        self._prediction_errors: deque = deque(maxlen=500)
        self._novelty_scores: deque = deque(maxlen=500)
        self._state_visit_counts: Dict[str, int] = {}
        
        # Performance by timescale
        self._timescale_performance: List[deque] = [
            deque(maxlen=200) for _ in range(cfg.n_timescales)
        ]
        
        # Goal achievement history
        self._achievement_history: deque = deque(maxlen=100)
    
    def _init_default_goals(self):
        """Initialize default goals at each timescale."""
        self.goals['minimize_loss_short'] = Goal(
            name='minimize_loss_short',
            timescale=0,
            target_value=0.1,
            priority=1.0,
        )
        self.goals['minimize_loss_medium'] = Goal(
            name='minimize_loss_medium',
            timescale=1,
            target_value=0.01,
            priority=0.8,
        )
        self.goals['maximize_generalization'] = Goal(
            name='maximize_generalization',
            timescale=2,
            target_value=0.9,
            priority=0.6,
        )
        self.goals['maximize_efficiency'] = Goal(
            name='maximize_efficiency',
            timescale=2,
            target_value=0.8,
            priority=0.5,
        )
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage goal hierarchy and reward allocation.
        
        1. Receive reward signals from environment and lower layers
        2. Decompose reward across timescales
        3. Update goal progress
        4. Compute intrinsic motivation (curiosity)
        5. Set exploration rate
        6. Signal reward/guidance DOWN (every 10 steps to reduce noise)
        """
        self.process_inbox()
        
        # Only send suggestions every 10 steps to reduce signal noise
        should_send_suggestions = (self._optimization_step % 10 == 0)
        
        # Get reward info
        extrinsic_reward = context.get('reward', 0.0)
        system_summary = context.get('system_summary', {})
        
        # Compute intrinsic reward (curiosity/novelty)
        intrinsic_reward = self._compute_intrinsic_reward(context)
        
        # Combined reward
        total_reward = (
            self.goal_config.extrinsic_reward_weight * extrinsic_reward +
            self.goal_config.intrinsic_reward_weight * intrinsic_reward
        )
        
        # Decompose across timescales
        timescale_rewards = self._decompose_reward(total_reward)
        
        # Update value estimates
        for i, (reward, gamma) in enumerate(
            zip(timescale_rewards, self.goal_config.discount_factors)
        ):
            self.reward_accumulators[i].append(reward)
            # TD-like update
            self.value_estimates[i] = (
                gamma * self.value_estimates[i] + (1 - gamma) * reward
            )
            self._timescale_performance[i].append(reward)
        
        # Update goal progress
        self._update_goals(context)
        
        # Compute optimal exploration rate
        self._update_exploration()
        
        # Signal reward DOWN to lower layers (only every 10 steps)
        if should_send_suggestions:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.REWARD,
                payload={
                    'total_reward': total_reward,
                    'extrinsic': extrinsic_reward,
                    'intrinsic': intrinsic_reward,
                    'timescale_rewards': timescale_rewards,
                    'exploration_rate': self.exploration_rate,
                    'active_goals': self._get_active_goals(),
                },
                target_level=None,  # Broadcast to all lower layers
            )
            
            # Signal UP about goal status
            self.signal(
                direction=SignalDirection.UP,
                signal_type=SignalType.PERFORMANCE,
                payload={
                    'value_estimates': list(self.value_estimates),
                    'exploration_rate': self.exploration_rate,
                    'n_active_goals': sum(1 for g in self.goals.values() if g.active),
                    'n_achieved_goals': sum(1 for g in self.goals.values() if g.achieved),
                    'intrinsic_reward': intrinsic_reward,
                },
            )
        
        # Metrics
        goal_achievement_rate = sum(
            1 for g in self.goals.values() if g.achieved
        ) / max(len(self.goals), 1)
        self.metrics.update(1.0 - goal_achievement_rate, goal_achievement_rate)
        
        return {
            'total_reward': total_reward,
            'intrinsic_reward': intrinsic_reward,
            'value_estimates': list(self.value_estimates),
            'exploration_rate': self.exploration_rate,
            'goal_achievement_rate': goal_achievement_rate,
            'metrics': self.metrics,
        }
    
    def _compute_intrinsic_reward(self, context: Dict) -> float:
        """
        Compute intrinsic motivation reward (curiosity-driven).
        
        Based on:
        - Prediction error (surprise)
        - State novelty (visit counts)
        - Information gain
        """
        # Prediction error component
        holographic = context.get('holographic_view', np.zeros(10))
        if isinstance(holographic, np.ndarray):
            state_key = str(np.round(holographic[:5], 2).tolist())
        else:
            state_key = "default"
        
        # Novelty from visit counts
        self._state_visit_counts[state_key] = self._state_visit_counts.get(state_key, 0) + 1
        visit_count = self._state_visit_counts[state_key]
        novelty = 1.0 / np.sqrt(visit_count)
        self._novelty_scores.append(novelty)
        
        # Prediction error from system coherence
        coherence = context.get('system_summary', {}).get('coherence', 1.0)
        prediction_error = 1.0 - coherence
        self._prediction_errors.append(prediction_error)
        
        # Combine
        intrinsic = 0.5 * novelty + 0.5 * prediction_error
        return float(np.clip(intrinsic, 0, 1))
    
    def _decompose_reward(self, total_reward: float) -> List[float]:
        """Decompose a single reward into multi-timescale components."""
        rewards = []
        for i in range(self.n_timescales):
            gamma = self.goal_config.discount_factors[i]
            # Short timescale: more of immediate reward
            # Long timescale: smoothed running average
            weight = (1 - gamma)  # Short = high weight on immediate
            rewards.append(total_reward * weight + self.value_estimates[i] * (1 - weight))
        return rewards
    
    def _update_goals(self, context: Dict):
        """Update goal progress based on current state."""
        system_summary = context.get('system_summary', {})
        layer_results = context.get('layer_results', {})
        
        for name, goal in self.goals.items():
            if not goal.active or goal.achieved:
                continue
            
            # Update progress based on goal type
            if 'loss' in name:
                # Loss-based goals: check system loss
                losses = []
                for level, result in layer_results.items() if isinstance(layer_results, dict) else []:
                    if isinstance(result, dict) and 'loss' in result:
                        losses.append(result['loss'])
                
                if losses:
                    avg_loss = np.mean(losses)
                    goal.current_value = 1.0 - min(avg_loss, 1.0)
            
            elif 'efficiency' in name:
                n_reporting = system_summary.get('n_layers_reporting', 0)
                goal.current_value = n_reporting / max(6, 1)  # Rough efficiency proxy
            
            elif 'generalization' in name:
                # Placeholder: would need actual generalization test
                goal.current_value = self.metrics.stability
            
            # Check achievement
            if goal.current_value >= goal.target_value:
                goal.achieved = True
                self._achievement_history.append({
                    'goal': name,
                    'step': self._optimization_step,
                    'value': goal.current_value,
                })
    
    def _update_exploration(self):
        """Update exploration rate based on goal progress."""
        # Decay exploration
        self.exploration_rate *= self.goal_config.exploration_decay
        self.exploration_rate = max(
            self.exploration_rate,
            self.goal_config.min_exploration
        )
        
        # Boost exploration if stuck
        if len(self._novelty_scores) > 50:
            recent_novelty = np.mean(list(self._novelty_scores)[-50:])
            if recent_novelty < 0.1:  # Very low novelty = stuck
                self.exploration_rate = min(
                    self.exploration_rate * 1.5,
                    self.goal_config.exploration_rate
                )
    
    def _get_active_goals(self) -> List[Dict]:
        """Get summary of active goals."""
        return [
            {
                'name': g.name,
                'timescale': g.timescale,
                'progress': g.current_value / max(g.target_value, 1e-8),
                'priority': g.priority,
            }
            for g in self.goals.values()
            if g.active and not g.achieved
        ]
    
    def add_goal(self, name: str, timescale: int, target: float,
                 priority: float = 1.0, deadline: Optional[int] = None):
        """Add a new goal to the hierarchy."""
        self.goals[name] = Goal(
            name=name,
            timescale=timescale,
            target_value=target,
            priority=priority,
            created_step=self._optimization_step,
            deadline=deadline,
        )
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state for Meta^4."""
        goal_progress = [
            g.current_value / max(g.target_value, 1e-8)
            for g in self.goals.values()
        ]
        
        return np.array(
            self.value_estimates +
            goal_progress[:10] +  # Cap at 10 goals
            [0.0] * max(0, 10 - len(goal_progress)) +
            [
                self.exploration_rate,
                self.metrics.loss,
                self.metrics.accuracy,
                float(len(self._achievement_history)),
            ]
        )
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestions from Meta^5 or LLM."""
        applied = False
        
        if 'exploration_rate' in suggestion:
            self.exploration_rate = np.clip(suggestion['exploration_rate'], 0, 1)
            applied = True
        
        if 'new_goal' in suggestion:
            g = suggestion['new_goal']
            self.add_goal(
                name=g.get('name', f'custom_{len(self.goals)}'),
                timescale=g.get('timescale', 1),
                target=g.get('target', 0.5),
                priority=g.get('priority', 1.0),
            )
            applied = True
        
        if 'discount_factors' in suggestion:
            factors = suggestion['discount_factors']
            if len(factors) == self.n_timescales:
                self.goal_config.discount_factors = [
                    np.clip(f, 0.5, 0.9999) for f in factors
                ]
                applied = True
        
        if 'reward_weights' in suggestion:
            w = suggestion['reward_weights']
            if 'intrinsic' in w:
                self.goal_config.intrinsic_reward_weight = np.clip(w['intrinsic'], 0, 1)
            if 'extrinsic' in w:
                self.goal_config.extrinsic_reward_weight = np.clip(w['extrinsic'], 0, 1)
            applied = True
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate suggestions."""
        score = 0.5
        reasons = []
        
        if 'exploration_rate' in suggestion:
            rate = suggestion['exploration_rate']
            if 0 <= rate <= 1:
                score = 0.7
                reasons.append(f"Valid exploration rate: {rate}")
            else:
                score = 0.0
                reasons.append("Invalid exploration rate")
        
        if 'new_goal' in suggestion:
            score = max(score, 0.8)
            reasons.append("New goal — generally welcome")
        
        return score, "; ".join(reasons) if reasons else "No criteria"
    
    def _self_optimize_weights(self):
        """Auto-tune reward weights based on recent performance."""
        # If intrinsic motivation is producing results, increase its weight
        if len(self._novelty_scores) > 50:
            recent_novelty = np.mean(list(self._novelty_scores)[-50:])
            if recent_novelty > 0.5:
                # High novelty = exploring new territory, boost intrinsic
                self.goal_config.intrinsic_reward_weight = min(
                    self.goal_config.intrinsic_reward_weight * 1.001, 0.5
                )
    
    def _self_optimize_synapses(self):
        """Clean stale state visit counts."""
        if len(self._state_visit_counts) > 10000:
            # Keep only recently visited states
            sorted_states = sorted(
                self._state_visit_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self._state_visit_counts = dict(sorted_states[:5000])
    
    def _self_optimize_neurons(self):
        """Archive achieved goals."""
        pass
