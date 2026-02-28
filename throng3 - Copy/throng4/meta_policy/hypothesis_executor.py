"""
Hypothesis Executor — Translate LLM's linguistic hypotheses into executable strategies.

Parses LLM responses like "try defensive positioning" or "test timing-based attacks"
and converts them into concrete policy modifications:
- Exploration bias (favor/avoid certain actions)
- Reward shaping (add intrinsic rewards)
- Action timing (wait for patterns)
- Concept weighting (boost relevant concepts)

This closes the loop: LLM reasons abstractly → Executor grounds it → Agent tests it.
"""

import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class StrategyType(Enum):
    """Types of strategies the LLM can suggest."""
    DEFENSIVE = "defensive"
    OFFENSIVE = "offensive"
    SPATIAL = "spatial"
    TIMING = "timing"
    EXPLORATORY = "exploratory"
    CONSERVATIVE = "conservative"


@dataclass
class ExecutableStrategy:
    """A strategy that can be applied to a pipeline."""
    name: str
    strategy_type: StrategyType
    exploration_modifier: float = 1.0  # Multiply epsilon by this
    action_bias: Dict[int, float] = field(default_factory=dict)  # Favor/avoid actions
    reward_shaping: Dict[str, float] = field(default_factory=dict)  # Intrinsic rewards
    timing_delay: int = 0  # Wait N steps between actions
    description: str = ""
    
    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"{self.name} ({self.strategy_type.value})"]
        
        if self.exploration_modifier != 1.0:
            parts.append(f"exploration×{self.exploration_modifier:.2f}")
        
        if self.action_bias:
            parts.append(f"bias_{len(self.action_bias)}_actions")
        
        if self.reward_shaping:
            parts.append(f"shape_{len(self.reward_shaping)}_rewards")
        
        if self.timing_delay > 0:
            parts.append(f"delay={self.timing_delay}")
        
        return ", ".join(parts)


class HypothesisExecutor:
    """
    Parse LLM's linguistic hypotheses and execute them.
    
    The LLM sees abstract patterns and suggests strategies in natural language.
    This class translates those suggestions into concrete policy changes.
    """
    
    # Strategy templates map keywords → modifications
    STRATEGY_TEMPLATES = {
        'defensive': {
            'type': StrategyType.DEFENSIVE,
            'exploration_modifier': 0.5,  # Reduce exploration
            'reward_shaping': {'survival': 0.01},  # Reward staying alive
        },
        'offensive': {
            'type': StrategyType.OFFENSIVE,
            'exploration_modifier': 1.5,  # Increase exploration
            'reward_shaping': {'aggression': 0.1},  # Reward taking risks
        },
        'spatial': {
            'type': StrategyType.SPATIAL,
            'exploration_modifier': 0.8,
            'reward_shaping': {'positioning': 0.05},
        },
        'timing': {
            'type': StrategyType.TIMING,
            'timing_delay': 3,  # Wait between actions
            'exploration_modifier': 0.7,
        },
        'exploratory': {
            'type': StrategyType.EXPLORATORY,
            'exploration_modifier': 2.0,  # Max exploration
        },
        'conservative': {
            'type': StrategyType.CONSERVATIVE,
            'exploration_modifier': 0.3,  # Min exploration
        },
    }
    
    def __init__(self):
        self.active_strategy: Optional[ExecutableStrategy] = None
    
    def parse_hypothesis(self, llm_text: str,
                         visual_patterns: Dict = None,
                         causal_effects: Dict = None) -> ExecutableStrategy:
        """
        Parse LLM's text response into an executable strategy.
        
        Args:
            llm_text: LLM's natural language suggestion
            visual_patterns: Visual patterns from VisualPatternExtractor
            causal_effects: Action effects from CausalDiscovery
            
        Returns:
            ExecutableStrategy ready to apply
        """
        llm_lower = llm_text.lower()
        
        # Extract strategy type from keywords
        strategy_type = self._extract_strategy_type(llm_lower)
        
        # Start with template
        template = self.STRATEGY_TEMPLATES.get(
            strategy_type.value,
            self.STRATEGY_TEMPLATES['exploratory']
        )
        
        strategy = ExecutableStrategy(
            name=f"llm_{strategy_type.value}",
            strategy_type=strategy_type,
            exploration_modifier=template.get('exploration_modifier', 1.0),
            reward_shaping=template.get('reward_shaping', {}).copy(),
            timing_delay=template.get('timing_delay', 0),
            description=llm_text[:200],
        )
        
        # Extract action biases from causal effects
        if causal_effects:
            strategy.action_bias = self._extract_action_bias(
                llm_lower, causal_effects, strategy_type
            )
        
        return strategy
    
    def _extract_strategy_type(self, text: str) -> StrategyType:
        """Extract strategy type from LLM text."""
        # Check for keywords
        if any(word in text for word in ['defensive', 'defend', 'avoid', 'safe']):
            return StrategyType.DEFENSIVE
        elif any(word in text for word in ['offensive', 'attack', 'aggressive', 'shoot']):
            return StrategyType.OFFENSIVE
        elif any(word in text for word in ['spatial', 'position', 'location', 'zone']):
            return StrategyType.SPATIAL
        elif any(word in text for word in ['timing', 'wait', 'pattern', 'rhythm']):
            return StrategyType.TIMING
        elif any(word in text for word in ['explore', 'try', 'experiment', 'test']):
            return StrategyType.EXPLORATORY
        elif any(word in text for word in ['conservative', 'careful', 'cautious']):
            return StrategyType.CONSERVATIVE
        else:
            return StrategyType.EXPLORATORY
    
    def _extract_action_bias(self, text: str, causal_effects: Dict,
                             strategy_type: StrategyType) -> Dict[int, float]:
        """
        Extract which actions to favor/avoid based on strategy and causal effects.
        
        Returns:
            Dict mapping action_id → bias multiplier (>1 = favor, <1 = avoid)
        """
        bias = {}
        
        if strategy_type == StrategyType.OFFENSIVE:
            # Favor entity-creating actions (likely attacks)
            for action_id, effect in causal_effects.items():
                if effect.creates_entities:
                    bias[action_id] = 2.0  # Favor shooting
                elif effect.avg_state_delta < 0.01:
                    bias[action_id] = 0.5  # Avoid do-nothing
        
        elif strategy_type == StrategyType.DEFENSIVE:
            # Favor movement, avoid entity creation
            for action_id, effect in causal_effects.items():
                if effect.creates_entities:
                    bias[action_id] = 0.3  # Avoid shooting
                elif effect.avg_state_delta > 0.02:
                    bias[action_id] = 1.5  # Favor movement
        
        elif strategy_type == StrategyType.SPATIAL:
            # Favor high-delta actions (movement)
            for action_id, effect in causal_effects.items():
                if effect.avg_state_delta > 0.02:
                    bias[action_id] = 1.8
        
        elif strategy_type == StrategyType.EXPLORATORY:
            # Favor actions with high reward variance (uncertain)
            for action_id, effect in causal_effects.items():
                if effect.reward_variance > 0.1:
                    bias[action_id] = 1.5
        
        return bias
    
    def apply_strategy(self, strategy: ExecutableStrategy, pipeline) -> Dict:
        """
        Apply strategy to a MetaStackPipeline.
        
        Modifies:
        - Exploration rate (epsilon)
        - Action selection (via bias weights)
        - Reward shaping (intrinsic rewards)
        
        Returns:
            Dict with modifications made
        """
        modifications = {}
        
        # Modify exploration rate
        if hasattr(pipeline, 'epsilon'):
            old_epsilon = pipeline.epsilon
            pipeline.epsilon *= strategy.exploration_modifier
            pipeline.epsilon = np.clip(pipeline.epsilon, 0.05, 0.95)
            modifications['epsilon'] = {
                'old': old_epsilon,
                'new': pipeline.epsilon,
            }
        
        # Store action bias for select_action
        if strategy.action_bias:
            pipeline._hypothesis_action_bias = strategy.action_bias
            modifications['action_bias'] = strategy.action_bias
        
        # Store reward shaping for update
        if strategy.reward_shaping:
            pipeline._hypothesis_reward_shaping = strategy.reward_shaping
            modifications['reward_shaping'] = strategy.reward_shaping
        
        # Store timing delay
        if strategy.timing_delay > 0:
            pipeline._hypothesis_timing_delay = strategy.timing_delay
            pipeline._hypothesis_step_counter = 0
            modifications['timing_delay'] = strategy.timing_delay
        
        self.active_strategy = strategy
        
        return modifications
    
    def remove_strategy(self, pipeline) -> None:
        """Remove active strategy from pipeline."""
        # Reset to defaults
        if hasattr(pipeline, '_hypothesis_action_bias'):
            delattr(pipeline, '_hypothesis_action_bias')
        if hasattr(pipeline, '_hypothesis_reward_shaping'):
            delattr(pipeline, '_hypothesis_reward_shaping')
        if hasattr(pipeline, '_hypothesis_timing_delay'):
            delattr(pipeline, '_hypothesis_timing_delay')
        if hasattr(pipeline, '_hypothesis_step_counter'):
            delattr(pipeline, '_hypothesis_step_counter')
        
        self.active_strategy = None


if __name__ == "__main__":
    """Test hypothesis executor."""
    print("=" * 60)
    print("HYPOTHESIS EXECUTOR TEST")
    print("=" * 60)
    
    executor = HypothesisExecutor()
    
    # Mock causal effects (from causal_discovery)
    from throng4.meta_policy.causal_discovery import ActionEffect
    
    causal_effects = {
        0: ActionEffect(0, 100, 0.02, [10, 11], False, 0.0, 0.01),  # Movement
        1: ActionEffect(1, 100, 0.02, [10, 11], False, 0.0, 0.01),  # Movement
        4: ActionEffect(4, 80, 0.05, [50, 51], True, 0.1, 0.05),    # Shoot
        5: ActionEffect(5, 50, 0.001, [], False, 0.0, 0.0),         # Do nothing
    }
    
    # Test 1: Defensive strategy
    print("\nTest 1: Defensive strategy")
    llm_response = "Try defensive positioning. Avoid aggressive actions and focus on survival."
    strategy = executor.parse_hypothesis(llm_response, causal_effects=causal_effects)
    print(f"  Parsed: {strategy.summary()}")
    print(f"  Type: {strategy.strategy_type}")
    print(f"  Exploration modifier: {strategy.exploration_modifier}")
    print(f"  Action bias: {strategy.action_bias}")
    
    assert strategy.strategy_type == StrategyType.DEFENSIVE
    assert strategy.action_bias[4] < 1.0  # Should avoid shooting
    
    # Test 2: Offensive strategy
    print("\nTest 2: Offensive strategy")
    llm_response = "The environment has projectile mechanics. Try offensive bursts with action 4."
    strategy = executor.parse_hypothesis(llm_response, causal_effects=causal_effects)
    print(f"  Parsed: {strategy.summary()}")
    print(f"  Action bias: {strategy.action_bias}")
    
    assert strategy.strategy_type == StrategyType.OFFENSIVE
    assert strategy.action_bias[4] > 1.0  # Should favor shooting
    
    # Test 3: Timing strategy
    print("\nTest 3: Timing strategy")
    llm_response = "Wait for synchronized entity patterns before acting."
    strategy = executor.parse_hypothesis(llm_response, causal_effects=causal_effects)
    print(f"  Parsed: {strategy.summary()}")
    print(f"  Timing delay: {strategy.timing_delay}")
    
    assert strategy.strategy_type == StrategyType.TIMING
    assert strategy.timing_delay > 0
    
    print("\n✅ Hypothesis executor test complete!")
