"""
Reward Chaser — Explore variations around rewarding actions.

When a rewarding action is detected, the reward chaser:
1. Explores variations (different states, similar actions)
2. Finds boundaries (when does reward stop?)
3. Generalizes the rule (what's the underlying pattern?)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .hypothesis import DiscoveredRule, TestResult, RuleStatus


@dataclass
class RewardChaseResult:
    """Result of chasing a reward signal."""
    original_rule: DiscoveredRule
    generalized_rule: Optional[DiscoveredRule]
    boundary_conditions: Dict[str, Any]
    n_variations_tested: int
    success_rate: float


class RewardChaser:
    """
    Chase reward signals to find underlying patterns.
    
    Strategy:
    1. When reward detected, try variations
    2. Find what's essential vs. incidental
    3. Discover boundaries and generalize
    """
    
    def __init__(
        self,
        n_variations: int = 10,
        boundary_search_steps: int = 5
    ):
        self.n_variations = n_variations
        self.boundary_search_steps = boundary_search_steps
        self.chase_count = 0
    
    def chase_reward(
        self,
        env,
        rewarding_rule: DiscoveredRule,
        env_profile
    ) -> RewardChaseResult:
        """
        Chase a rewarding action to understand the pattern.
        
        Args:
            env: Environment
            rewarding_rule: Rule that showed positive reward
            env_profile: Environment profile from analyzer
            
        Returns:
            RewardChaseResult with generalized understanding
        """
        self.chase_count += 1
        
        # Extract action from rule
        action = self._extract_action_from_rule(rewarding_rule)
        
        # Test variations
        variations = self._test_variations(env, action, env_profile)
        
        # Find boundaries
        boundaries = self._find_boundaries(env, action, variations, env_profile)
        
        # Generalize rule
        generalized = self._generalize_rule(rewarding_rule, variations, boundaries)
        
        # Compute success rate
        successes = sum(1 for v in variations if v['reward'] > 0)
        success_rate = successes / len(variations) if variations else 0.0
        
        return RewardChaseResult(
            original_rule=rewarding_rule,
            generalized_rule=generalized,
            boundary_conditions=boundaries,
            n_variations_tested=len(variations),
            success_rate=success_rate
        )
    
    def _extract_action_from_rule(self, rule: DiscoveredRule) -> int:
        """Extract action ID from rule."""
        # Parse rule ID like "reward_action_2"
        if "action_" in rule.id:
            parts = rule.id.split("_")
            for i, part in enumerate(parts):
                if part == "action" and i + 1 < len(parts):
                    try:
                        return int(parts[i + 1])
                    except ValueError:
                        pass
        return 0  # Default fallback
    
    def _test_variations(
        self,
        env,
        action: int,
        env_profile
    ) -> List[Dict[str, Any]]:
        """
        Test variations of the rewarding action.
        
        Variations include:
        - Different starting states
        - Repeated actions
        - Action sequences
        """
        variations = []
        
        for _ in range(self.n_variations):
            # Random starting state
            state = env.reset()
            
            # Try the action
            next_state, reward, done, info = env.step(action)
            
            variations.append({
                'state_before': state,
                'state_after': next_state,
                'reward': reward,
                'done': done,
                'action': action,
                'info': info
            })
            
            if done:
                continue
            
            # Try repeating the action
            state2 = next_state
            next_state2, reward2, done2, info2 = env.step(action)
            
            variations.append({
                'state_before': state2,
                'state_after': next_state2,
                'reward': reward2,
                'done': done2,
                'action': action,
                'info': info2,
                'repeated': True
            })
        
        return variations
    
    def _find_boundaries(
        self,
        env,
        action: int,
        variations: List[Dict[str, Any]],
        env_profile
    ) -> Dict[str, Any]:
        """
        Find boundaries where the reward pattern breaks.
        
        Returns:
            Dictionary of boundary conditions
        """
        boundaries = {}
        
        # Analyze state dimensions where reward occurs
        rewarding_states = [v['state_before'] for v in variations if v['reward'] > 0]
        non_rewarding_states = [v['state_before'] for v in variations if v['reward'] <= 0]
        
        if not rewarding_states or not non_rewarding_states:
            return boundaries
        
        rewarding_states = np.array(rewarding_states)
        non_rewarding_states = np.array(non_rewarding_states)
        
        # For each dimension, find range where reward occurs
        for dim in range(rewarding_states.shape[1]):
            reward_min = rewarding_states[:, dim].min()
            reward_max = rewarding_states[:, dim].max()
            
            non_reward_min = non_rewarding_states[:, dim].min()
            non_reward_max = non_rewarding_states[:, dim].max()
            
            # If ranges don't overlap much, this dim is important
            overlap = min(reward_max, non_reward_max) - max(reward_min, non_reward_min)
            total_range = max(reward_max, non_reward_max) - min(reward_min, non_reward_min)
            
            if total_range > 0:
                overlap_ratio = overlap / total_range
                
                if overlap_ratio < 0.5:  # Low overlap = important boundary
                    boundaries[f'dim_{dim}'] = {
                        'reward_range': (float(reward_min), float(reward_max)),
                        'non_reward_range': (float(non_reward_min), float(non_reward_max)),
                        'importance': 1.0 - overlap_ratio
                    }
        
        return boundaries
    
    def _generalize_rule(
        self,
        original_rule: DiscoveredRule,
        variations: List[Dict[str, Any]],
        boundaries: Dict[str, Any]
    ) -> Optional[DiscoveredRule]:
        """
        Create a generalized version of the rule based on variations.
        
        Args:
            original_rule: Original rewarding rule
            variations: Test results from variations
            boundaries: Discovered boundary conditions
            
        Returns:
            Generalized DiscoveredRule or None
        """
        if not variations:
            return None
        
        # Count successes
        successes = [v for v in variations if v['reward'] > 0]
        success_rate = len(successes) / len(variations)
        
        # If success rate is too low, rule doesn't generalize
        if success_rate < 0.3:
            return None
        
        # Create generalized rule
        generalized = DiscoveredRule(
            id=f"{original_rule.id}_generalized",
            description=f"{original_rule.description} (generalized, {success_rate:.1%} success rate)",
            feature=original_rule.feature,
            direction=original_rule.direction,
            status=RuleStatus.CONFIRMED if success_rate > 0.7 else RuleStatus.TENTATIVE,
            confidence=success_rate,
            source="reward_chase",
            conditions=boundaries
        )
        
        # Add all variation results as test results
        for var in variations:
            result = TestResult(
                action=var['action'],
                state_before=var['state_before'],
                state_after=var['state_after'],
                reward=var['reward'],
                done=var['done'],
                metadata=var.get('info', {})
            )
            success = var['reward'] > 0
            generalized.add_test_result(result, success=success)
        
        generalized.update_status()
        
        return generalized
    
    def get_summary(self) -> str:
        """Get summary of reward chasing."""
        return f"""Reward Chaser Summary:
  Total chases: {self.chase_count}
  Variations per chase: {self.n_variations}"""
