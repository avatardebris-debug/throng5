"""
Attribution Diagnoser — Determine WHY an observation occurred.

When conflicting observations are seen, diagnose the cause:
1. Is it RNG? (Same state + same action -> different outcomes)
2. Is it state-dependent? (Different states -> different outcomes)
3. Is it time-dependent? (Same state + same action but at different times)
4. Is it an environment property? (Fundamental game mechanic)
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from .hypothesis import DiscoveredRule, TestResult, OutcomeDistribution


class Attribution(Enum):
    """What caused the observed behavior."""
    DETERMINISTIC = "deterministic"           # Same input -> same output always
    STOCHASTIC = "stochastic"                 # Same input -> random output
    STATE_DEPENDENT = "state_dependent"       # Depends on specific state values
    TIME_DEPENDENT = "time_dependent"         # Depends on step count or history
    ENVIRONMENT_PROPERTY = "environment"      # Fundamental game mechanic
    UNKNOWN = "unknown"                       # Not enough data


@dataclass
class AttributionResult:
    """Result of an attribution diagnosis."""
    attribution: Attribution
    confidence: float                         # How confident in this diagnosis
    stochasticity_score: float                # 0=deterministic, 1=fully random
    outcome_distribution: OutcomeDistribution
    details: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        dist_str = ", ".join(
            f"{k}: {v}/{self.outcome_distribution.total}"
            for k, v in self.outcome_distribution.outcomes.items()
        )
        return (
            f"Attribution: {self.attribution.value} "
            f"(confidence={self.confidence:.2f}, "
            f"stochasticity={self.stochasticity_score:.2f})\n"
            f"  Distribution: {dist_str}"
        )


class AttributionDiagnoser:
    """
    Diagnose WHY an observation occurred before labeling WHAT happened.
    
    Follows the attribution hierarchy:
    1. RNG? (repeat same action from same state)
    2. State-dependent? (same action, different states)
    3. Time-dependent? (same action/state, different episode steps)
    4. Environment-level? (fundamental mechanic)
    """
    
    def __init__(
        self,
        n_rng_trials: int = 20,
        n_state_trials: int = 10,
        stochastic_threshold: float = 0.15
    ):
        """
        Args:
            n_rng_trials: How many times to repeat same action for RNG test
            n_state_trials: How many different states to test
            stochastic_threshold: Min stochasticity score to classify as stochastic
        """
        self.n_rng_trials = n_rng_trials
        self.n_state_trials = n_state_trials
        self.stochastic_threshold = stochastic_threshold
    
    def diagnose(
        self,
        env,
        action: int,
        conflicting_results: Optional[List[TestResult]] = None
    ) -> AttributionResult:
        """
        Full attribution diagnosis for an action.
        
        Args:
            env: Environment to test
            action: Action to diagnose
            conflicting_results: Optional prior conflicting observations
            
        Returns:
            AttributionResult with diagnosis
        """
        # Step 1: Test for RNG
        rng_result = self.test_rng(env, action)
        
        if rng_result.stochasticity_score > self.stochastic_threshold:
            return rng_result
        
        # Step 2: Test for state-dependence
        state_result = self.test_state_dependence(env, action)
        
        if state_result.attribution == Attribution.STATE_DEPENDENT:
            return state_result
        
        # Step 3: If neither RNG nor state-dependent, it's deterministic
        return AttributionResult(
            attribution=Attribution.DETERMINISTIC,
            confidence=0.9,
            stochasticity_score=rng_result.stochasticity_score,
            outcome_distribution=rng_result.outcome_distribution,
            details={'rng_test': rng_result.details, 'state_test': state_result.details}
        )
    
    def test_rng(self, env, action: int) -> AttributionResult:
        """
        Test if an action has random outcomes.
        
        Repeats the same action from the same starting state many times.
        If outcomes differ, the action is stochastic.
        """
        distribution = OutcomeDistribution()
        
        for _ in range(self.n_rng_trials):
            # Always start from the same state (reset)
            state = env.reset()
            next_state, reward, done, info = env.step(action)
            
            # Create outcome key from state change + reward
            delta = np.array(next_state).flatten() - np.array(state).flatten()
            # Quantize to handle floating point noise
            delta_key = tuple(np.round(delta, 4))
            outcome_key = f"d={delta_key}_r={reward:.3f}_done={done}"
            
            distribution.record(outcome_key)
        
        stoch_score = distribution.stochasticity_score()
        
        if stoch_score > self.stochastic_threshold:
            attribution = Attribution.STOCHASTIC
            confidence = min(0.95, stoch_score + 0.3)
        else:
            attribution = Attribution.DETERMINISTIC
            confidence = 1.0 - stoch_score
        
        return AttributionResult(
            attribution=attribution,
            confidence=confidence,
            stochasticity_score=stoch_score,
            outcome_distribution=distribution,
            details={
                'n_trials': self.n_rng_trials,
                'n_unique_outcomes': len(distribution.outcomes),
                'test_type': 'rng'
            }
        )
    
    def test_state_dependence(self, env, action: int) -> AttributionResult:
        """
        Test if action outcome depends on state.
        
        Tries the same action from different starting states. If outcomes
        differ consistently by state (not randomly), it's state-dependent.
        """
        state_outcomes = {}  # state_key -> list of outcome_keys
        distribution = OutcomeDistribution()
        
        for _ in range(self.n_state_trials):
            state = env.reset()
            next_state, reward, done, info = env.step(action)
            
            state_key = tuple(np.round(np.array(state).flatten(), 4))
            delta = np.array(next_state).flatten() - np.array(state).flatten()
            delta_key = tuple(np.round(delta, 4))
            outcome_key = f"d={delta_key}_r={reward:.3f}"
            
            if state_key not in state_outcomes:
                state_outcomes[state_key] = []
            state_outcomes[state_key].append(outcome_key)
            distribution.record(outcome_key)
        
        # Analyze: do different states produce different outcomes?
        unique_outcomes_per_state = {
            sk: len(set(oks)) for sk, oks in state_outcomes.items()
        }
        
        # If all states produce same outcome -> not state-dependent
        all_outcomes = set()
        for oks in state_outcomes.values():
            all_outcomes.update(oks)
        
        if len(all_outcomes) <= 1:
            return AttributionResult(
                attribution=Attribution.DETERMINISTIC,
                confidence=0.8,
                stochasticity_score=0.0,
                outcome_distribution=distribution,
                details={
                    'n_states_tested': len(state_outcomes),
                    'n_unique_outcomes': len(all_outcomes),
                    'test_type': 'state_dependence'
                }
            )
        
        # Check if different states consistently give different outcomes
        # vs just random variation
        consistent_per_state = all(
            len(set(oks)) == 1 for oks in state_outcomes.values() 
            if len(oks) > 1
        )
        
        if consistent_per_state and len(all_outcomes) > 1:
            return AttributionResult(
                attribution=Attribution.STATE_DEPENDENT,
                confidence=0.85,
                stochasticity_score=0.0,
                outcome_distribution=distribution,
                details={
                    'n_states_tested': len(state_outcomes),
                    'n_unique_outcomes': len(all_outcomes),
                    'test_type': 'state_dependence',
                    'consistent_per_state': True
                }
            )
        
        # Mixed: some states have variable outcomes
        return AttributionResult(
            attribution=Attribution.UNKNOWN,
            confidence=0.5,
            stochasticity_score=distribution.stochasticity_score(),
            outcome_distribution=distribution,
            details={
                'n_states_tested': len(state_outcomes),
                'n_unique_outcomes': len(all_outcomes),
                'test_type': 'state_dependence',
                'consistent_per_state': False
            }
        )
    
    def estimate_stochasticity(self, env, action: int) -> float:
        """Quick stochasticity estimate for a specific action."""
        result = self.test_rng(env, action)
        return result.stochasticity_score
    
    def diagnose_all_actions(self, env, n_actions: int) -> Dict[int, AttributionResult]:
        """Diagnose all actions in an environment."""
        results = {}
        for action in range(n_actions):
            results[action] = self.diagnose(env, action)
        return results
