"""
Micro-Test Engine — Within-episode single-trial probing (v3).

Runs quick, targeted tests to assess immediate effects of actions:
- Rewarding (positive reward signal)
- Catastrophic (episode ends) -> generates anti-policy
- Neutral (no significant effect)

v3 changes:
- Catastrophic events generate anti-policies instead of CatastrophicRule
- Uses add_with_anti_policy for low-confidence rules
- Tracks environment context on all rules
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from .hypothesis import (
    DiscoveredRule, TestResult, 
    RuleStatus, RuleLibrary
)


@dataclass
class ProbeResult:
    """Result of a single micro-test probe."""
    action: Any
    state_before: np.ndarray
    state_after: np.ndarray
    reward: float
    done: bool
    classification: str  # "rewarding", "catastrophic", "neutral"
    state_delta: np.ndarray  # change in state
    info: Dict[str, Any]


class MicroTester:
    """
    Performs within-episode single-trial probes.
    
    Strategy:
    1. Try each action once from current state
    2. Classify immediate effect (rewarding/catastrophic/neutral)
    3. Generate hypotheses about what each action does
    4. Catastrophic -> auto-generates anti-policy (v3)
    """
    
    def __init__(
        self,
        reward_threshold: float = 0.01,
        catastrophic_threshold: float = -0.5,
        environment_context: str = ""
    ):
        self.reward_threshold = reward_threshold
        self.catastrophic_threshold = catastrophic_threshold
        self.environment_context = environment_context
        self.rule_library = RuleLibrary()
        self.probe_count = 0
    
    def probe_all_actions(
        self, 
        env, 
        n_actions: int,
        max_probes_per_action: int = 3
    ) -> List[ProbeResult]:
        """Probe all actions from current state."""
        results = []
        
        for action in range(n_actions):
            action_results = []
            
            for _ in range(max_probes_per_action):
                state_before = env.reset()
                state_after, reward, done, info = env.step(action)
                state_delta = state_after - state_before
                classification = self._classify_effect(reward, done)
                
                probe = ProbeResult(
                    action=action,
                    state_before=state_before,
                    state_after=state_after,
                    reward=reward,
                    done=done,
                    classification=classification,
                    state_delta=state_delta,
                    info=info
                )
                
                action_results.append(probe)
                self.probe_count += 1
                
                if classification == "catastrophic":
                    break
            
            results.extend(action_results)
        
        return results
    
    def probe_single_action(
        self,
        env,
        action: int,
        state: Optional[np.ndarray] = None
    ) -> ProbeResult:
        """Probe a single action from given state."""
        if state is not None:
            if hasattr(env, 'set_state'):
                env.set_state(state)
            state_before = state
        else:
            state_before = env.reset()
        
        state_after, reward, done, info = env.step(action)
        state_delta = state_after - state_before
        classification = self._classify_effect(reward, done)
        
        self.probe_count += 1
        
        return ProbeResult(
            action=action,
            state_before=state_before,
            state_after=state_after,
            reward=reward,
            done=done,
            classification=classification,
            state_delta=state_delta,
            info=info
        )
    
    def _classify_effect(self, reward: float, done: bool) -> str:
        """Classify the immediate effect of an action."""
        if done and reward < self.catastrophic_threshold:
            return "catastrophic"
        elif reward > self.reward_threshold:
            return "rewarding"
        else:
            return "neutral"
    
    def generate_hypotheses_from_probes(
        self,
        probes: List[ProbeResult],
        env_profile
    ) -> List[DiscoveredRule]:
        """Generate hypotheses from probe results (v3: anti-policies for catastrophic)."""
        hypotheses = []
        
        # Group probes by action
        probes_by_action = {}
        for probe in probes:
            if probe.action not in probes_by_action:
                probes_by_action[probe.action] = []
            probes_by_action[probe.action].append(probe)
        
        for action, action_probes in probes_by_action.items():
            # Check for catastrophic effects -> anti-policy
            catastrophic_probes = [p for p in action_probes if p.classification == "catastrophic"]
            if catastrophic_probes:
                rule = self._create_catastrophic_anti_policy(action, catastrophic_probes)
                hypotheses.append(rule)
                self.rule_library.add_rule(rule)
            
            # Check for rewarding effects
            rewarding_probes = [p for p in action_probes if p.classification == "rewarding"]
            if rewarding_probes:
                rules = self._create_reward_rules(action, rewarding_probes, env_profile)
                for rule in rules:
                    hypotheses.append(rule)
                    self.rule_library.add_rule(rule)
            
            # State change rules (even if neutral reward)
            state_change_rules = self._create_state_change_rules(action, action_probes, env_profile)
            for rule in state_change_rules:
                hypotheses.append(rule)
                # Use add_with_anti_policy for low-confidence state changes
                self.rule_library.add_with_anti_policy(rule)
        
        return hypotheses
    
    def _create_catastrophic_anti_policy(
        self,
        action: int,
        probes: List[ProbeResult]
    ) -> DiscoveredRule:
        """Create an anti-policy from catastrophic probes (v3: no more CatastrophicRule)."""
        avg_reward = np.mean([p.reward for p in probes])
        
        rule = DiscoveredRule(
            id=f"avoid_action_{action}",
            description=f"AVOID action {action} — causes termination (avg reward: {avg_reward:.3f})",
            feature="survival",
            direction="maximize",
            status=RuleStatus.ANTI_POLICY,
            confidence=0.95,  # High confidence after observing catastrophe
            source="micro_test",
            environment_context=self.environment_context
        )
        
        for probe in probes[:5]:
            result = TestResult(
                action=probe.action,
                state_before=probe.state_before,
                state_after=probe.state_after,
                reward=probe.reward,
                done=probe.done
            )
            rule.add_test_result(result, success=False)
        
        return rule
    
    def _create_reward_rules(
        self,
        action: int,
        probes: List[ProbeResult],
        env_profile
    ) -> List[DiscoveredRule]:
        """Create rules for rewarding actions."""
        rules = []
        avg_reward = np.mean([p.reward for p in probes])
        
        rule = DiscoveredRule(
            id=f"reward_action_{action}",
            description=f"Action {action} yields positive reward (avg: {avg_reward:.3f})",
            feature="reward",
            direction="maximize",
            status=RuleStatus.TENTATIVE,
            confidence=0.0,
            source="micro_test",
            environment_context=self.environment_context
        )
        
        for probe in probes:
            result = TestResult(
                action=probe.action,
                state_before=probe.state_before,
                state_after=probe.state_after,
                reward=probe.reward,
                done=probe.done
            )
            rule.add_test_result(result, success=True)
        
        rule.update_status()
        rules.append(rule)
        
        return rules
    
    def _create_state_change_rules(
        self,
        action: int,
        probes: List[ProbeResult],
        env_profile
    ) -> List[DiscoveredRule]:
        """Create rules for state changes caused by actions."""
        rules = []
        
        deltas = np.array([p.state_delta for p in probes])
        avg_delta = deltas.mean(axis=0)
        
        for dim in range(len(avg_delta)):
            if abs(avg_delta[dim]) > 0.01:
                direction = "increase" if avg_delta[dim] > 0 else "decrease"
                
                rule = DiscoveredRule(
                    id=f"action_{action}_affects_dim_{dim}",
                    description=f"Action {action} causes dim {dim} to {direction} by {abs(avg_delta[dim]):.3f}",
                    feature=f"state_dim_{dim}",
                    direction=direction,
                    status=RuleStatus.TENTATIVE,
                    confidence=0.0,
                    source="micro_test",
                    environment_context=self.environment_context
                )
                
                for probe in probes:
                    result = TestResult(
                        action=probe.action,
                        state_before=probe.state_before,
                        state_after=probe.state_after,
                        reward=probe.reward,
                        done=probe.done
                    )
                    expected_positive = (direction == "increase")
                    actual_positive = (probe.state_delta[dim] > 0)
                    success = (expected_positive == actual_positive)
                    rule.add_test_result(result, success=success)
                
                rule.update_status()
                
                if rule.confidence > 0.5:
                    rules.append(rule)
        
        return rules
    
    def get_summary(self) -> str:
        """Get summary of micro-testing results."""
        return f"""Micro-Tester Summary:
  Total probes: {self.probe_count}
  {self.rule_library.summary()}"""
