"""
Hypothesis and Rule Tracking for Policy Decomposition (v3).

Soft epistemology: nothing is eliminated, everything is archived.
- Confidence decays over time
- Anti-policies are first-class
- Stochasticity is tracked per-rule
- Cross-game context preserved
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import time
import math


class RuleStatus(Enum):
    """Status of a discovered rule — no permanent elimination."""
    ACTIVE = "active"               # Confirmed, regularly used
    TENTATIVE = "tentative"         # Assumed true, needs more data
    DORMANT = "dormant"             # Low confidence, archived but revisitable
    ANTI_POLICY = "anti_policy"     # "Do the opposite" — generated from failures
    BOUNDARY_FOUND = "boundary_found"  # Applies only under specific conditions


@dataclass
class TestResult:
    """Single test outcome for a hypothesis."""
    action: Any
    state_before: Any
    state_after: Any
    reward: float
    done: bool
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OutcomeDistribution:
    """Tracks distribution of outcomes for stochastic rules."""
    outcomes: Dict[str, int] = field(default_factory=dict)  # outcome_key -> count
    total: int = 0
    
    def record(self, outcome_key: str):
        """Record an observed outcome."""
        self.outcomes[outcome_key] = self.outcomes.get(outcome_key, 0) + 1
        self.total += 1
    
    def probability(self, outcome_key: str) -> float:
        """Get probability of a specific outcome."""
        if self.total == 0:
            return 0.0
        return self.outcomes.get(outcome_key, 0) / self.total
    
    def entropy(self) -> float:
        """Shannon entropy — higher means more random."""
        if self.total == 0:
            return 0.0
        h = 0.0
        for count in self.outcomes.values():
            p = count / self.total
            if p > 0:
                h -= p * math.log2(p)
        return h
    
    def stochasticity_score(self) -> float:
        """0.0 = deterministic, 1.0 = maximum entropy for observed outcomes."""
        if len(self.outcomes) <= 1:
            return 0.0
        max_entropy = math.log2(len(self.outcomes))
        if max_entropy == 0:
            return 0.0
        return self.entropy() / max_entropy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'outcomes': dict(self.outcomes),
            'total': self.total,
            'stochasticity': self.stochasticity_score()
        }


@dataclass
class DiscoveredRule:
    """
    A discovered game mechanic or strategy rule (v3).
    
    Nothing is ever permanently eliminated. Low-confidence rules
    become DORMANT and can be revisited. Anti-policies are auto-generated
    from consistent failures.
    """
    id: str
    description: str
    feature: str                              # What feature/dimension this affects
    direction: Optional[str] = None           # "increase", "decrease", "maximize", "minimize"
    status: RuleStatus = RuleStatus.TENTATIVE
    confidence: float = 0.0                   # 0-1, DECAYS over time
    
    # Testing history
    test_results: List[TestResult] = field(default_factory=list)
    n_tests: int = 0
    n_successes: int = 0
    n_failures: int = 0
    
    # Confidence tracking
    confidence_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, confidence)
    last_tested: float = field(default_factory=time.time)
    decay_rate: float = 0.01                  # Confidence lost per hour of inactivity
    
    # Stochasticity
    stochasticity: float = 0.0                # 0=deterministic, 1=fully random
    outcome_distribution: OutcomeDistribution = field(default_factory=OutcomeDistribution)
    
    # Anti-policy link
    anti_policy_id: Optional[str] = None      # ID of auto-generated anti-policy
    parent_rule_id: Optional[str] = None      # If this IS an anti-policy, link to parent
    
    # Context
    conditions: Dict[str, Any] = field(default_factory=dict)
    source: str = "micro_test"
    discovered_at: float = field(default_factory=time.time)
    environment_context: str = ""             # Which game/level
    transferable: bool = False                # Could apply to other games?
    
    # Boundaries
    boundary_min: Optional[float] = None
    boundary_max: Optional[float] = None
    
    def add_test_result(self, result: TestResult, success: bool):
        """Record a test and update confidence."""
        self.test_results.append(result)
        self.n_tests += 1
        self.last_tested = time.time()
        
        if success:
            self.n_successes += 1
        else:
            self.n_failures += 1
        
        # Update confidence (exponential moving average for stability)
        # Floor at 0.01 — nothing is ever fully zero (v3 soft epistemology)
        if self.n_tests == 1:
            self.confidence = max(0.01, 1.0 if success else 0.0)
        else:
            alpha = min(0.3, 2.0 / (self.n_tests + 1))  # Decaying learning rate
            raw = (1 - alpha) * self.confidence + alpha * (1.0 if success else 0.0)
            self.confidence = max(0.01, raw)
        
        self.confidence_history.append((time.time(), self.confidence))
        
        # Track outcome distribution
        outcome_key = "success" if success else "failure"
        self.outcome_distribution.record(outcome_key)
    
    def apply_confidence_decay(self, current_time: Optional[float] = None):
        """Decay confidence based on time since last test."""
        now = current_time or time.time()
        hours_since_test = (now - self.last_tested) / 3600.0
        
        if hours_since_test > 0:
            decay = self.decay_rate * hours_since_test
            self.confidence = max(0.01, self.confidence - decay)  # Never fully zero
    
    def update_status(self):
        """Update rule status with soft epistemology — no permanent elimination."""
        if self.n_tests < 3:
            self.status = RuleStatus.TENTATIVE
            return
        
        # High confidence = ACTIVE
        if self.confidence > 0.7:
            self.status = RuleStatus.ACTIVE
        # Low confidence = DORMANT (not eliminated!)
        elif self.confidence < 0.15:
            self.status = RuleStatus.DORMANT
        # Medium confidence with inconsistent results = possible boundary
        elif 0.3 < self.confidence < 0.7 and self.n_tests > 10:
            self.status = RuleStatus.BOUNDARY_FOUND
        # Otherwise stays TENTATIVE
        else:
            self.status = RuleStatus.TENTATIVE
    
    def generate_anti_policy(self) -> 'DiscoveredRule':
        """Generate an anti-policy from this rule's failures."""
        # Invert the direction
        inverted_direction = {
            "increase": "decrease", "decrease": "increase",
            "maximize": "minimize", "minimize": "maximize"
        }.get(self.direction, self.direction)
        
        anti = DiscoveredRule(
            id=f"{self.id}_anti",
            description=f"ANTI: {self.description} (do the opposite)",
            feature=self.feature,
            direction=inverted_direction,
            status=RuleStatus.ANTI_POLICY,
            confidence=1.0 - self.confidence,  # Inverse confidence
            source="anti_policy",
            parent_rule_id=self.id,
            environment_context=self.environment_context,
            conditions=self.conditions.copy()
        )
        
        self.anti_policy_id = anti.id
        return anti
    
    def should_revisit(self, staleness_hours: float = 24.0) -> bool:
        """Check if this rule needs re-evaluation."""
        hours_since_test = (time.time() - self.last_tested) / 3600.0
        
        # Dormant rules with some original confidence
        if self.status == RuleStatus.DORMANT and hours_since_test > staleness_hours:
            return True
        
        # Boundary rules that might have shifted
        if self.status == RuleStatus.BOUNDARY_FOUND and hours_since_test > staleness_hours / 2:
            return True
        
        # Active rules that haven't been tested in a while
        if self.status == RuleStatus.ACTIVE and hours_since_test > staleness_hours * 2:
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'id': self.id,
            'description': self.description,
            'feature': self.feature,
            'direction': self.direction,
            'status': self.status.value,
            'confidence': self.confidence,
            'n_tests': self.n_tests,
            'n_successes': self.n_successes,
            'n_failures': self.n_failures,
            'stochasticity': self.stochasticity,
            'outcome_distribution': self.outcome_distribution.to_dict(),
            'anti_policy_id': self.anti_policy_id,
            'parent_rule_id': self.parent_rule_id,
            'conditions': self.conditions,
            'source': self.source,
            'discovered_at': self.discovered_at,
            'last_tested': self.last_tested,
            'decay_rate': self.decay_rate,
            'environment_context': self.environment_context,
            'transferable': self.transferable,
            'boundary_min': self.boundary_min,
            'boundary_max': self.boundary_max
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveredRule':
        """Deserialize from dictionary."""
        rule = cls(
            id=data['id'],
            description=data['description'],
            feature=data['feature'],
            direction=data.get('direction'),
            status=RuleStatus(data['status']),
            confidence=data['confidence'],
            n_tests=data['n_tests'],
            n_successes=data['n_successes'],
            n_failures=data['n_failures'],
            stochasticity=data.get('stochasticity', 0.0),
            anti_policy_id=data.get('anti_policy_id'),
            parent_rule_id=data.get('parent_rule_id'),
            conditions=data.get('conditions', {}),
            source=data.get('source', 'micro_test'),
            discovered_at=data.get('discovered_at', time.time()),
            last_tested=data.get('last_tested', time.time()),
            decay_rate=data.get('decay_rate', 0.01),
            environment_context=data.get('environment_context', ''),
            transferable=data.get('transferable', False),
            boundary_min=data.get('boundary_min'),
            boundary_max=data.get('boundary_max')
        )
        return rule


class RuleLibrary:
    """Collection of discovered rules — nothing is ever deleted."""
    
    def __init__(self):
        self.rules: Dict[str, DiscoveredRule] = {}
    
    def add_rule(self, rule: DiscoveredRule):
        """Add or update a rule."""
        self.rules[rule.id] = rule
    
    def add_with_anti_policy(self, rule: DiscoveredRule):
        """Add a rule and auto-generate its anti-policy if confidence is low."""
        self.rules[rule.id] = rule
        
        if rule.confidence < 0.3 and rule.n_tests >= 3 and not rule.anti_policy_id:
            anti = rule.generate_anti_policy()
            self.rules[anti.id] = anti
    
    def get_active_rules(self) -> List[DiscoveredRule]:
        """Get all active (high-confidence) rules."""
        return [r for r in self.rules.values() if r.status == RuleStatus.ACTIVE]
    
    def get_anti_policies(self) -> List[DiscoveredRule]:
        """Get all anti-policies."""
        return [r for r in self.rules.values() if r.status == RuleStatus.ANTI_POLICY]
    
    def get_dormant_rules(self) -> List[DiscoveredRule]:
        """Get all dormant rules (candidates for re-evaluation)."""
        return [r for r in self.rules.values() if r.status == RuleStatus.DORMANT]
    
    def get_rules_needing_revisit(self) -> List[DiscoveredRule]:
        """Get rules that should be re-evaluated."""
        return [r for r in self.rules.values() if r.should_revisit()]
    
    def get_rules_by_feature(self, feature: str) -> List[DiscoveredRule]:
        """Get all rules affecting a specific feature."""
        return [r for r in self.rules.values() if r.feature == feature]
    
    def get_rules_by_status(self, status: RuleStatus) -> List[DiscoveredRule]:
        """Get all rules with a specific status."""
        return [r for r in self.rules.values() if r.status == status]
    
    def get_rules_for_env(self, env_context: str) -> List[DiscoveredRule]:
        """Get all rules from a specific environment."""
        return [r for r in self.rules.values() if r.environment_context == env_context]
    
    def get_transferable_rules(self) -> List[DiscoveredRule]:
        """Get rules marked as potentially transferable to other games."""
        return [r for r in self.rules.values() if r.transferable]
    
    def apply_decay(self, current_time: Optional[float] = None):
        """Apply confidence decay to all rules and update statuses."""
        for rule in self.rules.values():
            rule.apply_confidence_decay(current_time)
            rule.update_status()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize library to dictionary."""
        return {
            'rules': {rid: r.to_dict() for rid, r in self.rules.items()}
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        active = len(self.get_active_rules())
        tentative = len(self.get_rules_by_status(RuleStatus.TENTATIVE))
        dormant = len(self.get_dormant_rules())
        anti = len(self.get_anti_policies())
        boundary = len(self.get_rules_by_status(RuleStatus.BOUNDARY_FOUND))
        
        return f"""Rule Library Summary:
  Active: {active}
  Tentative: {tentative}
  Dormant: {dormant}
  Anti-policies: {anti}
  Boundary: {boundary}
  Total: {len(self.rules)}"""
