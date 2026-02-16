"""
Causal Discovery — Track action→effect relationships.

Discovers what each action does by observing state changes:
- Which actions cause movement
- Which actions create entities (projectiles, shields)
- Which state features each action affects
- Reward correlation per action

This enables the LLM to reason about action mechanics without being told.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ActionEffect:
    """Discovered effects of a single action."""
    action_id: int
    sample_count: int
    avg_state_delta: float  # How much state changes
    affected_features: List[int]  # Which RAM bytes change
    creates_entities: bool  # Does this spawn new objects?
    avg_reward: float  # Average reward when taking this action
    reward_variance: float  # Variance in reward
    
    def summary(self) -> str:
        """Human-readable summary."""
        effects = []
        effects.append(f"state_delta={self.avg_state_delta:.3f}")
        
        if self.creates_entities:
            effects.append("creates_entities")
        
        if abs(self.avg_reward) > 0.01:
            effects.append(f"avg_reward={self.avg_reward:.2f}")
        
        if len(self.affected_features) > 0:
            effects.append(f"affects_{len(self.affected_features)}_features")
        
        return f"Action {self.action_id}: {', '.join(effects)}"


class CausalDiscovery:
    """
    Discover action→effect relationships from experience.
    
    Tracks transitions and identifies:
    - Movement actions (large state delta)
    - Entity creation actions (new active features appear)
    - Reward-correlated actions
    """
    
    def __init__(self, entity_creation_threshold: float = 0.1):
        self.entity_creation_threshold = entity_creation_threshold
        self.transitions: List[Dict] = []
    
    def record_transition(self, state: np.ndarray, action: int, 
                          reward: float, next_state: np.ndarray):
        """Record a single transition for analysis."""
        self.transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
        })
    
    def discover_action_effects(self, 
                                 transitions: List[Dict] = None) -> Dict[int, ActionEffect]:
        """
        Analyze transitions to discover what each action does.
        
        Args:
            transitions: Optional list of transitions (uses recorded if None)
            
        Returns:
            Dict mapping action_id → ActionEffect
        """
        if transitions is None:
            transitions = self.transitions
        
        if len(transitions) == 0:
            return {}
        
        # Group transitions by action
        action_groups = defaultdict(list)
        for t in transitions:
            action_groups[t['action']].append(t)
        
        # Analyze each action
        effects = {}
        for action_id, group in action_groups.items():
            effects[action_id] = self._analyze_action(action_id, group)
        
        return effects
    
    def _analyze_action(self, action_id: int, 
                        transitions: List[Dict]) -> ActionEffect:
        """Analyze all transitions for a single action."""
        states = np.array([t['state'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions])
        
        # Compute state deltas
        deltas = next_states - states
        avg_delta = np.mean(np.abs(deltas))
        
        # Find affected features (features that change consistently)
        feature_changes = np.abs(deltas)
        avg_feature_change = np.mean(feature_changes, axis=0)
        affected = np.where(avg_feature_change > 0.01)[0].tolist()
        
        # Detect entity creation
        # (new features become active that weren't before)
        creates_entities = self._detect_entity_creation(states, next_states)
        
        # Reward statistics
        avg_reward = np.mean(rewards)
        reward_var = np.var(rewards)
        
        return ActionEffect(
            action_id=action_id,
            sample_count=len(transitions),
            avg_state_delta=float(avg_delta),
            affected_features=affected,
            creates_entities=creates_entities,
            avg_reward=float(avg_reward),
            reward_variance=float(reward_var),
        )
    
    def _detect_entity_creation(self, states: np.ndarray, 
                                 next_states: np.ndarray) -> bool:
        """
        Detect if this action creates new entities.
        
        Strategy: check if new features become active in next_state
        that were inactive in state.
        """
        # Count active features before and after
        active_before = np.sum(np.abs(states) > 0.01, axis=1)
        active_after = np.sum(np.abs(next_states) > 0.01, axis=1)
        
        # Entity creation = consistent increase in active features
        increases = active_after - active_before
        avg_increase = np.mean(increases)
        
        return avg_increase > self.entity_creation_threshold
    
    def get_summary(self, effects: Dict[int, ActionEffect]) -> str:
        """Generate human-readable summary for LLM prompts."""
        if not effects:
            return "Causal Discovery: No actions analyzed yet"
        
        lines = ["Causal Discovery (action effects):"]
        
        # Sort by state delta (movement actions first)
        sorted_effects = sorted(effects.values(), 
                                key=lambda e: e.avg_state_delta, 
                                reverse=True)
        
        for effect in sorted_effects[:10]:  # Top 10 most impactful
            lines.append(f"  - {effect.summary()}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear recorded transitions."""
        self.transitions.clear()


if __name__ == "__main__":
    """Test causal discovery on synthetic data."""
    print("=" * 60)
    print("CAUSAL DISCOVERY TEST")
    print("=" * 60)
    
    discovery = CausalDiscovery()
    
    # Simulate a simple game with 6 actions:
    # 0-1: move left/right (large state delta)
    # 2-3: move up/down (large state delta)
    # 4: shoot (creates entities)
    # 5: do nothing (small delta)
    
    print("\nSimulating 500 transitions...")
    for _ in range(500):
        state = np.random.rand(128) * 0.5
        action = np.random.randint(0, 6)
        
        # Simulate effects
        next_state = state.copy()
        reward = 0.0
        
        if action == 0:  # Move left
            next_state[10:20] += 0.3
            reward = 0.0
        elif action == 1:  # Move right
            next_state[10:20] -= 0.3
            reward = 0.0
        elif action == 2:  # Move up
            next_state[30:40] += 0.3
            reward = 0.0
        elif action == 3:  # Move down
            next_state[30:40] -= 0.3
            reward = 0.0
        elif action == 4:  # Shoot (creates projectile)
            next_state[50:60] = 1.0  # New entity appears
            reward = 0.1
        else:  # Do nothing
            next_state += np.random.randn(128) * 0.01
            reward = 0.0
        
        discovery.record_transition(state, action, reward, next_state)
    
    # Discover effects
    effects = discovery.discover_action_effects()
    
    print("\n" + discovery.get_summary(effects))
    
    # Verify entity creation detection
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)
    
    for action_id in range(6):
        if action_id in effects:
            e = effects[action_id]
            print(f"\nAction {action_id}:")
            print(f"  Creates entities: {e.creates_entities}")
            print(f"  State delta: {e.avg_state_delta:.3f}")
            print(f"  Avg reward: {e.avg_reward:.3f}")
    
    # Verify action 4 is detected as entity creator
    assert effects[4].creates_entities, "Action 4 should create entities!"
    assert not effects[5].creates_entities, "Action 5 should NOT create entities!"
    
    print("\n✅ Causal discovery test complete!")
