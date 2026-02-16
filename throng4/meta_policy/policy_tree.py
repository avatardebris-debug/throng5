"""
Policy Tree — Manages policy lifecycle through fingerprint-based matching.

No game names. Policies are identified by environment fingerprints.
Supports branching (inherit weights from similar environment's policy),
retiring (mark underperformers, keep concepts), and matching (find
closest fingerprint in the tree).
"""

import numpy as np
import json
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from throng4.meta_policy.env_fingerprint import EnvironmentFingerprint


@dataclass
class PerformanceTracker:
    """Track policy performance over time."""
    episode_rewards: List[float] = field(default_factory=list)
    best_reward: float = float('-inf')
    convergence_episode: Optional[int] = None
    plateau_count: int = 0
    total_episodes: int = 0
    
    def update(self, reward: float):
        self.episode_rewards.append(reward)
        self.total_episodes += 1
        if reward > self.best_reward:
            self.best_reward = reward
    
    @property
    def avg_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        # Last 25 episodes
        window = self.episode_rewards[-25:]
        return float(np.mean(window))
    
    @property
    def is_improving(self) -> bool:
        if len(self.episode_rewards) < 20:
            return True  # too early to tell
        first_half = np.mean(self.episode_rewards[:len(self.episode_rewards)//2])
        second_half = np.mean(self.episode_rewards[len(self.episode_rewards)//2:])
        return second_half > first_half * 1.05  # 5% improvement threshold
    
    def to_dict(self) -> dict:
        return {
            'total_episodes': self.total_episodes,
            'best_reward': self.best_reward,
            'avg_reward': self.avg_reward,
            'convergence_episode': self.convergence_episode,
            'plateau_count': self.plateau_count,
        }


@dataclass
class PolicyNode:
    """A single policy in the tree."""
    id: str
    parent_id: Optional[str]
    fingerprint: EnvironmentFingerprint
    weights: Optional[dict]  # Neural network weights
    performance: PerformanceTracker
    discovered_concepts: List[str]  # IDs of concepts discovered while using this policy
    status: str  # 'active', 'candidate', 'retired'
    created_at: float
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'status': self.status,
            'created_at': self.created_at,
            'performance': self.performance.to_dict(),
            'discovered_concepts': self.discovered_concepts,
            'fingerprint_summary': self.fingerprint.summary(),
        }


class PolicyTree:
    """
    Manages a tree of policies, matched by environment fingerprint.
    
    No game names — policies are identified and matched purely
    by their environment fingerprints (state distribution, reward
    structure, action sensitivity, temporal dynamics).
    """
    
    SIMILARITY_THRESHOLD = 0.85  # Above this = "same type of environment"
    BRANCH_THRESHOLD = 0.60      # Above this = "similar enough to branch from"
    RETIRE_AFTER_PLATEAUS = 5    # Retire if plateaued this many times
    
    def __init__(self):
        self.nodes: Dict[str, PolicyNode] = {}
        self.root_ids: List[str] = []  # Policies with no parent
    
    def find_best_match(self, fingerprint: EnvironmentFingerprint) -> Optional[PolicyNode]:
        """
        Find the most similar existing policy by fingerprint distance.
        
        Returns None if nothing is similar enough.
        """
        best_sim = -1.0
        best_node = None
        
        for node in self.nodes.values():
            if node.status == 'retired':
                continue
            
            sim = fingerprint.similarity(node.fingerprint)
            if sim > best_sim:
                best_sim = sim
                best_node = node
        
        if best_sim >= self.BRANCH_THRESHOLD and best_node is not None:
            return best_node
        
        return None
    
    def create_root(self, fingerprint: EnvironmentFingerprint, weights: dict = None) -> PolicyNode:
        """Create a new root policy (no parent — truly novel environment)."""
        node = PolicyNode(
            id=str(uuid.uuid4())[:8],
            parent_id=None,
            fingerprint=fingerprint,
            weights=weights,
            performance=PerformanceTracker(),
            discovered_concepts=[],
            status='active',
            created_at=time.time(),
        )
        
        self.nodes[node.id] = node
        self.root_ids.append(node.id)
        
        print(f"[PolicyTree] Created ROOT policy {node.id} "
              f"(state_dim={fingerprint.state_dim}, "
              f"actions={fingerprint.action_count})")
        
        return node
    
    def branch(self, parent_id: str, fingerprint: EnvironmentFingerprint) -> PolicyNode:
        """
        Create a child policy inheriting parent's weights.
        
        The child starts with the parent's learned weights but will
        diverge as it trains on its own environment.
        """
        parent = self.nodes[parent_id]
        
        node = PolicyNode(
            id=str(uuid.uuid4())[:8],
            parent_id=parent_id,
            fingerprint=fingerprint,
            weights=parent.weights.copy() if parent.weights else None,
            performance=PerformanceTracker(),
            discovered_concepts=[],
            status='candidate',
            created_at=time.time(),
        )
        
        self.nodes[node.id] = node
        
        sim = fingerprint.similarity(parent.fingerprint)
        print(f"[PolicyTree] BRANCHED policy {node.id} from {parent_id} "
              f"(similarity={sim:.3f})")
        
        return node
    
    def promote(self, node_id: str):
        """Promote a candidate to active (it's performing well)."""
        if node_id in self.nodes:
            self.nodes[node_id].status = 'active'
            print(f"[PolicyTree] Promoted {node_id} to ACTIVE")
    
    def retire(self, node_id: str):
        """
        Retire a policy (preserve its concepts).
        
        Retired policies can't be matched to but their discovered
        concepts remain in the concept library.
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = 'retired'
            print(f"[PolicyTree] Retired {node_id} "
                  f"(avg_reward={node.performance.avg_reward:.1f}, "
                  f"concepts={len(node.discovered_concepts)})")
    
    def should_retire(self, node_id: str) -> bool:
        """Check if a policy should be retired due to poor performance."""
        node = self.nodes.get(node_id)
        if not node:
            return False
        
        return (
            node.performance.plateau_count >= self.RETIRE_AFTER_PLATEAUS
            and not node.performance.is_improving
        )
    
    def update_weights(self, node_id: str, weights: dict):
        """Store updated weights for a policy."""
        if node_id in self.nodes:
            self.nodes[node_id].weights = weights
    
    def get_lineage(self, node_id: str) -> List[str]:
        """Get the chain of parent IDs back to root."""
        lineage = [node_id]
        current = self.nodes.get(node_id)
        
        while current and current.parent_id:
            lineage.append(current.parent_id)
            current = self.nodes.get(current.parent_id)
        
        return list(reversed(lineage))
    
    def summary(self) -> str:
        """Printable tree summary."""
        lines = ["PolicyTree:"]
        lines.append(f"  Total policies: {len(self.nodes)}")
        lines.append(f"  Active: {sum(1 for n in self.nodes.values() if n.status == 'active')}")
        lines.append(f"  Candidates: {sum(1 for n in self.nodes.values() if n.status == 'candidate')}")
        lines.append(f"  Retired: {sum(1 for n in self.nodes.values() if n.status == 'retired')}")
        
        for node_id in self.root_ids:
            self._print_subtree(node_id, lines, indent=2)
        
        return "\n".join(lines)
    
    def _print_subtree(self, node_id: str, lines: List[str], indent: int):
        """Recursively print subtree."""
        node = self.nodes.get(node_id)
        if not node:
            return
        
        status_icon = {'active': '●', 'candidate': '◐', 'retired': '○'}.get(
            node.status, '?'
        )
        
        lines.append(
            f"{' ' * indent}{status_icon} {node.id} "
            f"[{node.fingerprint.state_dim}D×{node.fingerprint.action_count}A] "
            f"avg={node.performance.avg_reward:.1f} "
            f"concepts={len(node.discovered_concepts)}"
        )
        
        # Find children
        for child_id, child in self.nodes.items():
            if child.parent_id == node_id:
                self._print_subtree(child_id, lines, indent + 4)


if __name__ == "__main__":
    """Test policy tree operations."""
    print("=" * 50)
    print("POLICY TREE TEST")
    print("=" * 50)
    
    tree = PolicyTree()
    
    # Create a mock fingerprint
    fp1 = EnvironmentFingerprint(state_dim=128, action_count=4)
    fp1.reward_density = 0.05
    fp1.state_mean = 0.3
    fp1.to_vector()
    
    # Create root policy
    root = tree.create_root(fp1, weights={'W1': np.random.randn(128, 64)})
    root.performance.update(1.0)
    root.performance.update(2.0)
    root.performance.update(1.5)
    
    # Create similar fingerprint
    fp2 = EnvironmentFingerprint(state_dim=128, action_count=4)
    fp2.reward_density = 0.06
    fp2.state_mean = 0.31
    fp2.to_vector()
    
    # Should match root
    match = tree.find_best_match(fp2)
    if match:
        print(f"\n✅ Found match: {match.id} (expected root)")
        branch = tree.branch(match.id, fp2)
        branch.performance.update(3.0)
    else:
        print("\n⚠️  No match found, creating root")
    
    # Create very different fingerprint
    fp3 = EnvironmentFingerprint(state_dim=128, action_count=18)
    fp3.reward_density = 0.80
    fp3.state_mean = 0.7
    fp3.state_std = 0.3
    fp3.to_vector()
    
    match3 = tree.find_best_match(fp3)
    if match3:
        print(f"\nMatched to: {match3.id}")
        branch3 = tree.branch(match3.id, fp3)
    else:
        print(f"\n✅ No match — creating new root (very different env)")
        root2 = tree.create_root(fp3)
    
    print(f"\n{tree.summary()}")
    print("\n✅ PolicyTree test complete!")
