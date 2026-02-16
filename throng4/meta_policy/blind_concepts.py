"""
Blind Concept Library — Discover patterns from signals, not game names.

Concepts are discovered purely from:
- Reward spikes (what action/state pattern preceded sudden improvement?)
- State clusters (RAM regions that correlate with high reward)
- Action sequences (repeating patterns before rewards)

Transferability is measured by testing concepts on environments with
SIMILAR FINGERPRINTS, not by game name lookup.
"""

import numpy as np
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from throng4.meta_policy.env_fingerprint import EnvironmentFingerprint


@dataclass
class DiscoveredConcept:
    """A pattern discovered purely from signals."""
    id: str
    pattern_type: str  # 'reward_spike', 'state_cluster', 'action_sequence', 'weight_pattern'
    
    # Discovery context (fingerprint-based, no game names)
    discovery_fingerprint_vector: Optional[np.ndarray] = None
    discovery_episode_range: Tuple[int, int] = (0, 0)
    
    # The actual pattern
    pattern_data: dict = field(default_factory=dict)
    
    # Evidence
    evidence_count: int = 0          # how many times this pattern was observed
    evidence_reward_boost: float = 0.0  # avg reward when pattern is active vs not
    
    # Transferability (measured empirically)
    tested_on_fingerprints: int = 0
    successful_transfers: int = 0
    
    @property
    def transferability(self) -> float:
        if self.tested_on_fingerprints == 0:
            return 0.5  # Unknown
        return self.successful_transfers / self.tested_on_fingerprints
    
    @property
    def confidence(self) -> float:
        """How confident are we this is a real pattern?"""
        # More evidence = more confidence, with diminishing returns
        return min(1.0, self.evidence_count / 10.0) * (
            0.5 + 0.5 * min(1.0, abs(self.evidence_reward_boost) / 2.0)
        )
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.pattern_type,
            'confidence': self.confidence,
            'transferability': self.transferability,
            'evidence': self.evidence_count,
            'reward_boost': self.evidence_reward_boost,
            'tested_on': self.tested_on_fingerprints,
            'successful': self.successful_transfers,
        }


class BlindConceptLibrary:
    """
    Discover and manage concepts from raw signals.
    No game names. No external knowledge.
    """
    
    # Thresholds for concept discovery
    REWARD_SPIKE_THRESHOLD = 2.0   # std devs above mean
    MIN_EVIDENCE = 3               # minimum observations before concept is "real"
    CLUSTER_MIN_CORRELATION = 0.3  # min correlation with reward
    
    def __init__(self):
        self.concepts: Dict[str, DiscoveredConcept] = {}
        self._discovery_buffer = deque(maxlen=200)  # recent (state, action, reward) tuples
    
    def record_step(self, state: np.ndarray, action: int, reward: float,
                    next_state: np.ndarray):
        """Record a step for concept discovery."""
        self._discovery_buffer.append({
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'delta': np.linalg.norm(next_state - state),
        })
    
    def discover_concepts(self, 
                          episode_history: List[Dict],
                          fingerprint: EnvironmentFingerprint
                          ) -> List[DiscoveredConcept]:
        """
        Analyze episode history for discoverable patterns.
        
        Called at end of environment run.
        Returns newly discovered concepts.
        """
        new_concepts = []
        
        # 1. Reward spikes — what happened before sudden improvement?
        spike_concepts = self._discover_reward_spikes(episode_history, fingerprint)
        new_concepts.extend(spike_concepts)
        
        # 2. State clusters — which state features correlate with reward?
        cluster_concepts = self._discover_state_clusters(fingerprint)
        new_concepts.extend(cluster_concepts)
        
        # 3. Action sequences — repeating patterns before rewards
        seq_concepts = self._discover_action_sequences(fingerprint)
        new_concepts.extend(seq_concepts)
        
        # Add to library
        for concept in new_concepts:
            self.concepts[concept.id] = concept
        
        if new_concepts:
            print(f"[BlindConcepts] Discovered {len(new_concepts)} new concepts "
                  f"(total library: {len(self.concepts)})")
        
        return new_concepts
    
    def _discover_reward_spikes(self, episode_history: List[Dict],
                                 fingerprint: EnvironmentFingerprint
                                 ) -> List[DiscoveredConcept]:
        """Find episodes where reward jumped significantly."""
        if len(episode_history) < 10:
            return []
        
        rewards = [ep.get('reward', 0) for ep in episode_history]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        
        if std_r < 1e-8:
            return []
        
        concepts = []
        
        for i in range(5, len(rewards)):
            # Is this a spike?
            z_score = (rewards[i] - mean_r) / std_r
            if z_score > self.REWARD_SPIKE_THRESHOLD:
                # What was different about this episode?
                concept = DiscoveredConcept(
                    id=f"spike_{uuid.uuid4().hex[:6]}",
                    pattern_type='reward_spike',
                    discovery_fingerprint_vector=fingerprint.fingerprint_vector.copy() if fingerprint.fingerprint_vector is not None else None,
                    discovery_episode_range=(max(0, i - 5), i),
                    pattern_data={
                        'spike_magnitude': float(z_score),
                        'episode_index': i,
                        'reward': float(rewards[i]),
                        'context_rewards': [float(r) for r in rewards[max(0,i-5):i]],
                    },
                    evidence_count=1,
                    evidence_reward_boost=float(rewards[i] - mean_r),
                )
                concepts.append(concept)
        
        # Deduplicate — keep the strongest spike
        if len(concepts) > 3:
            concepts.sort(key=lambda c: c.pattern_data['spike_magnitude'], reverse=True)
            concepts = concepts[:3]
        
        return concepts
    
    def _discover_state_clusters(self, fingerprint: EnvironmentFingerprint
                                  ) -> List[DiscoveredConcept]:
        """Find state features that correlate with reward."""
        if len(self._discovery_buffer) < 50:
            return []
        
        buffer = list(self._discovery_buffer)
        states = np.array([b['state'] for b in buffer])
        rewards = np.array([b['reward'] for b in buffer])
        
        # Skip if all rewards are the same
        if np.std(rewards) < 1e-8:
            return []
        
        concepts = []
        
        # Find state features most correlated with reward
        correlations = []
        for feat_idx in range(min(states.shape[1], 128)):
            feat_values = states[:, feat_idx]
            if np.std(feat_values) < 1e-8:
                continue
            
            corr = np.corrcoef(feat_values, rewards)[0, 1]
            if not np.isnan(corr):
                correlations.append((feat_idx, float(corr)))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top correlated features
        for feat_idx, corr in correlations[:3]:
            if abs(corr) >= self.CLUSTER_MIN_CORRELATION:
                concept = DiscoveredConcept(
                    id=f"cluster_{uuid.uuid4().hex[:6]}",
                    pattern_type='state_cluster',
                    discovery_fingerprint_vector=fingerprint.fingerprint_vector.copy() if fingerprint.fingerprint_vector is not None else None,
                    pattern_data={
                        'feature_index': feat_idx,
                        'correlation': corr,
                        'feature_mean': float(np.mean(states[:, feat_idx])),
                        'feature_std': float(np.std(states[:, feat_idx])),
                        'reward_when_high': float(np.mean(
                            rewards[states[:, feat_idx] > np.median(states[:, feat_idx])]
                        )),
                        'reward_when_low': float(np.mean(
                            rewards[states[:, feat_idx] <= np.median(states[:, feat_idx])]
                        )),
                    },
                    evidence_count=len(buffer),
                    evidence_reward_boost=abs(corr),
                )
                concepts.append(concept)
        
        return concepts
    
    def _discover_action_sequences(self, fingerprint: EnvironmentFingerprint
                                    ) -> List[DiscoveredConcept]:
        """Find repeating action patterns that precede rewards."""
        if len(self._discovery_buffer) < 30:
            return []
        
        buffer = list(self._discovery_buffer)
        actions = [b['action'] for b in buffer]
        rewards = [b['reward'] for b in buffer]
        
        concepts = []
        
        # Look for action pairs that precede non-zero rewards
        seq_len = 3
        seq_rewards: Dict[tuple, List[float]] = {}
        
        for i in range(seq_len, len(actions)):
            seq = tuple(actions[i-seq_len:i])
            if rewards[i] != 0:
                if seq not in seq_rewards:
                    seq_rewards[seq] = []
                seq_rewards[seq].append(rewards[i])
        
        # Find sequences with consistently positive rewards
        for seq, rews in seq_rewards.items():
            if len(rews) >= 2 and np.mean(rews) > 0:
                concept = DiscoveredConcept(
                    id=f"seq_{uuid.uuid4().hex[:6]}",
                    pattern_type='action_sequence',
                    discovery_fingerprint_vector=fingerprint.fingerprint_vector.copy() if fingerprint.fingerprint_vector is not None else None,
                    pattern_data={
                        'sequence': list(seq),
                        'avg_reward_after': float(np.mean(rews)),
                        'occurrences': len(rews),
                    },
                    evidence_count=len(rews),
                    evidence_reward_boost=float(np.mean(rews)),
                )
                concepts.append(concept)
        
        # Keep top 3 sequences
        if concepts:
            concepts.sort(key=lambda c: c.evidence_reward_boost, reverse=True)
            concepts = concepts[:3]
        
        return concepts
    
    def find_transferable(self, fingerprint: EnvironmentFingerprint,
                          min_confidence: float = 0.3) -> List[DiscoveredConcept]:
        """
        Find concepts discovered in SIMILAR fingerprints.
        
        No game names — purely fingerprint similarity.
        """
        transferable = []
        
        for concept in self.concepts.values():
            if concept.confidence < min_confidence:
                continue
            
            # Check fingerprint similarity
            if (concept.discovery_fingerprint_vector is not None 
                and fingerprint.fingerprint_vector is not None):
                sim = float(np.dot(
                    concept.discovery_fingerprint_vector,
                    fingerprint.fingerprint_vector
                ) / (
                    np.linalg.norm(concept.discovery_fingerprint_vector) *
                    np.linalg.norm(fingerprint.fingerprint_vector) + 1e-8
                ))
                
                if sim > 0.5:  # Similar enough environment
                    transferable.append(concept)
        
        return transferable
    
    def mark_transfer_result(self, concept_id: str, success: bool):
        """Record whether a concept transfer was successful."""
        if concept_id in self.concepts:
            self.concepts[concept_id].tested_on_fingerprints += 1
            if success:
                self.concepts[concept_id].successful_transfers += 1
    
    def apply_concepts_to_weights(self, weights: dict,
                                   concepts: List[DiscoveredConcept]) -> dict:
        """
        Apply discovered concepts as weight biases.
        
        - State cluster concepts: boost weights for correlated features
        - Action sequence concepts: bias toward rewarding actions  
        - Reward spike concepts: scale learning based on spike magnitude
        """
        if not concepts:
            return weights
        
        modified = {k: v.copy() for k, v in weights.items()}
        
        for concept in concepts:
            strength = concept.confidence * 0.05  # Subtle bias
            
            if concept.pattern_type == 'state_cluster':
                feat_idx = concept.pattern_data.get('feature_index', 0)
                corr = concept.pattern_data.get('correlation', 0)
                
                # Boost weights from this feature
                if 'W1' in modified and feat_idx < modified['W1'].shape[0]:
                    modified['W1'][feat_idx, :] += corr * strength
            
            elif concept.pattern_type == 'action_sequence':
                seq = concept.pattern_data.get('sequence', [])
                if seq and 'W2' in modified:
                    # Bias toward the last action in the sequence
                    last_action = seq[-1]
                    if last_action < modified['W2'].shape[1]:
                        modified['W2'][:, last_action] += strength
            
            elif concept.pattern_type == 'reward_spike':
                # General weight scaling based on spike magnitude
                mag = concept.pattern_data.get('spike_magnitude', 1.0)
                scale = 1.0 + strength * min(mag, 3.0) * 0.01
                if 'W1' in modified:
                    modified['W1'] *= scale
        
        return modified
    
    def summary(self) -> str:
        lines = [f"BlindConceptLibrary ({len(self.concepts)} concepts):"]
        
        by_type = {}
        for c in self.concepts.values():
            by_type.setdefault(c.pattern_type, []).append(c)
        
        for ptype, concepts in by_type.items():
            avg_conf = np.mean([c.confidence for c in concepts])
            avg_trans = np.mean([c.transferability for c in concepts])
            lines.append(f"  {ptype}: {len(concepts)} concepts, "
                        f"avg_confidence={avg_conf:.2f}, "
                        f"avg_transferability={avg_trans:.2f}")
        
        return "\n".join(lines)


if __name__ == "__main__":
    """Test blind concept discovery."""
    print("=" * 50)
    print("BLIND CONCEPT LIBRARY TEST")
    print("=" * 50)
    
    library = BlindConceptLibrary()
    
    # Simulate episode data
    fp = EnvironmentFingerprint(state_dim=128, action_count=4)
    fp.to_vector()
    
    # Record some steps with patterns
    for i in range(100):
        state = np.random.rand(128).astype(np.float32)
        action = i % 4
        # Create a pattern: action 2 followed by action 3 gives reward
        reward = 0.0
        if i > 2 and action == 3 and (i - 1) % 4 == 2:
            reward = 1.0
        # State feature 10 correlates with reward
        if reward > 0:
            state[10] = 0.9
        
        next_state = state + np.random.randn(128) * 0.01
        library.record_step(state, action, reward, next_state.astype(np.float32))
    
    # Create episode history with a spike
    episode_history = [
        {'reward': 1.0} for _ in range(20)
    ]
    episode_history[15] = {'reward': 10.0}  # Spike!
    
    # Discover concepts
    new_concepts = library.discover_concepts(episode_history, fp)
    
    print(f"\nDiscovered {len(new_concepts)} concepts:")
    for c in new_concepts:
        print(f"  - {c.id}: type={c.pattern_type}, "
              f"confidence={c.confidence:.2f}, "
              f"boost={c.evidence_reward_boost:.2f}")
    
    print(f"\n{library.summary()}")
    
    # Test transferability lookup
    similar_fp = EnvironmentFingerprint(state_dim=128, action_count=4)
    similar_fp.to_vector()
    
    transferable = library.find_transferable(similar_fp)
    print(f"\nTransferable concepts for similar env: {len(transferable)}")
    
    print("\n✅ BlindConceptLibrary test complete!")
