"""
Environment Fingerprinting — Characterize environments WITHOUT knowing what they are.

Builds a numerical fingerprint from ~20 random exploration episodes:
- State distribution (mean, std, sparsity, range)
- Reward structure (density, scale, sign ratio, variance)
- Action sensitivity (which actions change state most)
- Temporal dynamics (state change rate, autocorrelation)

Used by MetaPolicyController to match new environments to existing policies.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvironmentFingerprint:
    """Numerical characterization of an environment."""
    
    # Dimensionality
    state_dim: int = 0
    action_count: int = 0
    
    # State distribution
    state_mean: float = 0.0
    state_std: float = 0.0
    state_sparsity: float = 0.0        # fraction of near-zero values
    state_range: float = 0.0
    state_entropy: float = 0.0         # how uniform the distribution is
    
    # Reward structure  
    reward_density: float = 0.0        # fraction of steps with non-zero reward
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    reward_sign_ratio: float = 0.0     # positive / total non-zero
    reward_sparsity: float = 0.0       # how sparse rewards are in time
    
    # Action sensitivity
    action_state_deltas: Dict[int, float] = field(default_factory=dict)
    action_reward_means: Dict[int, float] = field(default_factory=dict)
    most_impactful_action: int = 0
    action_diversity_score: float = 0.0  # how different actions affect state
    
    # Temporal dynamics
    state_change_rate: float = 0.0     # avg L2 norm of state deltas
    state_autocorrelation: float = 0.0 # temporal correlation
    episode_length_mean: float = 0.0
    episode_length_std: float = 0.0
    
    # Derived
    fingerprint_vector: Optional[np.ndarray] = None  # for fast similarity
    
    def similarity(self, other: 'EnvironmentFingerprint') -> float:
        """Cosine similarity between fingerprint vectors."""
        if self.fingerprint_vector is None or other.fingerprint_vector is None:
            return 0.0
        
        a = self.fingerprint_vector
        b = other.fingerprint_vector
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def to_vector(self) -> np.ndarray:
        """Convert to fixed-size vector for comparison."""
        # Normalize action-specific values to fixed size
        max_actions = max(self.action_count, 1)
        action_deltas = np.zeros(18)  # max 18 Atari actions
        action_rewards = np.zeros(18)
        
        for a, delta in self.action_state_deltas.items():
            if a < 18:
                action_deltas[a] = delta
        for a, r in self.action_reward_means.items():
            if a < 18:
                action_rewards[a] = r
        
        vec = np.array([
            # Dimensionality (2)
            self.state_dim / 256.0,
            self.action_count / 18.0,
            
            # State distribution (5)
            self.state_mean,
            self.state_std,
            self.state_sparsity,
            self.state_range,
            self.state_entropy,
            
            # Reward structure (7)
            self.reward_density,
            np.tanh(self.reward_mean / 10.0),  # normalize
            np.tanh(self.reward_std / 10.0),
            np.tanh(self.reward_min / 100.0),
            np.tanh(self.reward_max / 100.0),
            self.reward_sign_ratio,
            self.reward_sparsity,
            
            # Temporal (4)
            np.tanh(self.state_change_rate),
            self.state_autocorrelation,
            np.tanh(self.episode_length_mean / 1000.0),
            np.tanh(self.episode_length_std / 500.0),
            
            # Action diversity (1)
            self.action_diversity_score,
        ], dtype=np.float32)
        
        # Append action-specific values
        vec = np.concatenate([vec, action_deltas, action_rewards])
        
        self.fingerprint_vector = vec
        return vec
    
    def summary(self) -> str:
        """Human-readable summary (no game names)."""
        return (
            f"Environment Fingerprint:\n"
            f"  State: {self.state_dim}D, mean={self.state_mean:.3f}, "
            f"std={self.state_std:.3f}, sparsity={self.state_sparsity:.1%}\n"
            f"  Actions: {self.action_count}, "
            f"diversity={self.action_diversity_score:.3f}\n"
            f"  Reward: density={self.reward_density:.1%}, "
            f"range=[{self.reward_min:.1f}, {self.reward_max:.1f}], "
            f"sign_ratio={self.reward_sign_ratio:.1%}\n"
            f"  Dynamics: change_rate={self.state_change_rate:.3f}, "
            f"autocorr={self.state_autocorrelation:.3f}, "
            f"ep_len={self.episode_length_mean:.0f}±{self.episode_length_std:.0f}"
        )


def fingerprint_environment(env, n_exploration_episodes: int = 20) -> EnvironmentFingerprint:
    """
    Build a fingerprint by running random episodes.
    
    Args:
        env: Environment with reset(), step(action), get_info() methods
        n_exploration_episodes: Number of random episodes to sample
        
    Returns:
        EnvironmentFingerprint characterizing the environment
    """
    fp = EnvironmentFingerprint()
    
    # Get basic info
    info = env.get_info()
    fp.state_dim = env.n_features
    fp.action_count = info['n_actions']
    
    # Collect data from random exploration
    all_states = []
    all_rewards = []
    all_state_deltas = []
    action_deltas = {a: [] for a in range(fp.action_count)}
    action_rewards = {a: [] for a in range(fp.action_count)}
    episode_lengths = []
    
    for ep in range(n_exploration_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 1000:
            action = np.random.randint(fp.action_count)
            next_state, reward, done, _ = env.step(action)
            
            all_states.append(state)
            all_rewards.append(reward)
            
            delta = np.linalg.norm(next_state - state)
            all_state_deltas.append(delta)
            action_deltas[action].append(delta)
            action_rewards[action].append(reward)
            
            state = next_state
            steps += 1
        
        episode_lengths.append(steps)
    
    if not all_states:
        fp.to_vector()
        return fp
    
    states = np.array(all_states)
    rewards = np.array(all_rewards)
    deltas = np.array(all_state_deltas)
    
    # === State distribution ===
    fp.state_mean = float(np.mean(states))
    fp.state_std = float(np.std(states))
    fp.state_sparsity = float(np.mean(np.abs(states) < 0.01))
    fp.state_range = float(np.max(states) - np.min(states))
    
    # State entropy (how uniform the value distribution is)
    hist, _ = np.histogram(states.flatten(), bins=50)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    fp.state_entropy = float(-np.sum(hist * np.log(hist)))
    
    # === Reward structure ===
    nonzero_rewards = rewards[rewards != 0]
    fp.reward_density = float(len(nonzero_rewards) / max(len(rewards), 1))
    fp.reward_mean = float(np.mean(rewards))
    fp.reward_std = float(np.std(rewards)) if len(rewards) > 1 else 0.0
    fp.reward_min = float(np.min(rewards))
    fp.reward_max = float(np.max(rewards))
    
    if len(nonzero_rewards) > 0:
        fp.reward_sign_ratio = float(np.sum(nonzero_rewards > 0) / len(nonzero_rewards))
    
    # Reward sparsity (average gap between rewards)
    reward_steps = np.where(rewards != 0)[0]
    if len(reward_steps) > 1:
        gaps = np.diff(reward_steps)
        fp.reward_sparsity = float(np.mean(gaps) / max(len(rewards), 1))
    else:
        fp.reward_sparsity = 1.0
    
    # === Action sensitivity ===
    for a in range(fp.action_count):
        if action_deltas[a]:
            fp.action_state_deltas[a] = float(np.mean(action_deltas[a]))
        else:
            fp.action_state_deltas[a] = 0.0
        
        if action_rewards[a]:
            fp.action_reward_means[a] = float(np.mean(action_rewards[a]))
        else:
            fp.action_reward_means[a] = 0.0
    
    # Most impactful action
    if fp.action_state_deltas:
        fp.most_impactful_action = max(
            fp.action_state_deltas, key=fp.action_state_deltas.get
        )
    
    # Action diversity: how different are action effects?
    delta_values = list(fp.action_state_deltas.values())
    if len(delta_values) > 1 and np.mean(delta_values) > 0:
        fp.action_diversity_score = float(
            np.std(delta_values) / max(np.mean(delta_values), 1e-8)
        )
    
    # === Temporal dynamics ===
    fp.state_change_rate = float(np.mean(deltas))
    
    # Autocorrelation of state changes
    if len(deltas) > 2:
        centered = deltas - np.mean(deltas)
        var = np.var(deltas)
        if var > 1e-8:
            autocorr = np.correlate(centered[:-1], centered[1:])[0]
            fp.state_autocorrelation = float(autocorr / (var * len(centered)))
    
    fp.episode_length_mean = float(np.mean(episode_lengths))
    fp.episode_length_std = float(np.std(episode_lengths)) if len(episode_lengths) > 1 else 0.0
    
    # Build comparison vector
    fp.to_vector()
    
    return fp


if __name__ == "__main__":
    """Test fingerprinting on available Atari games."""
    import sys
    sys.path.insert(0, '.')
    from throng4.environments.atari_adapter import AtariAdapter
    
    games = ['Breakout', 'Pong', 'SpaceInvaders']
    fingerprints = {}
    
    print("=" * 60)
    print("ENVIRONMENT FINGERPRINTING TEST")
    print("=" * 60)
    
    for game in games:
        print(f"\nFingerprinting environment (20 random episodes)...")
        env = AtariAdapter(game)
        fp = fingerprint_environment(env, n_exploration_episodes=20)
        fingerprints[game] = fp
        env.close()
        
        print(fp.summary())
    
    print("\n" + "=" * 60)
    print("SIMILARITY MATRIX")
    print("=" * 60)
    
    names = list(fingerprints.keys())
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            sim = fingerprints[name_i].similarity(fingerprints[name_j])
            print(f"  Env{i+1} vs Env{j+1}: {sim:.3f}")
    
    print("\n(Game names hidden from fingerprints — only revealed here for validation)")
    for i, name in enumerate(names):
        print(f"  Env{i+1} = {name}")
    
    print("\n✅ Fingerprinting test complete!")
