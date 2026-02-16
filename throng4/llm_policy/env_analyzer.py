"""
Environment Analyzer — Programmatic introspection of game environments.

Probes an environment to build a structured profile:
- Observation/action space characteristics
- Reward distribution and sparsity
- State dynamics (which dims change, correlations)
- Controllability (which actions affect which states)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum


class SpaceType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MIXED = "mixed"


@dataclass
class ActionSpaceInfo:
    """Characterization of the action space."""
    space_type: SpaceType
    n_actions: Optional[int] = None          # for discrete
    action_shape: Optional[Tuple] = None     # for continuous
    action_ranges: Optional[List[Tuple[float, float]]] = None


@dataclass
class RewardStats:
    """Statistical summary of reward distribution."""
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float              # fraction of zero rewards
    positive_fraction: float     # fraction of positive rewards
    negative_fraction: float     # fraction of negative rewards


@dataclass
class DynamicsProfile:
    """State transition dynamics."""
    controllable_dims: List[int]             # which state dims respond to actions
    gravity_like_dims: List[int]             # dims that change without action
    correlation_matrix: np.ndarray           # pairwise correlations between dims
    variance_per_dim: np.ndarray             # how much each dim varies


@dataclass
class StateGroup:
    """Cluster of state dimensions that co-vary."""
    dims: List[int]
    correlation_strength: float
    interpretation: str          # "spatial_grid", "moving_object", "derived_feature", etc.


@dataclass
class EnvProfile:
    """Complete environment characterization."""
    obs_shape: Tuple[int, ...]
    obs_ranges: Dict[int, Tuple[float, float]]    # per-dim min/max
    action_space: ActionSpaceInfo
    reward_stats: RewardStats
    dynamics: DynamicsProfile
    state_groups: List[StateGroup]
    terminal_conditions: List[str]                # inferred stop conditions
    n_probe_episodes: int                         # how many episodes were used


class EnvironmentAnalyzer:
    """Analyze an environment through systematic probing."""
    
    def __init__(self, n_probe_episodes: int = 500, max_steps_per_episode: int = 100):
        """
        Initialize analyzer.
        
        Args:
            n_probe_episodes: Number of random episodes to run
            max_steps_per_episode: Max steps per episode
        """
        self.n_probe_episodes = n_probe_episodes
        self.max_steps_per_episode = max_steps_per_episode
    
    def analyze(self, env) -> EnvProfile:
        """
        Full analysis pipeline.
        
        Args:
            env: Environment with reset(), step(action) interface
            
        Returns:
            EnvProfile with complete characterization
        """
        print(f"[*] Analyzing environment with {self.n_probe_episodes} probe episodes...")
        
        # Collect transitions
        transitions = self._collect_transitions(env)
        
        # Analyze each component
        obs_shape, obs_ranges = self._analyze_observation_space(transitions)
        action_space = self._analyze_action_space(env, transitions)
        reward_stats = self._analyze_rewards(transitions)
        dynamics = self._analyze_dynamics(transitions)
        state_groups = self._cluster_state_dims(dynamics)
        terminal_conditions = self._infer_terminal_conditions(transitions)
        
        profile = EnvProfile(
            obs_shape=obs_shape,
            obs_ranges=obs_ranges,
            action_space=action_space,
            reward_stats=reward_stats,
            dynamics=dynamics,
            state_groups=state_groups,
            terminal_conditions=terminal_conditions,
            n_probe_episodes=self.n_probe_episodes
        )
        
        print(f"[OK] Analysis complete. Profile summary:")
        print(f"   Obs shape: {obs_shape}")
        print(f"   Actions: {action_space.n_actions if action_space.n_actions else action_space.action_shape}")
        print(f"   Reward: mean={reward_stats.mean:.3f}, std={reward_stats.std:.3f}, sparsity={reward_stats.sparsity:.1%}")
        print(f"   Controllable dims: {len(dynamics.controllable_dims)}/{len(obs_shape) if isinstance(obs_shape, tuple) else obs_shape[0]}")
        
        return profile
    
    def _collect_transitions(self, env) -> List[Dict]:
        """Run random episodes and collect all transitions."""
        transitions = []
        
        for ep in range(self.n_probe_episodes):
            state = env.reset()
            
            for step in range(self.max_steps_per_episode):
                # Random action
                if hasattr(env, 'action_space'):
                    action = env.action_space.sample()
                elif hasattr(env, 'get_valid_actions'):
                    valid = env.get_valid_actions()
                    action = valid[np.random.randint(len(valid))] if valid else None
                else:
                    # Assume discrete action space 0-3 (up, down, left, right)
                    action = np.random.randint(0, 4)
                
                if action is None:
                    break
                
                next_state, reward, done, info = env.step(action)
                
                transitions.append({
                    'state': np.array(state).flatten(),
                    'action': action,
                    'reward': reward,
                    'next_state': np.array(next_state).flatten(),
                    'done': done,
                    'info': info
                })
                
                if done:
                    break
                state = next_state
            
            if (ep + 1) % 100 == 0:
                print(f"   Collected {ep + 1}/{self.n_probe_episodes} episodes...")
        
        return transitions
    
    def _analyze_observation_space(self, transitions) -> Tuple[Tuple, Dict]:
        """Determine obs shape and ranges."""
        states = np.array([t['state'] for t in transitions])
        obs_shape = states.shape[1:]  # (n_samples, *obs_shape)
        
        # Compute per-dimension ranges
        obs_ranges = {}
        for dim in range(states.shape[1]):
            obs_ranges[dim] = (float(states[:, dim].min()), float(states[:, dim].max()))
        
        return obs_shape, obs_ranges
    
    def _analyze_action_space(self, env, transitions) -> ActionSpaceInfo:
        """Characterize action space."""
        if hasattr(env, 'action_space'):
            if hasattr(env.action_space, 'n'):
                # Discrete
                return ActionSpaceInfo(
                    space_type=SpaceType.DISCRETE,
                    n_actions=env.action_space.n
                )
            else:
                # Continuous
                return ActionSpaceInfo(
                    space_type=SpaceType.CONTINUOUS,
                    action_shape=env.action_space.shape,
                    action_ranges=[(low, high) for low, high in 
                                  zip(env.action_space.low, env.action_space.high)]
                )
        else:
            # Infer from transitions - find unique integer actions
            actions = [t['action'] for t in transitions]
            unique_actions = sorted(set(actions))
            # Assume discrete actions are 0, 1, 2, ... n-1
            n_actions = len(unique_actions)
            return ActionSpaceInfo(
                space_type=SpaceType.DISCRETE,
                n_actions=n_actions
            )
    
    def _analyze_rewards(self, transitions) -> RewardStats:
        """Compute reward statistics."""
        rewards = np.array([t['reward'] for t in transitions])
        
        return RewardStats(
            mean=float(rewards.mean()),
            std=float(rewards.std()),
            min_val=float(rewards.min()),
            max_val=float(rewards.max()),
            sparsity=float((rewards == 0).mean()),
            positive_fraction=float((rewards > 0).mean()),
            negative_fraction=float((rewards < 0).mean())
        )
    
    def _analyze_dynamics(self, transitions) -> DynamicsProfile:
        """Analyze state transition dynamics."""
        states = np.array([t['state'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        
        # Which dims change?
        state_deltas = next_states - states
        variance_per_dim = state_deltas.var(axis=0)
        
        # Controllable dims: those that change significantly
        controllable_threshold = variance_per_dim.mean() * 0.1
        controllable_dims = [i for i, var in enumerate(variance_per_dim) 
                            if var > controllable_threshold]
        
        # Gravity-like dims: always change in same direction
        mean_delta_per_dim = state_deltas.mean(axis=0)
        gravity_threshold = variance_per_dim.mean() * 0.5
        gravity_like_dims = [i for i, (mean_d, var) in enumerate(zip(mean_delta_per_dim, variance_per_dim))
                            if abs(mean_d) > 0.1 and var < gravity_threshold]
        
        # Correlation matrix
        correlation_matrix = np.corrcoef(states.T)
        
        return DynamicsProfile(
            controllable_dims=controllable_dims,
            gravity_like_dims=gravity_like_dims,
            correlation_matrix=correlation_matrix,
            variance_per_dim=variance_per_dim
        )
    
    def _cluster_state_dims(self, dynamics: DynamicsProfile) -> List[StateGroup]:
        """Find clusters of co-varying state dimensions."""
        corr = dynamics.correlation_matrix
        groups = []
        
        # Simple clustering: find dims with high correlation
        visited = set()
        for i in range(len(corr)):
            if i in visited:
                continue
            
            # Find all dims highly correlated with i
            cluster = [i]
            for j in range(i + 1, len(corr)):
                if j not in visited and abs(corr[i, j]) > 0.7:
                    cluster.append(j)
                    visited.add(j)
            
            if len(cluster) > 1:
                groups.append(StateGroup(
                    dims=cluster,
                    correlation_strength=float(np.mean([corr[i, j] for i in cluster for j in cluster if i != j])),
                    interpretation="unknown"  # will be refined by object detector
                ))
                visited.add(i)
        
        return groups
    
    def _infer_terminal_conditions(self, transitions) -> List[str]:
        """Infer what causes episodes to end."""
        terminal_transitions = [t for t in transitions if t['done']]
        
        if not terminal_transitions:
            return ["no_terminal_states_observed"]
        
        conditions = []
        
        # Check for reward-based termination
        terminal_rewards = [t['reward'] for t in terminal_transitions]
        if np.mean(terminal_rewards) < -0.5:
            conditions.append("catastrophic_failure")
        elif np.mean(terminal_rewards) > 0.5:
            conditions.append("goal_reached")
        
        # Check for state-based termination
        # (would need more sophisticated analysis)
        
        return conditions if conditions else ["unknown"]
