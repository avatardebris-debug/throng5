"""
Test Environment Analyzer on GridWorld.

Verifies that the analyzer correctly characterizes a simple known environment.
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.llm_policy.env_analyzer import EnvironmentAnalyzer
from throng4.environments.gridworld_variants import SparseRewardGridWorld


def test_gridworld_analysis():
    """Test analyzer on GridWorld (known structure)."""
    print("="*70)
    print("TEST: Environment Analyzer on GridWorld")
    print("="*70)
    
    # Create environment
    env = SparseRewardGridWorld(size=5)
    
    # Analyze
    analyzer = EnvironmentAnalyzer(n_probe_episodes=100, max_steps_per_episode=50)
    profile = analyzer.analyze(env)
    
    # Verify results
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    # Check obs shape
    expected_obs_size = 2  # (x, y) normalized coordinates
    actual_obs_size = profile.obs_shape[0] if isinstance(profile.obs_shape, tuple) else profile.obs_shape
    print(f"\n✓ Obs shape: {profile.obs_shape}")
    assert actual_obs_size == expected_obs_size, f"Expected {expected_obs_size}, got {actual_obs_size}"
    
    # Check action space
    print(f"✓ Action space: {profile.action_space.n_actions} discrete actions")
    assert profile.action_space.n_actions == 4, "GridWorld should have 4 actions"
    
    # Check reward stats
    print(f"✓ Reward stats:")
    print(f"    Mean: {profile.reward_stats.mean:.3f}")
    print(f"    Sparsity: {profile.reward_stats.sparsity:.1%}")
    print(f"    Positive fraction: {profile.reward_stats.positive_fraction:.1%}")
    
    # GridWorld rewards are sparse (only at goal)
    assert profile.reward_stats.sparsity > 0.8, "GridWorld rewards should be sparse"
    
    # Check controllability
    print(f"✓ Controllable dims: {len(profile.dynamics.controllable_dims)}/{actual_obs_size}")
    print(f"    Dims: {profile.dynamics.controllable_dims[:10]}...")  # show first 10
    
    # Check terminal conditions
    print(f"✓ Terminal conditions: {profile.terminal_conditions}")
    
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED")
    print("="*70)
    
    return profile


if __name__ == '__main__':
    profile = test_gridworld_analysis()
