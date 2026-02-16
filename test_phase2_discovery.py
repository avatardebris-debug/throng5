"""
End-to-End Test: Phase 2 Micro-Test Engine

Tests the complete discovery pipeline:
1. Environment Analyzer profiles the environment
2. Micro-Tester probes actions and generates hypotheses
3. Reward Chaser explores rewarding actions and generalizes

Goal: Discover that moving right/down in GridWorld leads to goal
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.llm_policy import (
    EnvironmentAnalyzer,
    MicroTester,
    RewardChaser,
    RuleStatus
)
from throng4.environments.gridworld_variants import SparseRewardGridWorld


def test_phase2_discovery_pipeline():
    """Test complete Phase 2 discovery pipeline on GridWorld."""
    print("="*70)
    print("END-TO-END TEST: Phase 2 Discovery Pipeline")
    print("="*70)
    
    # Create environment
    env = SparseRewardGridWorld(size=5)
    
    # PHASE 1: Analyze environment
    print("\n[PHASE 1] Analyzing Environment")
    print("-"*70)
    analyzer = EnvironmentAnalyzer(n_probe_episodes=30, max_steps_per_episode=50)
    profile = analyzer.analyze(env)
    
    print(f"\n  Environment Profile:")
    print(f"    Obs shape: {profile.obs_shape}")
    print(f"    Actions: {profile.action_space.n_actions}")
    print(f"    Reward sparsity: {profile.reward_stats.sparsity:.1%}")
    print(f"    Controllable dims: {len(profile.dynamics.controllable_dims)}")
    
    # PHASE 2A: Micro-test all actions
    print("\n[PHASE 2A] Micro-Testing Actions")
    print("-"*70)
    tester = MicroTester(reward_threshold=0.01)
    
    # Probe from multiple starting states
    all_probes = []
    for trial in range(5):
        probes = tester.probe_all_actions(env, n_actions=4, max_probes_per_action=2)
        all_probes.extend(probes)
    
    print(f"\n  Completed {len(all_probes)} probes across {5} trials")
    
    # Generate hypotheses
    hypotheses = tester.generate_hypotheses_from_probes(all_probes, profile)
    
    print(f"\n  Generated {len(hypotheses)} hypotheses")
    print(f"\n  Top Hypotheses:")
    for rule in sorted(hypotheses, key=lambda r: r.confidence, reverse=True)[:5]:
        print(f"    - {rule.description}")
        print(f"      Status: {rule.status.value}, Confidence: {rule.confidence:.2f}")
    
    # PHASE 2B: Chase rewarding actions
    print("\n[PHASE 2B] Chasing Reward Signals")
    print("-"*70)
    chaser = RewardChaser(n_variations=15)
    
    # Find rewarding rules
    rewarding_rules = [r for r in hypotheses if r.feature == "reward" and r.confidence > 0.1]
    
    print(f"\n  Found {len(rewarding_rules)} rewarding rules to chase")
    
    chase_results = []
    for rule in rewarding_rules[:3]:  # Chase top 3
        print(f"\n  Chasing: {rule.description}")
        result = chaser.chase_reward(env, rule, profile)
        chase_results.append(result)
        
        print(f"    Variations tested: {result.n_variations_tested}")
        print(f"    Success rate: {result.success_rate:.1%}")
        
        if result.generalized_rule:
            print(f"    Generalized: {result.generalized_rule.description}")
            print(f"    Boundaries: {len(result.boundary_conditions)} dimensions")
    
    # SUMMARY
    print("\n" + "="*70)
    print("DISCOVERY SUMMARY")
    print("="*70)
    
    print(f"\n{tester.get_summary()}")
    print(f"\n{chaser.get_summary()}")
    
    # Show final rule library
    print(f"\n{tester.rule_library.summary()}")
    
    # Verify we discovered something useful
    active_rules = tester.rule_library.get_active_rules()
    anti_policies = tester.rule_library.get_anti_policies()
    dormant_rules = tester.rule_library.get_dormant_rules()
    print(f"\n[VERIFICATION]")
    print(f"  Active rules: {len(active_rules)}")
    print(f"  Anti-policies: {len(anti_policies)}")
    print(f"  Dormant rules: {len(dormant_rules)}")
    
    for rule in active_rules[:3]:
        print(f"\n  ACTIVE: {rule.description}")
        print(f"    Confidence: {rule.confidence:.2f}")
        print(f"    Tests: {rule.n_tests} ({rule.n_successes} successes)")
    
    # Check that we discovered state-change rules
    state_change_rules = [r for r in hypotheses if "dim" in r.feature]
    print(f"\n  State-change rules: {len(state_change_rules)}")
    
    assert len(hypotheses) > 0, "Should discover hypotheses"
    assert len(state_change_rules) > 0, "Should discover state-change rules"
    
    print("\n" + "="*70)
    print("[OK] Phase 2 discovery pipeline test PASSED!")
    print("="*70)
    
    return {
        'profile': profile,
        'hypotheses': hypotheses,
        'chase_results': chase_results,
        'tester': tester,
        'chaser': chaser
    }


if __name__ == '__main__':
    results = test_phase2_discovery_pipeline()
