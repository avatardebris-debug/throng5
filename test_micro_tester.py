"""
Test Micro-Tester on GridWorld.

Verifies that the micro-tester can:
1. Probe all actions
2. Classify effects (rewarding/catastrophic/neutral)
3. Generate hypotheses about action effects
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.llm_policy.micro_tester import MicroTester
from throng4.llm_policy.env_analyzer import EnvironmentAnalyzer
from throng4.environments.gridworld_variants import SparseRewardGridWorld


def test_micro_tester_gridworld():
    """Test micro-tester on GridWorld."""
    print("="*70)
    print("TEST: Micro-Tester on GridWorld")
    print("="*70)
    
    # Create environment
    env = SparseRewardGridWorld(size=5)
    
    # Analyze environment first
    print("\n[1] Analyzing environment...")
    analyzer = EnvironmentAnalyzer(n_probe_episodes=20, max_steps_per_episode=30)
    profile = analyzer.analyze(env)
    
    print(f"\n  Detected {profile.action_space.n_actions} actions")
    
    # Create micro-tester
    print("\n[2] Creating micro-tester...")
    tester = MicroTester(reward_threshold=0.01, catastrophic_threshold=-0.5)
    
    # Probe all actions from start state
    print("\n[3] Probing all actions from start state...")
    probes = tester.probe_all_actions(env, n_actions=4, max_probes_per_action=3)
    
    print(f"\n  Completed {len(probes)} probes")
    
    # Classify results
    rewarding = [p for p in probes if p.classification == "rewarding"]
    catastrophic = [p for p in probes if p.classification == "catastrophic"]
    neutral = [p for p in probes if p.classification == "neutral"]
    
    print(f"\n  Rewarding: {len(rewarding)}")
    print(f"  Catastrophic: {len(catastrophic)}")
    print(f"  Neutral: {len(neutral)}")
    
    # Generate hypotheses
    print("\n[4] Generating hypotheses from probes...")
    hypotheses = tester.generate_hypotheses_from_probes(probes, profile)
    
    print(f"\n  Generated {len(hypotheses)} hypotheses")
    
    # Show discovered rules
    print("\n[5] Discovered Rules:")
    for rule in hypotheses[:5]:  # Show first 5
        print(f"\n  Rule: {rule.description}")
        print(f"    Status: {rule.status.value}")
        print(f"    Confidence: {rule.confidence:.2f}")
        print(f"    Tests: {rule.n_tests} ({rule.n_successes} successes)")
    
    # Show catastrophic rules
    if tester.rule_library.catastrophic_rules:
        print("\n[6] Catastrophic Rules:")
        for rule_id, rule in tester.rule_library.catastrophic_rules.items():
            print(f"\n  {rule.description}")
    
    # Summary
    print("\n" + "="*70)
    print(tester.get_summary())
    print("="*70)
    
    # Verify basic properties
    assert len(probes) > 0, "Should have probe results"
    assert len(hypotheses) > 0, "Should generate hypotheses"
    
    # GridWorld: most actions should be neutral from start (no immediate reward)
    # Only reaching goal gives reward
    assert len(neutral) > len(rewarding), "Most actions should be neutral in GridWorld"
    
    print("\n[OK] Micro-tester test passed!")
    return tester, hypotheses


if __name__ == '__main__':
    tester, hypotheses = test_micro_tester_gridworld()
