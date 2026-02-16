"""
🧩 LIVE FrozenLake Discovery Loop with Tetra
=============================================

Runs the full discovery pipeline on FrozenLake with Tetra watching in real-time:

1. EnvironmentAnalyzer profiles FrozenLake (obs/action space, rewards, dynamics)
2. AttributionDiagnoser diagnoses stochasticity per action  
3. MicroTester probes each action for effects
4. Bridge sends each discovery to Tetra → gets hypothesis back
5. Results written to memory + archive

Usage:
    python live_frozenlake_tetra.py
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.llm_policy.env_analyzer import EnvironmentAnalyzer
from throng4.llm_policy.micro_tester import MicroTester
from throng4.llm_policy.attribution import AttributionDiagnoser, Attribution
from throng4.llm_policy.hypothesis import RuleLibrary, RuleStatus
from throng4.llm_policy.openclaw_bridge import OpenClawBridge
from throng4.llm_policy.rule_archive import RuleArchive


def format_env_profile_for_tetra(profile) -> str:
    """Convert EnvProfile to a readable summary for Tetra."""
    lines = [
        "## Environment Profile",
        f"- Observation shape: {profile.obs_shape}",
        f"- Action space: {profile.action_space.space_type.value}, {profile.action_space.n_actions} actions",
        f"- Reward: mean={profile.reward_stats.mean:.3f}, std={profile.reward_stats.std:.3f}",
        f"  range=[{profile.reward_stats.min_val:.1f}, {profile.reward_stats.max_val:.1f}]",
        f"  sparsity={profile.reward_stats.sparsity:.2f} (fraction of zero-reward steps)",
        f"  positive fraction: {profile.reward_stats.positive_fraction:.3f}",
        f"- Controllable dims: {profile.dynamics.controllable_dims}",
        f"- Gravity-like dims: {profile.dynamics.gravity_like_dims}",
        f"- Probed episodes: {profile.n_probe_episodes}",
    ]
    
    if profile.terminal_conditions:
        lines.append(f"- Terminal conditions: {profile.terminal_conditions}")
    
    if profile.state_groups:
        for g in profile.state_groups:
            lines.append(f"- State group: dims {g.dims} (corr={g.correlation_strength:.2f}) — {g.interpretation}")
    
    return "\n".join(lines)


def format_attribution_for_tetra(action, result) -> str:
    """Convert AttributionResult to a readable observation for Tetra."""
    action_names = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}
    name = action_names.get(action, f"action_{action}")
    
    lines = [
        f"Action {action} ({name}):",
        f"  Attribution: {result.attribution.value}",
        f"  Stochasticity: {result.stochasticity_score:.3f}",
        f"  Confidence: {result.confidence:.2f}",
    ]
    
    if result.outcome_distribution and result.outcome_distribution.outcomes:
        sorted_outcomes = sorted(
            result.outcome_distribution.outcomes.items(), 
            key=lambda x: x[1], reverse=True
        )[:5]
        lines.append(f"  Top outcomes: {dict(sorted_outcomes)}")
    
    lines.append(f"  Summary: {result.summary()}")
    
    return "\n".join(lines)


def main():
    print("\n" + "=" * 70)
    print("🧩 LIVE FrozenLake Discovery Loop with Tetra")
    print("=" * 70)
    
    # ── Step 0: Initialize bridge and check gateway ──────────────────────
    print("\n[1/5] Connecting to Tetra...")
    bridge = OpenClawBridge(game="FrozenLake_4x4_slippery")
    
    if not bridge.check_gateway():
        print("  ❌ Gateway not available! Running offline (will queue observations)")
    else:
        print("  ✅ Gateway LIVE — Tetra is listening")
    
    # ── Step 1: Environment Analysis ─────────────────────────────────────
    print("\n[2/5] Profiling FrozenLake environment...")
    
    try:
        from throng4.environments.gym_envs import FrozenLakeAdapter
        env = FrozenLakeAdapter(is_slippery=True)
    except ImportError as e:
        print(f"  ❌ Cannot create FrozenLake: {e}")
        return
    
    analyzer = EnvironmentAnalyzer(n_probe_episodes=200, max_steps_per_episode=50)
    profile = analyzer.analyze(env)
    
    profile_text = format_env_profile_for_tetra(profile)
    print(f"\n{profile_text}")
    
    # Send profile to Tetra
    print("\n  → Sending environment profile to Tetra...")
    profile_response = bridge.send_observation(
        episode=0,
        observation=f"Environment profiling complete for FrozenLake 4x4 (is_slippery=True).\n{profile_text}",
        context={
            "obs_shape": list(profile.obs_shape),
            "n_actions": profile.action_space.n_actions,
            "reward_mean": float(profile.reward_stats.mean),
            "reward_sparsity": float(profile.reward_stats.sparsity),
            "controllable_dims": profile.dynamics.controllable_dims,
            "phase": "environment_profiling"
        }
    )
    print(f"  ← Tetra: {profile_response.raw[:300]}")
    
    # ── Step 2: Attribution Diagnosis ────────────────────────────────────
    print("\n[3/5] Running attribution diagnosis on all 4 actions...")
    
    diagnoser = AttributionDiagnoser(n_rng_trials=20, stochastic_threshold=0.1)
    
    attribution_summary = []
    for action in range(4):
        result = diagnoser.diagnose(env, action=action)
        attr_text = format_attribution_for_tetra(action, result)
        attribution_summary.append(attr_text)
        print(f"\n  {attr_text}")
    
    # Send attribution results to Tetra
    print("\n  → Sending attribution results to Tetra...")
    attr_response = bridge.send_observation(
        episode=0,
        observation="Attribution diagnosis complete for all 4 actions on FrozenLake.\n\n" 
                    + "\n\n".join(attribution_summary),
        context={
            "phase": "attribution_diagnosis",
            "game": "FrozenLake_4x4",
            "is_slippery": True,
            "n_actions": 4
        }
    )
    print(f"  ← Tetra: {attr_response.raw[:400]}")
    
    # ── Step 3: Micro-Testing ────────────────────────────────────────────
    print("\n[4/5] Running micro-tests (probing each action from 5 random states)...")
    
    tester = MicroTester(
        reward_threshold=0.01,
        catastrophic_threshold=-0.5,
        environment_context="FrozenLake_4x4_slippery"
    )
    
    library = RuleLibrary()
    all_probes = []
    
    for trial in range(5):
        state = env.reset()
        
        # Take a few random steps to reach different states
        for _ in range(np.random.randint(0, 8)):
            action = np.random.randint(0, 4)
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
                break
        
        # Probe all actions from this state
        probes = tester.probe_all_actions(env, n_actions=4, max_probes_per_action=2)
        all_probes.extend(probes)
        
        # Generate hypotheses
        hypotheses = tester.generate_hypotheses_from_probes(probes, profile)
        
        for rule in hypotheses:
            library.add_with_anti_policy(rule)
        
        print(f"\n  Trial {trial+1}/5 from state={state[:4]}...")
        print(f"    Probes: {len(probes)}")
        print(f"    Hypotheses generated: {len(hypotheses)}")
        
        for rule in hypotheses:
            print(f"      [{rule.status.value}] {rule.description} (conf={rule.confidence:.2f})")
    
    # Send micro-test results to Tetra
    active_rules = library.get_active_rules()
    anti_policies = library.get_anti_policies()
    dormant_rules = [r for r in library.rules.values() if r.status == RuleStatus.DORMANT]
    
    rules_summary = []
    for r in list(library.rules.values())[:15]:  # Top 15 rules
        rules_summary.append(f"- [{r.status.value}] {r.description} (conf={r.confidence:.2f}, tests={r.n_tests})")
    
    print(f"\n  → Sending {len(library.rules)} discovered rules to Tetra...")
    micro_response = bridge.send_observation(
        episode=0,
        observation=(
            f"Micro-testing complete on FrozenLake 4x4.\n"
            f"Total rules discovered: {len(library.rules)}\n"
            f"  Active: {len(active_rules)}\n"
            f"  Anti-policies: {len(anti_policies)}\n"
            f"  Dormant: {len(dormant_rules)}\n\n"
            f"Rules:\n" + "\n".join(rules_summary)
        ),
        context={
            "phase": "micro_testing",
            "total_rules": len(library.rules),
            "active": len(active_rules),
            "anti_policies": len(anti_policies),
            "dormant": len(dormant_rules),
            "total_probes": len(all_probes)
        }
    )
    print(f"  ← Tetra: {micro_response.raw[:400]}")
    
    # ── Step 4: Ask Tetra for synthesis ──────────────────────────────────
    print("\n[5/5] Asking Tetra for synthesis...")
    
    synthesis_response = bridge.query(
        f"Based on the FrozenLake 4x4 (slippery) analysis:\n"
        f"- Environment is stochastic (is_slippery=True)\n"
        f"- {len(active_rules)} active rules, {len(anti_policies)} anti-policies\n"
        f"- Reward is extremely sparse (only at goal state)\n\n"
        f"Questions:\n"
        f"1. What meta-concepts can we extract that would transfer to other grid games?\n"
        f"2. Are there any rules that seem like artifacts of stochasticity rather than real mechanics?\n"
        f"3. What should the agent prioritize learning next?"
    )
    
    print(f"\n  ← Tetra's synthesis:\n{synthesis_response.raw[:600]}")
    
    # ── Write to memory and archive ──────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("Writing to memory and archive...")
    
    # Write rules to memory
    for rule in list(library.rules.values())[:5]:
        bridge.write_rule_to_memory(rule)
    
    # Save to archive
    archive = RuleArchive(":memory:")  # Use temp DB for demo
    archive.store_library(library, session_id="FrozenLake_4x4_slippery")
    stats = archive.get_statistics()
    
    print(f"\n  Archive stats: {json.dumps(stats, indent=2)}")
    print(f"  Memory written to: {bridge.memory_dir}")
    
    # Final summary
    print("\n" + "=" * 70)
    print(f"Bridge session:")
    print(bridge.get_summary())
    print("\n🧩 Live loop complete! Tetra has seen the full discovery pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
