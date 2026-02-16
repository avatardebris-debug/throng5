"""
Tetris Curriculum Learning with Dellacherie Comparison
======================================================

Train PortableNNAgent through Tetris curriculum (levels 1-7) and compare
against Dellacherie heuristic baseline. Optionally integrate with Tetra
for real-time strategy discovery.

Usage:
    # Without Tetra
    python train_tetris_curriculum.py
    
    # With Tetra integration
    python train_tetris_curriculum.py --tetra
"""

import sys
import os
import json
import time
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.environments.tetris_curriculum import TetrisCurriculum, TetrisCurriculumEnv
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.llm_policy.openclaw_bridge import OpenClawBridge
from throng4.llm_policy.eval_auditor import EvalAuditor


# ─── Configuration ───────────────────────────────────────────────────

# Adaptive max_pieces per level (level 1 too easy → reduce to avoid perpetual games)
MAX_PIECES_PER_LEVEL = {
    1: 50,     # O-block only, very easy
    2: 100,    # O + I blocks
    3: 150,    # + T blocks
    4: 200,    # + S, Z blocks
    5: 300,    # All blocks, 8-wide
    6: 400,    # All blocks, 10-wide
    7: 500,    # Standard Tetris
}

EPISODES_PER_LEVEL = {
    1: 10,     # Quick sanity check
    2: 50,     # Warmup
    3: 100,    # Moderate training
    4: 150,
    5: 200,
    6: 200,
    7: 200,
}

ADVANCE_THRESHOLD_PER_LEVEL = {
    1: 1.0,    # Very low bar for level 1
    2: 2.0,    # Gradually increase
    3: 3.0,
    4: 4.0,
    5: 5.0,
    6: 6.0,
    7: 8.0,
}


# ─── Training ────────────────────────────────────────────────────────

def run_episode(agent: PortableNNAgent, adapter: TetrisAdapter, log_actions: bool = False) -> Dict[str, Any]:
    """Run a single Tetris episode."""
    agent.reset_episode()
    state = adapter.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    action_log = [] if log_actions else None
    
    while not done:
        # Get valid actions
        valid_actions = adapter.get_valid_actions()
        if len(valid_actions) == 0:
            break
        
        # Agent selects action using PortableNNAgent API
        action = agent.select_action(
            valid_actions=valid_actions,
            feature_fn=adapter.make_features,
            explore=True
        )
        
        if log_actions:
            action_log.append(action)
        
        # Take step
        next_state, reward, done, info = adapter.step(action)
        
        # Record step for training
        features = adapter.make_features(action)
        agent.record_step(features, reward)
        
        episode_reward += reward
        steps += 1
        state = next_state
    
    # End episode and train
    info = adapter.get_info()
    agent.end_episode(final_score=float(info['lines_cleared']))
    
    result = {
        'lines': info['lines_cleared'],
        'pieces': info['pieces_placed'],
        'reward': episode_reward,
        'steps': steps,
    }
    
    if log_actions:
        result['action_log'] = action_log
    
    return result


def train_level(
    agent: PortableNNAgent,
    level: int,
    episodes: int,
    max_pieces: int,
    bridge: Optional[OpenClawBridge] = None,
    auditor: Optional[EvalAuditor] = None
) -> Dict[str, Any]:
    """Train on a single curriculum level."""
    adapter = TetrisAdapter(level=level, max_pieces=max_pieces)
    
    level_config = TetrisCurriculumEnv.LEVELS[level]
    print(f"\n{'='*70}")
    print(f"Level {level}: {level_config['name']}")
    print(f"  Board: {level_config['width']}×{level_config['height']}")
    print(f"  Pieces: {level_config['pieces']}")
    print(f"  Episodes: {episodes}, Max pieces: {max_pieces}")
    print(f"{'='*70}\n")
    
    episode_lines = []
    episode_rewards = []
    
    for ep in range(episodes):
        # Log actions every 10th episode for audit
        log_actions = (auditor is not None) and ((ep + 1) % 10 == 0)
        result = run_episode(agent, adapter, log_actions=log_actions)
        episode_lines.append(result['lines'])
        episode_rewards.append(result['reward'])
        
        # Audit episode if enabled
        if log_actions and auditor:
            # CRITICAL: Create fresh adapter for audit to avoid corrupting training state
            audit_adapter = TetrisAdapter(level=level, max_pieces=max_pieces)
            audit_report = auditor.audit_episode(
                episode_id=ep + 1,
                level=level,
                reported_metrics={'lines': result['lines'], 'reward': result['reward']},
                action_log=result['action_log'],
                env_adapter=audit_adapter
            )
            if audit_report.anomalies:
                print(f"    ⚠️ Audit found anomalies: {audit_report.anomalies[:2]}")
        
        if (ep + 1) % 10 == 0:
            recent_lines = episode_lines[-10:]
            recent_rewards = episode_rewards[-10:]
            print(f"  Episode {ep+1:3d}/{episodes}: "
                  f"Lines={np.mean(recent_lines):.2f}±{np.std(recent_lines):.2f}, "
                  f"Reward={np.mean(recent_rewards):.2f}±{np.std(recent_rewards):.2f}")
        
        # Send to Tetra every N episodes
        if bridge and (ep + 1) % 20 == 0:
            bridge.send_observation(
                episode=ep + 1,
                observation=f"Level {level} training progress: {ep+1}/{episodes} episodes completed. "
                           f"Recent performance: {np.mean(episode_lines[-20:]):.2f} lines avg.",
                context={
                    'level': level,
                    'level_name': level_config['name'],
                    'episodes_completed': ep + 1,
                    'mean_lines_recent': float(np.mean(episode_lines[-20:])),
                    'max_lines_recent': int(np.max(episode_lines[-20:])),
                    'board_size': [level_config['height'], level_config['width']],
                }
            )
    
    stats = {
        'level': level,
        'level_name': level_config['name'],
        'episodes': episodes,
        'max_pieces': max_pieces,
        'mean_lines': float(np.mean(episode_lines)),
        'std_lines': float(np.std(episode_lines)),
        'max_lines': int(np.max(episode_lines)),
        'mean_reward': float(np.mean(episode_rewards)),
        'lines_history': episode_lines,
    }
    
    print(f"\n  Level {level} Summary:")
    print(f"    Mean lines: {stats['mean_lines']:.2f} ± {stats['std_lines']:.2f}")
    print(f"    Max lines: {stats['max_lines']}")
    print(f"    Mean reward: {stats['mean_reward']:.2f}")
    
    return stats


def train_curriculum(
    agent: PortableNNAgent,
    start_level: int = 1,
    max_level: int = 7,
    bridge: Optional[OpenClawBridge] = None,
    auditor: Optional[EvalAuditor] = None,
    auto_advance: bool = True
) -> List[Dict[str, Any]]:
    """Train through Tetris curriculum."""
    all_stats = []
    
    for level in range(start_level, max_level + 1):
        episodes = EPISODES_PER_LEVEL.get(level, 100)
        max_pieces = MAX_PIECES_PER_LEVEL.get(level, 200)
        
        stats = train_level(agent, level, episodes, max_pieces, bridge, auditor)
        all_stats.append(stats)
        
        # Check if should advance
        if auto_advance and level < max_level:
            threshold = ADVANCE_THRESHOLD_PER_LEVEL.get(level, 2.0)
            if stats['mean_lines'] >= threshold:
                print(f"\n  ✅ Threshold met ({stats['mean_lines']:.2f} >= {threshold:.2f}) — advancing to level {level+1}")
                
                if bridge:
                    bridge.send_observation(
                        episode=stats['episodes'],
                        observation=f"Advanced from level {level} → {level+1}. "
                                   f"Agent exceeded threshold ({stats['mean_lines']:.2f} >= {threshold:.2f} lines).",
                        context={
                            'from_level': level,
                            'to_level': level + 1,
                            'threshold': threshold,
                            'actual_performance': stats['mean_lines'],
                            'phase': 'level_advancement'
                        }
                    )
            else:
                print(f"\n  ❌ Threshold not met ({stats['mean_lines']:.2f} < {threshold:.2f}) — stopping at level {level}")
                break
    
    return all_stats


# ─── Comparison ──────────────────────────────────────────────────────

def load_dellacherie_baseline(filepath: str = "dellacherie_results.txt") -> Dict[int, float]:
    """Load Dellacherie baseline results."""
    baseline = {}
    
    if not os.path.exists(filepath):
        print(f"⚠️ Baseline file not found: {filepath}")
        return baseline
    
    # Parse simple text format (Level, Lines)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                level = int(parts[0])
                lines = float(parts[1])
                baseline[level] = lines
    
    return baseline


def compare_results(
    curriculum_stats: List[Dict[str, Any]],
    baseline: Dict[int, float],
    bridge: Optional[OpenClawBridge] = None
):
    """Compare curriculum learning vs Dellacherie baseline."""
    print(f"\n{'='*70}")
    print("TETRIS CURRICULUM vs DELLACHERIE COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Level':<8} {'Curriculum':<15} {'Dellacherie':<15} {'Gap':<10}")
    print(f"{'-'*8} {'-'*15} {'-'*15} {'-'*10}")
    
    comparison = []
    for stats in curriculum_stats:
        level = stats['level']
        curr_perf = stats['mean_lines']
        base_perf = baseline.get(level, 0.0)
        gap = curr_perf - base_perf
        
        print(f"{level:<8} {curr_perf:>7.2f} ({stats['max_lines']:>3d})  "
              f"{base_perf:>7.2f}         {gap:>+7.2f}")
        
        comparison.append({
            'level': level,
            'curriculum': curr_perf,
            'dellacherie': base_perf,
            'gap': gap,
            'max': stats['max_lines']
        })
    
    print(f"\n{'='*70}")
    
    # Send summary to Tetra
    if bridge:
        total_episodes = sum(s['episodes'] for s in curriculum_stats)
        max_level_reached = curriculum_stats[-1]['level']
        
        bridge.send_observation(
            episode=total_episodes,
            observation=f"Tetris curriculum training complete. Reached level {max_level_reached}. "
                       f"Total episodes: {total_episodes}.",
            context={
                'phase': 'curriculum_complete',
                'max_level': max_level_reached,
                'total_episodes': total_episodes,
                'comparison': comparison
            }
        )
    
    return comparison


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tetris Curriculum Learning")
    parser.add_argument('--tetra', action='store_true', help="Enable Tetra integration")
    parser.add_argument('--start-level', type=int, default=1, help="Starting level")
    parser.add_argument('--max-level', type=int, default=7, help="Max level")
    parser.add_argument('--no-auto-advance', action='store_true', help="Disable auto-advance")
    parser.add_argument('--output', type=str, default='tetris_curriculum_results.json', help="Output file")
    args = parser.parse_args()
    
    print(f"\n🎮 Tetris Curriculum Learning Experiment")
    print(f"{'='*70}\n")
    
    # Initialize bridge if Tetra enabled
    bridge = None
    if args.tetra:
        bridge = OpenClawBridge(game="Tetris_Curriculum")
        if bridge.check_gateway():
            print("  ✅ Tetra bridge connected\n")
        else:
            print("  ⚠️ Gateway unavailable — continuing without Tetra\n")
            bridge = None
    
    # Send initial environment profile to Tetra
    if bridge:
        bridge.send_observation(
            episode=0,
            observation=f"Starting Tetris curriculum training: levels {args.start_level}-{args.max_level}.",
            context={
                'start_level': args.start_level,
                'max_level': args.max_level,
                'auto_advance': not args.no_auto_advance,
                'phase': 'training_start'
            }
        )
    
    # Initialize auditor
    auditor = EvalAuditor(audit_dir="eval_audits")
    print(f"  Auditor enabled: checking every 10th episode\n")
    
    # Initialize agent
    # Use a simple config for now — can be customized
    sample_adapter = TetrisAdapter(level=args.start_level)
    sample_adapter.reset()  # MUST reset before extracting features
    valid_actions = sample_adapter.get_valid_actions()
    if not valid_actions:
        print(f"  No valid actions after reset! Check environment setup.")
        return
    sample_features = sample_adapter.make_features(valid_actions[0])
    input_size = len(sample_features)
    
    config = AgentConfig(
        n_hidden=128,
        epsilon=0.2,
        gamma=0.95,
        learning_rate=0.005
    )
    
    agent = PortableNNAgent(n_features=input_size, config=config)
    
    print(f"  Agent: {input_size} features → [{config.n_hidden}] → value")
    print(f"  Learning rate: {config.learning_rate}, Gamma: {config.gamma}, Epsilon: {config.epsilon}\n")
    
    # Train through curriculum
    all_stats = train_curriculum(
        agent=agent,
        start_level=args.start_level,
        max_level=args.max_level,
        bridge=bridge,
        auditor=auditor,
        auto_advance=not args.no_auto_advance
    )
    
    # Load Dellacherie baseline
    baseline = load_dellacherie_baseline()
    
    # Compare results
    comparison = compare_results(all_stats, baseline, bridge)
    
    # Save results
    results = {
        'timestamp': time.time(),
        'config': {
            'start_level': args.start_level,
            'max_level': args.max_level,
            'auto_advance': not args.no_auto_advance,
            'tetra_enabled': args.tetra,
        },
        'agent_config': {
            'n_features': input_size,
            'n_hidden': config.n_hidden,
            'learning_rate': config.learning_rate,
            'gamma': config.gamma,
            'epsilon': config.epsilon,
        },
        'level_stats': all_stats,
        'comparison': comparison,
        'baseline': baseline,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n  Results saved to: {args.output}")
    
    if bridge:
        print(f"  Bridge stats:")
        print(f"    {bridge.get_summary()}")
    
    # Save and print audit report
    audit_file = auditor.save_session(f"tetris_audit_{args.start_level}_to_{args.max_level}.json")
    print(f"\n  Audit saved to: {audit_file}")
    auditor.print_report()
    
    print(f"\n🎮 Training complete!\n")


if __name__ == "__main__":
    main()
