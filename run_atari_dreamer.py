"""
Atari Dreamer Runner — Basal Ganglia Integration for Visual Games
==================================================================

Wraps PortableNNAgent + AtariAdapter with the full dreamer stack
(DreamerEngine, Amygdala, DreamerTeacher, OptionsLibrary).

The dreamer:
  1. Learns a world model from RAM state (128 bytes)
  2. Runs game-specific hypotheses (e.g., Breakout: aim_center, maximize_hits)
  3. Generates teaching signals that nudge action selection
  4. Tracks dreamer_reliance and backs off as the agent improves
  5. Discovers behavioral options and tracks their performance

Usage:
    python run_atari_dreamer.py --game Breakout --episodes 100
    python run_atari_dreamer.py --game Breakout --episodes 50 --no-dreamer  # baseline
    python run_atari_dreamer.py --game Breakout --episodes 100 --compare
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.environments.atari_adapter import AtariAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.basal_ganglia.dreamer_engine import DreamerEngine, Hypothesis
from throng4.basal_ganglia.amygdala import Amygdala
from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher


# ═══════════════════════════════════════════════════════════════
# Game-Specific Hypotheses
# ═══════════════════════════════════════════════════════════════

def create_breakout_hypotheses(n_actions: int) -> List[Hypothesis]:
    """
    Breakout-specific strategy hypotheses.
    
    These operate on the compressed RAM state from the dreamer's world model.
    """
    return [
        Hypothesis(
            id=0,
            name="aim_center",
            action_selector=lambda s: n_actions // 2 if n_actions > 2 else 0,
            description="Stay near center position",
        ),
        Hypothesis(
            id=1,
            name="maximize_hits",
            action_selector=lambda s: int(np.argmax(s[:min(n_actions, len(s))])),
            description="Maximize ball-paddle contact",
        ),
        Hypothesis(
            id=2,
            name="track_ball",
            action_selector=lambda s: int(np.argmin(np.abs(s[:min(n_actions, len(s))] - 0.5))),
            description="Track ball position",
        ),
    ]


def create_pong_hypotheses(n_actions: int) -> List[Hypothesis]:
    """Pong-specific hypotheses."""
    return [
        Hypothesis(
            id=0,
            name="track_ball",
            action_selector=lambda s: int(np.argmax(s[:min(n_actions, len(s))])),
            description="Follow ball vertically",
        ),
        Hypothesis(
            id=1,
            name="stay_center",
            action_selector=lambda s: n_actions // 2 if n_actions > 2 else 0,
            description="Return to center when no threat",
        ),
        Hypothesis(
            id=2,
            name="anticipate",
            action_selector=lambda s: int(np.argmin(s[:min(n_actions, len(s))])),
            description="Anticipate ball trajectory",
        ),
    ]


def create_generic_hypotheses(n_actions: int) -> List[Hypothesis]:
    """Generic hypotheses for unknown games."""
    return [
        Hypothesis(
            id=0,
            name="explore_high",
            action_selector=lambda s: int(np.argmax(s[:min(n_actions, len(s))])),
            description="Prefer high-activation actions",
        ),
        Hypothesis(
            id=1,
            name="explore_low",
            action_selector=lambda s: int(np.argmin(s[:min(n_actions, len(s))])),
            description="Prefer low-activation actions",
        ),
        Hypothesis(
            id=2,
            name="balanced",
            action_selector=lambda s: n_actions // 2 if n_actions > 2 else 0,
            description="Balanced/neutral strategy",
        ),
    ]


def create_game_hypotheses(game_name: str, n_actions: int) -> List[Hypothesis]:
    """Create hypotheses based on game name."""
    game_lower = game_name.lower()
    
    if 'breakout' in game_lower:
        return create_breakout_hypotheses(n_actions)
    elif 'pong' in game_lower:
        return create_pong_hypotheses(n_actions)
    else:
        return create_generic_hypotheses(n_actions)


# ═══════════════════════════════════════════════════════════════
# Dreamer Atari Runner
# ═══════════════════════════════════════════════════════════════

class DreamerAtariRunner:
    """
    Runs Atari training with the full basal ganglia dreamer active.
    
    The dreamer:
      - Learns RAM state transition dynamics from every real step
      - Runs 3 hypothesis dreams per episode (when calibrated)
      - Nudges action selection toward dreamer-recommended actions
      - Tracks which strategies work in which contexts
      - Automatically reduces its influence as the agent improves
    """
    
    def __init__(self, game: str = 'Breakout', max_steps: int = 10000,
                 dreamer_enabled: bool = True,
                 dreamer_state_size: int = 64,
                 dream_steps: int = 30,
                 nudge_strength: float = 0.3):
        """
        Args:
            game: Atari game name (e.g., 'Breakout', 'Pong')
            max_steps: Max steps per episode
            dreamer_enabled: Whether to use the dreamer
            dreamer_state_size: Compressed state dimension for world model
            dream_steps: Lookahead depth per dream
            nudge_strength: Max influence of dreamer on action selection (0-1)
        """
        self.game = game
        self.max_steps = max_steps
        self.dreamer_enabled = dreamer_enabled
        self.dream_steps = dream_steps
        self.nudge_strength = nudge_strength
        
        # Environment
        self.adapter = AtariAdapter(game_name=game, max_steps=max_steps)
        info = self.adapter.get_info()
        n_features = info['n_features']  # 128 (RAM state)
        n_actions = info['n_actions']
        
        # Agent
        self.agent = PortableNNAgent(
            n_features=n_features,
            config=AgentConfig(
                n_hidden=256,  # Larger than Tetris (more complex state)
                epsilon=0.15,
                gamma=0.99,
                learning_rate=0.001,
            )
        )
        
        # Dreamer stack
        if dreamer_enabled:
            self.dreamer_n_actions = n_actions
            self.dreamer = DreamerEngine(
                n_hypotheses=3,
                network_size='micro',
                state_size=dreamer_state_size,
                n_actions=n_actions,
                dream_interval=1,  # Teacher will adjust dynamically
            )
            self.amygdala = Amygdala()
            self.teacher = DreamerTeacher(
                n_actions=n_actions,
                state_dim=min(32, dreamer_state_size),
            )
            self.hypotheses = create_game_hypotheses(game, n_actions)
        else:
            self.dreamer = None
            self.amygdala = None
            self.teacher = None
            self.hypotheses = None
        
        # Tracking
        self.episode_results: List[Dict[str, Any]] = []
        self.dreamer_events: List[Dict[str, Any]] = []
        self._last_state: Optional[np.ndarray] = None
        self._step_count = 0
    
    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single episode with dreamer integration."""
        self.agent.reset_episode()
        state = self.adapter.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        dreamer_nudges = 0
        dreamer_agrees = 0
        
        while not done:
            valid_actions = self.adapter.get_valid_actions()
            
            # ── Agent selects action (with optional dreamer nudge) ──
            dreamer_bias = None
            if self.dreamer_enabled and self.teacher:
                dreamer_bias = self._get_dreamer_bias(
                    state, valid_actions, episode_num
                )
            
            action = self._select_action_with_nudge(
                state, valid_actions, dreamer_bias
            )
            
            # ── Step ──
            next_state, reward, done, info = self.adapter.step(action)
            self.agent.record_step(state, reward)
            
            # ── Feed dreamer world model ──
            if self.dreamer_enabled and self.dreamer:
                compressed_state = self.dreamer._ensure_compressed(
                    state if isinstance(state, np.ndarray)
                    else np.zeros(self.dreamer.state_size)
                )
                compressed_next = self.dreamer._ensure_compressed(
                    next_state if isinstance(next_state, np.ndarray)
                    else np.zeros(self.dreamer.state_size)
                )
                self.dreamer.learn(compressed_state, action,
                                   compressed_next, reward)
                self._last_state = compressed_state
                
                # Track agreement
                if dreamer_bias is not None:
                    dreamer_nudges += 1
                    dreamer_action = dreamer_bias.get('action')
                    if dreamer_action is not None and dreamer_action == action:
                        dreamer_agrees += 1
                    self.teacher.record_policy_action(
                        compressed_state, action,
                        dreamer_action if dreamer_action is not None else action,
                        reward
                    )
            
            episode_reward += reward
            steps += 1
            state = next_state
            self._step_count += 1
        
        # ── End episode ──
        self.agent.end_episode(final_score=episode_reward)
        
        # ── Dream cycle (post-episode) ──
        dream_info = {}
        if (self.dreamer_enabled and self.dreamer
                and self.dreamer.is_calibrated
                and self._last_state is not None):
            dream_info = self._run_dream_cycle(episode_num)
        
        result = {
            'episode': episode_num,
            'reward': episode_reward,
            'steps': steps,
            'dreamer_nudges': dreamer_nudges,
            'dreamer_agrees': dreamer_agrees,
            **dream_info,
        }
        
        self.episode_results.append(result)
        return result
    
    def _get_dreamer_bias(self, state: np.ndarray,
                          valid_actions: list,
                          episode_num: int) -> Optional[Dict]:
        """Get dreamer's action recommendation."""
        rec = self.teacher.get_best_action(
            state if isinstance(state, np.ndarray)
            else np.zeros(16)
        )
        if rec is None:
            return None
        
        action_idx, confidence = rec
        if confidence < 0.15:
            return None
        
        # Scale nudge by reliance
        reliance = self.teacher.dreamer_reliance
        effective_nudge = self.nudge_strength * confidence * reliance
        
        return {
            'action': action_idx % len(valid_actions),
            'confidence': confidence,
            'nudge': effective_nudge,
        }
    
    def _select_action_with_nudge(self, state: np.ndarray,
                                   valid_actions: list,
                                   dreamer_bias: Optional[Dict]) -> int:
        """Select action from agent, optionally nudged by dreamer."""
        # Epsilon-greedy exploration
        if np.random.rand() < self.agent.epsilon:
            return np.random.choice(valid_actions)
        
        # Score all actions
        scores = []
        for i, action in enumerate(valid_actions):
            score = float(self.agent.forward(state))
            
            # Apply dreamer nudge
            if (dreamer_bias is not None
                    and i == dreamer_bias['action']
                    and dreamer_bias['nudge'] > 0.01):
                score += dreamer_bias['nudge']
            
            scores.append(score)
        
        best_idx = int(np.argmax(scores))
        return valid_actions[best_idx]
    
    def _run_dream_cycle(self, episode_num: int) -> Dict:
        """Run dream cycle post-episode."""
        dream_state = self._last_state
        if dream_state is None:
            return {}
        
        # Run dreams
        dream_results = self.dreamer.dream(
            dream_state, self.hypotheses, n_steps=self.dream_steps
        )
        
        if not dream_results:
            return {}
        
        # Feed teacher
        signals = self.teacher.process_dream_results(
            dream_results, dream_state, self._step_count
        )
        
        # Dynamically adjust dream interval
        self.dreamer.dream_interval = (
            self.teacher.recommended_dream_interval
        )
        
        # Amygdala check
        danger = self.amygdala.assess_danger(
            dream_results, current_step=self._step_count
        )
        danger_level = danger.level.value if danger else "safe"
        
        if self.amygdala.should_override(danger, self._step_count):
            self.amygdala.record_override(self._step_count)
            self.dreamer_events.append({
                'episode': episode_num,
                'event': 'amygdala_override',
                'danger': danger_level,
            })
        
        return {
            'dreamer_reliance': self.teacher.dreamer_reliance,
            'dream_interval': self.dreamer.dream_interval,
            'active_options': len(self.teacher.options.active_options),
            'danger_level': danger_level,
            'best_hypothesis': dream_results[0].hypothesis_name,
            'best_h_reward': dream_results[0].total_predicted_reward,
            'teaching_signals': len(signals),
        }
    
    def run_training(self, n_episodes: int = 100,
                     verbose: bool = True) -> Dict[str, Any]:
        """Run full training session."""
        print(f"\n{'='*70}")
        print(f"ATARI DREAMER TRAINING — {self.game}")
        print(f"{'='*70}")
        print(f"  Dreamer: {'ENABLED' if self.dreamer_enabled else 'DISABLED (baseline)'}")
        if self.dreamer_enabled:
            print(f"  Dream steps: {self.dream_steps}")
            print(f"  Nudge strength: {self.nudge_strength}")
        print(f"  Episodes: {n_episodes}")
        print(f"{'='*70}\n")
        
        t0 = time.time()
        
        for ep in range(n_episodes):
            result = self.run_episode(ep)
            
            if verbose and (ep + 1) % 10 == 0:
                recent = self.episode_results[-10:]
                avg_reward = np.mean([r['reward'] for r in recent])
                max_reward = max(r['reward'] for r in recent)
                
                line = (
                    f"  Ep {ep+1:3d}/{n_episodes}: "
                    f"R={avg_reward:.1f} (max={max_reward:.0f})"
                )
                
                if self.dreamer_enabled and 'dreamer_reliance' in result:
                    line += (
                        f" | dream: reliance={result['dreamer_reliance']:.0%}"
                        f" int={result.get('dream_interval', '?')}"
                        f" opts={result.get('active_options', 0)}"
                    )
                
                print(line)
        
        elapsed = time.time() - t0
        
        # ── Summary ──
        all_rewards = [r['reward'] for r in self.episode_results]
        
        summary = {
            'game': self.game,
            'dreamer_enabled': self.dreamer_enabled,
            'episodes': n_episodes,
            'elapsed_s': elapsed,
            'mean_reward': float(np.mean(all_rewards)),
            'std_reward': float(np.std(all_rewards)),
            'max_reward': float(np.max(all_rewards)),
            'final_mean_reward': float(np.mean(all_rewards[-20:])),
            'final_max_reward': float(np.max(all_rewards[-20:])),
        }
        
        if self.dreamer_enabled and self.teacher:
            summary['dreamer'] = {
                'final_reliance': self.teacher.dreamer_reliance,
                'dreamer_needed': self.teacher.dreamer_is_needed,
                'signals_generated': self.teacher._total_signals_generated,
                'signals_followed': self.teacher._total_signals_followed,
                'active_options': len(self.teacher.options.active_options),
                'promoted_options': len(self.teacher.options.promoted_options),
                'hypothesis_profiles': {
                    p.hypothesis_name: {
                        'win_rate': p.win_rate,
                        'avg_reward': p.avg_reward,
                        'specialization': p.specialization_score,
                    }
                    for p in self.teacher.profiles.values()
                },
                'amygdala_overrides': len(self.dreamer_events),
            }
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*70}")
        print(f"  Mean reward: {summary['mean_reward']:.1f} ± {summary['std_reward']:.1f}")
        print(f"  Max reward: {summary['max_reward']:.0f}")
        print(f"  Final 20 avg: {summary['final_mean_reward']:.1f}")
        
        if self.dreamer_enabled and self.teacher:
            d = summary['dreamer']
            print(f"\n  Dreamer Stats:")
            print(f"    Reliance: {d['final_reliance']:.0%}")
            print(f"    Still needed: {d['dreamer_needed']}")
            print(f"    Signals: {d['signals_generated']} generated, "
                  f"{d['signals_followed']} followed")
            print(f"    Options: {d['active_options']} active, "
                  f"{d['promoted_options']} promoted")
            if d['hypothesis_profiles']:
                print(f"    Hypothesis profiles:")
                for name, stats in d['hypothesis_profiles'].items():
                    print(f"      {name}: win={stats['win_rate']:.0%}, "
                          f"r={stats['avg_reward']:+.2f}, "
                          f"spec={stats['specialization']:.2f}")
            if d['amygdala_overrides'] > 0:
                print(f"    Amygdala overrides: {d['amygdala_overrides']}")
        
        print(f"{'='*70}\n")
        
        return summary


def run_comparison(game: str = 'Breakout', episodes: int = 100):
    """Run dreamer vs baseline comparison."""
    print("\n" + "=" * 70)
    print("DREAMER vs BASELINE COMPARISON")
    print("=" * 70)
    
    # Baseline (no dreamer)
    print("\n▸ Running BASELINE (no dreamer)...")
    baseline = DreamerAtariRunner(
        game=game, dreamer_enabled=False
    )
    baseline_stats = baseline.run_training(n_episodes=episodes)
    
    # Dreamer
    print("\n▸ Running WITH DREAMER...")
    dreamer = DreamerAtariRunner(
        game=game, dreamer_enabled=True
    )
    dreamer_stats = dreamer.run_training(n_episodes=episodes)
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Dreamer':>12} {'Delta':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    
    for key, label in [
        ('mean_reward', 'Mean reward'),
        ('max_reward', 'Max reward'),
        ('final_mean_reward', 'Final 20 avg'),
    ]:
        b = baseline_stats[key]
        d = dreamer_stats[key]
        delta = d - b
        sign = "+" if delta > 0 else ""
        print(f"  {label:<23} {b:>10.1f}   {d:>10.1f}   {sign}{delta:>8.1f}")
    
    print(f"\n{'='*70}\n")
    
    return {'baseline': baseline_stats, 'dreamer': dreamer_stats}


def main():
    parser = argparse.ArgumentParser(description="Atari with Dreamer")
    parser.add_argument('--game', type=str, default='Breakout', help="Atari game name")
    parser.add_argument('--episodes', type=int, default=100, help="Episodes")
    parser.add_argument('--max-steps', type=int, default=10000, help="Max steps/ep")
    parser.add_argument('--no-dreamer', action='store_true', help="Disable dreamer")
    parser.add_argument('--compare', action='store_true', help="Run A/B comparison")
    parser.add_argument('--nudge', type=float, default=0.3, help="Nudge strength")
    parser.add_argument('--dream-steps', type=int, default=30, help="Dream lookahead")
    parser.add_argument('--output', type=str, default=None, help="Save results JSON")
    args = parser.parse_args()
    
    if args.compare:
        results = run_comparison(game=args.game, episodes=args.episodes)
    else:
        runner = DreamerAtariRunner(
            game=args.game,
            max_steps=args.max_steps,
            dreamer_enabled=not args.no_dreamer,
            dream_steps=args.dream_steps,
            nudge_strength=args.nudge,
        )
        results = runner.run_training(
            n_episodes=args.episodes, verbose=True
        )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
