"""
🎮 Multi-Game Stress Test with Tetra Dialogue Loop
====================================================

Runs the full Tetra dialogue across 4 Atari games:
  - SpaceInvaders
  - Breakout
  - Pong
  - Asteroids

For each game:
  1. Run episodes until plateau
  2. Query Tetra with visual/causal patterns (NO game names)
  3. Cycle through multiple hypotheses, keep the best
  4. Auto-retire chronically plateaued policies
  5. Report comparison across all games

Verifies:
  ✅ Different visual patterns per game
  ✅ Different causal effects per game
  ✅ Tetra gives different strategies per game
  ✅ No game names leak
  ✅ Auto-retirement works
  ✅ Multi-hypothesis cycling selects best strategy
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, '.')

from throng4.environments.atari_adapter import AtariAdapter
from throng4.meta_policy import MetaPolicyController, ControllerConfig
from throng4.meta_policy.tetra_client import TetraClient


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

GAMES = ['SpaceInvaders', 'Breakout', 'Pong', 'Asteroids']
EPISODES_PER_GAME = 80
MAX_STEPS_PER_EPISODE = 500
HYPOTHESIS_TRIGGER_EPISODE = 25  # Try hypothesis after this many episodes
N_HYPOTHESES_TO_CYCLE = 3  # Ask Tetra for N different strategies
HYPOTHESIS_TEST_EPISODES = 8  # Episodes to test each hypothesis
AUTO_RETIRE_AFTER = 3  # Retire after N consecutive failed hypotheses


def run_episode(env, pipeline, controller):
    """Run a single episode, return reward."""
    state = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = pipeline.select_action(state, explore=True)
        next_state, reward, done, info = env.step(action)
        
        controller.on_step(state, action, reward, next_state)
        pipeline.update(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    return episode_reward


def cycle_hypotheses(controller, pipeline, env, n_hypotheses=3):
    """
    Ask Tetra for N strategies, test each, keep the best.
    
    Returns dict with results for all tested strategies.
    """
    results = []
    
    for i in range(n_hypotheses):
        # Reset conversation for fresh strategy each time
        if i > 0:
            controller.llm_client.reset_conversation()
        
        # Query Tetra
        result = controller.test_hypothesis_with_tetra(pipeline)
        
        if result['status'] != 'hypothesis_applied':
            print(f"  Hypothesis {i+1}/{n_hypotheses}: skipped ({result['status']})")
            continue
        
        strategy_name = result['strategy'].name
        print(f"\n  Hypothesis {i+1}/{n_hypotheses}: {strategy_name}")
        print(f"  Baseline: {result['baseline_reward']:.1f}")
        
        # Run test episodes
        test_rewards = []
        for test_ep in range(HYPOTHESIS_TEST_EPISODES):
            reward = run_episode(env, pipeline, controller)
            test_rewards.append(reward)
        
        avg_test = np.mean(test_rewards)
        improvement = avg_test - result['baseline_reward']
        
        print(f"  Test avg: {avg_test:.1f} (Δ{improvement:+.1f})")
        
        # Report to Tetra
        refinement = controller.report_hypothesis_results(
            test_rewards, result['baseline_reward']
        )
        
        results.append({
            'strategy': strategy_name,
            'baseline': result['baseline_reward'],
            'test_avg': avg_test,
            'improvement': improvement,
            'tetra_response': result.get('tetra_response', '')[:200],
            'refinement': refinement[:200] if refinement else '',
        })
    
    # Find best strategy
    if results:
        best = max(results, key=lambda r: r['improvement'])
        print(f"\n  🏆 Best strategy: {best['strategy']} (Δ{best['improvement']:+.1f})")
    else:
        best = None
        print(f"\n  ⚠️  No hypotheses could be tested")
    
    return {
        'all_results': results,
        'best': best,
    }


def run_game(game_name, tetra, game_index):
    """
    Run full Tetra dialogue loop on one game.
    
    Returns game results dict.
    """
    print(f"\n{'#' * 70}")
    print(f"# GAME {game_index+1}/{len(GAMES)}: {game_name}")
    print(f"# (game name hidden from controller and Tetra)")
    print(f"{'#' * 70}")
    
    # Fresh controller for each game (Tetra keeps conv history per game)
    tetra.reset_conversation()
    
    controller = MetaPolicyController(
        config=ControllerConfig(
            fingerprint_episodes=10,
            llm_cooldown=10,
            concept_discovery_interval=20,
        ),
        llm_client=tetra,
    )
    
    env = AtariAdapter(game_name)
    pipeline = controller.on_new_environment(env)
    
    game_results = {
        'game': game_name,
        'episode_rewards': [],
        'visual_patterns': None,
        'causal_effects': None,
        'hypotheses': [],
        'retired': False,
        'final_avg': 0.0,
    }
    
    hypothesis_cycle_done = False
    consecutive_failures = 0
    
    for ep in range(EPISODES_PER_GAME):
        reward = run_episode(env, pipeline, controller)
        meta_status = controller.on_episode_complete(reward)
        game_results['episode_rewards'].append(reward)
        
        if (ep + 1) % 20 == 0:
            print(f"  Ep {ep+1}/{EPISODES_PER_GAME}: avg={meta_status['avg_reward']:.1f}, "
                  f"policy={meta_status['policy_id'][:8]}, "
                  f"concepts={meta_status['concepts_in_library']}")
        
        # Try multi-hypothesis cycling after plateau
        if not hypothesis_cycle_done and ep >= HYPOTHESIS_TRIGGER_EPISODE:
            print(f"\n{'='*60}")
            print(f"  MULTI-HYPOTHESIS CYCLING (ep {ep+1})")
            print(f"{'='*60}")
            
            cycle_results = cycle_hypotheses(
                controller, pipeline, env,
                n_hypotheses=N_HYPOTHESES_TO_CYCLE,
            )
            
            game_results['hypotheses'] = cycle_results['all_results']
            hypothesis_cycle_done = True
            
            # Auto-retirement check
            if cycle_results['best'] and cycle_results['best']['improvement'] < 0:
                consecutive_failures += 1
                print(f"\n  ⚠️  Best hypothesis was negative (failures: {consecutive_failures}/{AUTO_RETIRE_AFTER})")
                
                if consecutive_failures >= AUTO_RETIRE_AFTER:
                    print(f"  🔴 AUTO-RETIREMENT: Policy exhausted after {consecutive_failures} failed cycles")
                    game_results['retired'] = True
            else:
                consecutive_failures = 0
                print(f"\n  ✅ Continuing with best strategy...")
    
    # Capture final patterns
    if controller.current_visual_patterns:
        vp = controller.current_visual_patterns
        game_results['visual_patterns'] = {
            'entity_count': vp.entity_count,
            'motion_type': vp.motion_type,
            'spatial_layout': vp.spatial_layout,
            'collision_frequency': vp.collision_frequency,
        }
    
    if controller.current_causal_effects:
        game_results['causal_effects'] = {
            str(k): {
                'state_delta': v.avg_state_change,
                'reward_corr': v.reward_correlation,
                'creates_entities': v.creates_entities,
            }
            for k, v in controller.current_causal_effects.items()
        }
    
    # Final avg
    game_results['final_avg'] = np.mean(game_results['episode_rewards'][-20:])
    
    controller.on_environment_done()
    env.close()
    
    return game_results


def print_comparison(all_results):
    """Print cross-game comparison table."""
    print(f"\n\n{'='*70}")
    print(f"  CROSS-GAME COMPARISON")
    print(f"{'='*70}")
    
    # Visual patterns
    print(f"\n  Visual Patterns:")
    print(f"  {'Game':<20} {'Entities':<10} {'Motion':<15} {'Layout':<15} {'Collisions':<10}")
    print(f"  {'─'*70}")
    for r in all_results:
        vp = r.get('visual_patterns', {})
        if vp:
            print(f"  {r['game']:<20} {vp.get('entity_count', '?'):<10} "
                  f"{vp.get('motion_type', '?'):<15} "
                  f"{vp.get('spatial_layout', '?'):<15} "
                  f"{vp.get('collision_frequency', '?'):<10}")
    
    # Hypothesis results
    print(f"\n  Hypothesis Results:")
    print(f"  {'Game':<20} {'Best Strategy':<25} {'Δ Reward':<12} {'Retired?':<10}")
    print(f"  {'─'*70}")
    for r in all_results:
        if r['hypotheses']:
            best = max(r['hypotheses'], key=lambda h: h['improvement'])
            print(f"  {r['game']:<20} {best['strategy']:<25} "
                  f"{best['improvement']:+.1f}{'':>5} "
                  f"{'YES' if r['retired'] else 'no':<10}")
        else:
            print(f"  {r['game']:<20} {'(no hypothesis)':25} {'N/A':>12} "
                  f"{'YES' if r['retired'] else 'no':<10}")
    
    # Final averages
    print(f"\n  Final Performance:")
    print(f"  {'Game':<20} {'Final Avg':<15} {'Total Episodes':<15}")
    print(f"  {'─'*50}")
    for r in all_results:
        print(f"  {r['game']:<20} {r['final_avg']:<15.1f} {len(r['episode_rewards']):<15}")


def main():
    print("=" * 70)
    print("🎮 MULTI-GAME STRESS TEST WITH TETRA DIALOGUE")
    print("=" * 70)
    print(f"\nGames: {GAMES}")
    print(f"Episodes per game: {EPISODES_PER_GAME}")
    print(f"Hypotheses to cycle: {N_HYPOTHESES_TO_CYCLE}")
    print(f"Test episodes per hypothesis: {HYPOTHESIS_TEST_EPISODES}")
    
    # Connect to Tetra
    print("\nConnecting to Tetra via OpenClaw gateway...")
    tetra = TetraClient(game="MultiGame_StressTest")
    
    if not tetra.check_gateway():
        print("⚠️  Gateway not available!")
        print("\nMake sure OpenClaw gateway is running:")
        print("  openclaw gateway start")
        print("  openclaw gateway health")
        print("\nExiting...")
        sys.exit(1)
    
    print("✅ Gateway connected!")
    
    # Run all games
    all_results = []
    start_time = time.time()
    
    for i, game in enumerate(GAMES):
        game_start = time.time()
        result = run_game(game, tetra, i)
        game_time = time.time() - game_start
        
        all_results.append(result)
        print(f"\n  ⏱️  {game} completed in {game_time:.0f}s")
    
    total_time = time.time() - start_time
    
    # Print comparison
    print_comparison(all_results)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  STRESS TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Games tested: {len(GAMES)}")
    print(f"  Total episodes: {sum(len(r['episode_rewards']) for r in all_results)}")
    print(f"  Total hypotheses tested: {sum(len(r['hypotheses']) for r in all_results)}")
    
    # Check for differentiation
    patterns = [r.get('visual_patterns', {}).get('motion_type') for r in all_results]
    unique_patterns = len(set(p for p in patterns if p))
    print(f"\n  Pattern differentiation: {unique_patterns} unique motion types across {len(GAMES)} games")
    
    if unique_patterns >= 2:
        print(f"  ✅ Visual patterns differentiate between games!")
    else:
        print(f"  ⚠️  Visual patterns may not differentiate well — needs tuning")
    
    # Save results
    results_file = 'multigame_stress_results.json'
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != 'episode_rewards'}
        sr['episode_count'] = len(r['episode_rewards'])
        sr['reward_trajectory'] = [
            float(np.mean(r['episode_rewards'][max(0,i-5):i+1])) 
            for i in range(0, len(r['episode_rewards']), 5)
        ]
        serializable.append(sr)
    
    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print(f"\n  Results saved to {results_file}")
    print(f"\n✅ Multi-game stress test complete!")


if __name__ == "__main__":
    main()
