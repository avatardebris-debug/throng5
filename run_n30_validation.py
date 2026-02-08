"""
Run N=30 transfer learning validation with 5K neurons.
"""

import numpy as np
import json
from datetime import datetime
from throng3.environments import GridWorldAdapter
from throng3.benchmarks.stats import StatisticalAnalyzer
from train_5k_baseline import ImprovedPolicyNetwork


def measure_convergence_5k(env, network=None, max_episodes=200, target_return=0.85):
    """
    Measure episodes to reach target performance with 5K neurons.
    
    Args:
        env: Environment
        network: Pre-trained network (None for fresh)
        max_episodes: Maximum episodes
        target_return: Target average return over 10 episodes
    
    Returns:
        episodes_to_convergence
    """
    # Create network if not provided
    if network is None:
        network = ImprovedPolicyNetwork(
            n_inputs=2,
            n_hidden=5000,
            n_outputs=4,
            lr=0.001,
            epsilon_start=0.3,
            epsilon_end=0.01
        )
    
    recent_returns = []
    
    for episode in range(max_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            _, action = network.forward(obs, training=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            network.store_reward(reward)
        
        network.update()
        network.decay_epsilon()
        
        recent_returns.append(episode_reward)
        if len(recent_returns) > 10:
            recent_returns.pop(0)
        
        # Check convergence
        if len(recent_returns) >= 10:
            avg_return = np.mean(recent_returns)
            if avg_return >= target_return:
                return episode + 1
    
    return max_episodes


def run_n30_experiment(n_seeds=30, pretrain_episodes=250, max_eval_episodes=200):
    """Run N=30 transfer learning experiment."""
    print("="*60)
    print("N=30 Transfer Learning Validation (5000 neurons)")
    print("="*60)
    print(f"Seeds: {n_seeds}")
    print(f"Pretrain episodes: {pretrain_episodes}")
    print(f"Max eval episodes: {max_eval_episodes}")
    print(f"Target: 0.85 avg return over 10 episodes")
    print("="*60)
    
    fresh_steps = []
    pretrained_steps = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        
        env = GridWorldAdapter()
        
        # FRESH: Train from scratch
        fresh_result = measure_convergence_5k(env, network=None, max_episodes=max_eval_episodes)
        fresh_steps.append(fresh_result)
        print(f"  Fresh: {fresh_result} episodes")
        
        # PRETRAINED: Pretrain then fine-tune
        # Create and pretrain network
        pretrained_net = ImprovedPolicyNetwork(
            n_inputs=2,
            n_hidden=5000,
            n_outputs=4,
            lr=0.001,
            epsilon_start=0.3,
            epsilon_end=0.01
        )
        
        # Pretrain on GridWorld
        for ep in range(pretrain_episodes):
            obs = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:
                _, action = pretrained_net.forward(obs, training=True)
                obs, reward, done, info = env.step(action)
                steps += 1
                pretrained_net.store_reward(reward)
            
            pretrained_net.update()
            pretrained_net.decay_epsilon()
        
        # Reset epsilon for fine-tuning
        pretrained_net.epsilon = 0.1
        
        # Fine-tune (measure convergence)
        pretrained_result = measure_convergence_5k(env, network=pretrained_net, max_episodes=max_eval_episodes)
        pretrained_steps.append(pretrained_result)
        
        speedup = fresh_result / max(pretrained_result, 1)
        print(f"  Pretrained: {pretrained_result} episodes")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Analyze results
    analyzer = StatisticalAnalyzer()
    stats = analyzer.summarize_results(pretrained_steps, fresh_steps)
    
    # Save results
    results = {
        'n_seeds': n_seeds,
        'pretrain_episodes': pretrain_episodes,
        'n_hidden': 5000,
        'fresh_steps': fresh_steps,
        'pretrained_steps': pretrained_steps,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"results/phase3/gridworld_5k_n{n_seeds}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    fresh_ci_lower, fresh_ci_upper = stats['fresh_ci']
    pretrained_ci_lower, pretrained_ci_upper = stats['pretrained_ci']
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Fresh (no transfer):     {stats['fresh_mean']:.1f} episodes (95% CI: {fresh_ci_lower:.1f}-{fresh_ci_upper:.1f})")
    print(f"Pretrained (transfer):   {stats['pretrained_mean']:.1f} episodes (95% CI: {pretrained_ci_lower:.1f}-{pretrained_ci_upper:.1f})")
    print(f"Speedup:                 {stats['speedup']:.2f}x")
    print(f"\nStatistical Significance:")
    print(f"  t-statistic:             {stats['t_statistic']:.3f}")
    print(f"  p-value:                 {stats['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {stats['effect_size']:.3f}")
    print(f"  Significant (p<0.05):    {'✓ YES' if stats['significant'] else '✗ NO'}")
    print(f"{'='*60}")
    print(f"\n✓ Results saved to: {filename}")
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results/phase3", exist_ok=True)
    
    results = run_n30_experiment(n_seeds=30, pretrain_episodes=250, max_eval_episodes=200)
    
    print("\n" + "="*60)
    if results['statistics']['significant']:
        print("✓ PHASE 3 VALIDATION SUCCESSFUL!")
        print(f"  Transfer learning validated with p={results['statistics']['p_value']:.4f}")
        print(f"  Speedup: {results['statistics']['speedup']:.2f}x")
    else:
        print("⚠ Transfer effect not statistically significant")
        print(f"  p-value: {results['statistics']['p_value']:.4f}")
    print("="*60)
