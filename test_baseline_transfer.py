"""
Transfer learning benchmark using simple baseline.
Tests if pretraining on one task helps on another.
"""

import numpy as np
from simple_baseline import SimplePolicyNetwork, train_simple_baseline
from throng3.environments import GridWorldAdapter, CartPoleAdapter
from throng3.benchmarks.stats import StatisticalAnalyzer
import json
from datetime import datetime


def measure_convergence(env, network, max_steps: int = 500, target_return: float = 0.8):
    """
    Measure steps to reach target performance.
    
    Returns:
        steps_to_convergence: Number of episodes to reach target, or max_steps
    """
    episode_returns = []
    recent_returns = []
    
    for episode in range(max_steps):
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
        episode_returns.append(episode_reward)
        recent_returns.append(episode_reward)
        if len(recent_returns) > 10:
            recent_returns.pop(0)
        
        # Check convergence
        if len(recent_returns) >= 10:
            avg_return = np.mean(recent_returns)
            if avg_return >= target_return:
                return episode + 1
    
    return max_steps


def run_transfer_experiment(
    source_env_name: str,
    target_env_name: str,
    n_seeds: int = 10,
    pretrain_episodes: int = 50,
    max_eval_episodes: int = 100,
    verbose: bool = True
):
    """
    Run transfer learning experiment.
    
    Compares:
    - Fresh: Train from scratch on target task
    - Pretrained: Pretrain on source, then train on target
    
    Returns:
        results dict with statistics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Transfer Learning: {source_env_name} → {target_env_name}")
        print(f"{'='*60}")
        print(f"Seeds: {n_seeds}")
        print(f"Pretrain episodes: {pretrain_episodes}")
        print(f"Max eval episodes: {max_eval_episodes}\n")
    
    fresh_steps = []
    pretrained_steps = []
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        
        if verbose:
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")
        
        # Create environments
        if source_env_name == "gridworld":
            source_env = GridWorldAdapter()
            n_inputs_source = 2
            n_outputs_source = 4
        else:  # cartpole
            source_env = CartPoleAdapter()
            n_inputs_source = 4
            n_outputs_source = 2
        
        if target_env_name == "gridworld":
            target_env = GridWorldAdapter()
            n_inputs_target = 2
            n_outputs_target = 4
            target_return = 0.5  # GridWorld target
        else:  # cartpole
            target_env = CartPoleAdapter()
            n_inputs_target = 4
            n_outputs_target = 2
            target_return = 50.0  # CartPole target
        
        # FRESH: Train from scratch on target
        fresh_network = SimplePolicyNetwork(
            n_inputs=n_inputs_target,
            n_hidden=128,
            n_outputs=n_outputs_target,
            lr=0.001
        )
        fresh_result = measure_convergence(target_env, fresh_network, max_eval_episodes, target_return)
        fresh_steps.append(fresh_result)
        
        if verbose:
            print(f"  Fresh: {fresh_result} episodes")
        
        # PRETRAINED: Pretrain on source, then fine-tune on target
        if n_inputs_source == n_inputs_target and n_outputs_source == n_outputs_target:
            # Same dimensions - can transfer weights directly
            pretrained_network = SimplePolicyNetwork(
                n_inputs=n_inputs_source,
                n_hidden=128,
                n_outputs=n_outputs_source,
                lr=0.001
            )
            
            # Pretrain on source
            train_simple_baseline(source_env, n_episodes=pretrain_episodes, verbose=False)
            
            # Fine-tune on target (network already initialized)
            pretrained_result = measure_convergence(target_env, pretrained_network, max_eval_episodes, target_return)
        else:
            # Different dimensions - can't transfer directly
            # Just use fresh network (no transfer possible)
            pretrained_result = fresh_result
        
        pretrained_steps.append(pretrained_result)
        
        if verbose:
            speedup = fresh_result / max(pretrained_result, 1)
            print(f"  Pretrained: {pretrained_result} episodes")
            print(f"  Speedup: {speedup:.2f}x")
    
    # Analyze results
    analyzer = StatisticalAnalyzer()
    stats = analyzer.summarize_results(pretrained_steps, fresh_steps)
    
    results = {
        'source_task': source_env_name,
        'target_task': target_env_name,
        'n_seeds': n_seeds,
        'pretrain_episodes': pretrain_episodes,
        'fresh_steps': fresh_steps,
        'pretrained_steps': pretrained_steps,
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        fresh_ci_lower, fresh_ci_upper = stats['fresh_ci']
        pretrained_ci_lower, pretrained_ci_upper = stats['pretrained_ci']
        
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        print(f"Fresh (no transfer):     {stats['fresh_mean']:.1f} episodes (95% CI: {fresh_ci_lower:.1f}-{fresh_ci_upper:.1f})")
        print(f"Pretrained (transfer):   {stats['pretrained_mean']:.1f} episodes (95% CI: {pretrained_ci_lower:.1f}-{pretrained_ci_upper:.1f})")
        print(f"Speedup:                 {stats['speedup']:.2f}x")
        print(f"\nStatistical Significance:")
        print(f"  t-statistic:             {stats['t_statistic']:.3f}")
        print(f"  p-value:                 {stats['p_value']:.4f}")
        print(f"  Effect size (Cohen's d): {stats['effect_size']:.3f}")
        print(f"  Significant (p<0.05):    {'✓ YES' if stats['significant'] else '✗ NO'}")
        print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    import os
    
    # Create results directory
    os.makedirs("results/baseline_transfer", exist_ok=True)
    
    print("="*60)
    print("Baseline Transfer Learning Validation")
    print("="*60)
    
    # Test 1: CartPole → CartPole (sanity check - should show transfer)
    print("\n### Test 1: CartPole → CartPole (Same Task Transfer) ###")
    results_cp_cp = run_transfer_experiment(
        "cartpole", "cartpole",
        n_seeds=10,
        pretrain_episodes=30,
        max_eval_episodes=100
    )
    
    # Save results
    with open("results/baseline_transfer/cartpole_to_cartpole.json", "w") as f:
        json.dump(results_cp_cp, f, indent=2)
    
    print("\n✓ Baseline transfer test complete!")
    print(f"✓ Results saved to results/baseline_transfer/")
