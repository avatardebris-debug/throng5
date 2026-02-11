"""
Test: MAML vs EWC on GridWorld Transfer (Phase 3)

Validates that task-conditioned MAML outperforms EWC baseline on RL tasks.

Baseline (from test_compound_corrected.py): +0.250 mean, 67% positive
Goal: MAML should exceed this on GridWorld RL transfer.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from throng3.pipeline import MetaNPipeline
from throng3.envs.gridworld import GridWorld, create_gridworld_variants


def train_gridworld(pipeline, env, episodes: int, task_name: str = "Task", 
                   epsilon_start: float = 1.0, epsilon_end: float = 0.1):
    """
    Train on GridWorld environment with ε-greedy exploration.
    
    Uses single-step step_rl() API for correct TD timing:
    each env transition = exactly one pipeline call.
    
    Args:
        pipeline: MetaNPipeline to train
        env: GridWorld environment
        episodes: Number of training episodes
        task_name: Name for logging
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        
    Returns:
        Mean reward over last 20 episodes
    """
    total_rewards = []
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        prev_action = None
        prev_reward = 0.0
        
        # Decay epsilon linearly
        epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)
        
        while not done and steps < 50:
            # Single step: pass state + reward from previous action
            result = pipeline.step_rl(
                state, 
                reward=prev_reward, 
                done=False, 
                prev_action=prev_action
            )
            output = result['output']
            
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(len(output))
            else:
                action = np.argmax(output)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Store for next iteration
            prev_action = action
            prev_reward = reward
            state = next_state
            steps += 1
        
        # Final learning step with terminal reward
        if done:
            pipeline.step_rl(state, reward=prev_reward, done=True, prev_action=prev_action)
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards[-20:])  # Last 20 episodes
    print(f"  {task_name}: {episodes} episodes, mean reward={mean_reward:.3f}")
    return mean_reward


def test_maml_vs_ewc_gridworld():
    """
    Compare MAML vs EWC on GridWorld compound transfer.
    
    Curriculum: Empty → Obstacles → Different Goal
    """
    print("\n" + "="*70)
    print("PHASE 3: MAML vs EWC on GridWorld Transfer")
    print("="*70)
    
    print("\nTask: Grid navigation (5x5 variants)")
    print("Curriculum: Empty → Obstacles → Different Goal")
    print("Seeds: 3")
    print("\nBaseline to beat: +0.250 mean, 67% positive (EWC)")
    
    variants = create_gridworld_variants()
    env_a = variants['empty_5x5']  # Goal at (4,4)
    env_b = variants['obstacles_5x5']  # Obstacles + goal at (4,4)
    env_c = GridWorld(size=5, obstacles=[], goal=(3, 4))  # Different goal
    
    train_episodes = 200
    test_episodes = 50
    n_seeds = 3
    
    results_ewc = []
    results_maml = []
    baselines = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print("="*70)
        np.random.seed(seed)
        
        # Baseline: C cold (no transfer)
        print("\n[Baseline: Cold Start on C]")
        pipeline_cold = MetaNPipeline.create_with_ewc(
            n_inputs=25,  # 5x5 grid
            n_outputs=4,  # 4 actions
            ewc_lambda=1000.0
        )
        baseline = train_gridworld(pipeline_cold, env_c, test_episodes, "C (cold)")
        baselines.append(baseline)
        
        # EWC Baseline
        print("\n[EWC Baseline]")
        pipeline_ewc = MetaNPipeline.create_with_ewc(
            n_inputs=25,
            n_outputs=4,
            ewc_lambda=1000.0
        )
        
        train_gridworld(pipeline_ewc, env_a, train_episodes, "A (empty)")
        pipeline_ewc.consolidate_task()
        pipeline_ewc.reset_task_state()
        
        train_gridworld(pipeline_ewc, env_b, train_episodes, "B (obstacles)")
        pipeline_ewc.consolidate_task()
        pipeline_ewc.reset_task_state()
        
        reward_ewc = train_gridworld(pipeline_ewc, env_c, test_episodes, "C (diff goal)")
        improvement_ewc = reward_ewc - baseline
        results_ewc.append(improvement_ewc)
        print(f"  EWC improvement: {improvement_ewc:+.3f}")
        
        # MAML
        print("\n[MAML]")
        pipeline_maml = MetaNPipeline.create_with_maml(
            n_inputs=25,
            n_outputs=4,
            meta_lr=0.001
        )
        
        # Get MAML layer for verification
        maml_layer = pipeline_maml.stack.get_layer(3)
        
        # Train on A, collect experience, meta-update
        train_gridworld(pipeline_maml, env_a, train_episodes, "A (empty)")
        pipeline_maml.consolidate_maml_task()  # Convert experience → meta-update
        pipeline_maml.reset_task_state()
        
        # Train on B
        train_gridworld(pipeline_maml, env_b, train_episodes, "B (obstacles)")
        pipeline_maml.consolidate_maml_task()  # Convert experience → meta-update
        pipeline_maml.reset_task_state()
        
        # Test on C
        reward_maml = train_gridworld(pipeline_maml, env_c, test_episodes, "C (diff goal)")
        improvement_maml = reward_maml - baseline
        results_maml.append(improvement_maml)
        print(f"  MAML improvement: {improvement_maml:+.3f}")
        
        print(f"\n  Seed {seed} summary:")
        print(f"    Baseline (cold): {baseline:.3f}")
        print(f"    EWC: {reward_ewc:.3f} ({improvement_ewc:+.3f})")
        print(f"    MAML: {reward_maml:.3f} ({improvement_maml:+.3f})")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mean_baseline = np.mean(baselines)
    
    mean_ewc = np.mean(results_ewc)
    std_ewc = np.std(results_ewc)
    positive_ewc = sum(1 for x in results_ewc if x > 0) / len(results_ewc) * 100
    
    mean_maml = np.mean(results_maml)
    std_maml = np.std(results_maml)
    positive_maml = sum(1 for x in results_maml if x > 0) / len(results_maml) * 100
    
    print(f"\nBaseline (C cold): {mean_baseline:.3f} reward")
    
    print(f"\nEWC Baseline:")
    print(f"  Mean improvement: {mean_ewc:+.3f} reward")
    print(f"  Std: {std_ewc:.3f}")
    print(f"  Positive rate: {positive_ewc:.0f}%")
    
    print(f"\nMAML:")
    print(f"  Mean improvement: {mean_maml:+.3f} reward")
    print(f"  Std: {std_maml:.3f}")
    print(f"  Positive rate: {positive_maml:.0f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("MAML vs EWC")
    print("="*70)
    
    delta = mean_maml - mean_ewc
    print(f"\nMAML advantage: {delta:+.3f} reward")
    
    target_mean = 0.250
    target_positive = 67.0
    
    if mean_maml > target_mean and positive_maml >= target_positive:
        print(f"\n✓ MAML EXCEEDS BASELINE!")
        print(f"  - Mean: {mean_maml:.3f} > {target_mean:.3f} ✓")
        print(f"  - Positive: {positive_maml:.0f}% >= {target_positive:.0f}% ✓")
        print(f"  - Task conditioning works on RL!")
    elif mean_maml > mean_ewc:
        print(f"\n⚠ MAML IMPROVES OVER EWC")
        print(f"  - But below target ({target_mean:.3f} mean, {target_positive:.0f}% positive)")
        print(f"  - May need more meta-updates or tuning")
    else:
        print(f"\n✗ MAML UNDERPERFORMS")
        print(f"  - EWC is better: {mean_ewc:+.3f} vs {mean_maml:+.3f}")
        print(f"  - Need to debug meta-learning")
    
    return {
        'ewc': {'mean': mean_ewc, 'std': std_ewc, 'positive': positive_ewc},
        'maml': {'mean': mean_maml, 'std': std_maml, 'positive': positive_maml},
    }


if __name__ == '__main__':
    results = test_maml_vs_ewc_gridworld()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nEWC: {results['ewc']['mean']:+.3f} reward ({results['ewc']['positive']:.0f}% positive)")
    print(f"MAML: {results['maml']['mean']:+.3f} reward ({results['maml']['positive']:.0f}% positive)")
    
    if results['maml']['mean'] > 0.250:
        print("\n✓ Phase 3 complete — MAML validated on GridWorld!")
        print("  Ready for Phase 4: Analysis")
    else:
        print("\n⚠ MAML needs tuning or meta-update implementation")
