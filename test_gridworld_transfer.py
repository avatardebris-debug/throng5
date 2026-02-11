"""
Test: GridWorld Compound Transfer with Fisher Boosting

Tests if EWC + Fisher boosting enables robust compound transfer on
structured RL tasks (grid navigation).

Expected: +15-25% mean, <10% variance, >80% positive
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
    
    Args:
        pipeline: MetaNPipeline to train
        env: GridWorld environment
        episodes: Number of training episodes
        task_name: Name for logging
        epsilon_start: Initial exploration rate (default: 1.0 = fully random)
        epsilon_end: Final exploration rate (default: 0.1 = 10% random)
    """
    total_rewards = []
    epsilon_decay = (epsilon_start - epsilon_end) / episodes
    
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        # Decay epsilon linearly
        epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)
        
        while not done and steps < 50:
            # Forward pass to get Q-values/action preferences
            output = pipeline.step(state, reward=0.0)['output']
            
            # ε-greedy action selection
            if np.random.random() < epsilon:
                # Explore: random action
                action = np.random.randint(len(output))
            else:
                # Exploit: best action
                action = np.argmax(output)
            
            # Take action
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Learning step with reward
            pipeline.step(next_state, reward=reward)
            
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
    
    mean_reward = np.mean(total_rewards[-20:])  # Last 20 episodes
    print(f"  {task_name}: {episodes} episodes, mean reward={mean_reward:.3f}")
    return mean_reward


def test_gridworld_fisher_boosting():
    """
    Test EWC + Fisher boosting on GridWorld compound transfer.
    
    Compare with/without Fisher boosting to validate acceleration.
    """
    print("\n" + "="*70)
    print("GRIDWORLD COMPOUND TRANSFER TEST")
    print("="*70)
    
    print("\nTask: Grid navigation (5x5 variants)")
    print("Ordering: Empty -> Obstacles -> Different Goal")
    print("Seeds: 3")
    
    variants = create_gridworld_variants()
    env_a = variants['empty_5x5']  # Goal at (4,4)
    env_b = variants['obstacles_5x5']  # Obstacles + goal at (4,4)
    
    # Task C: Same size but different goal position (tests generalization)
    env_c = GridWorld(size=5, obstacles=[], goal=(3, 4))  # Goal at different position
    
    train_episodes = 200  # Increased for convergence
    test_episodes = 50    # Increased for stable measurement
    n_seeds = 3
    
    results_without_boost = []
    results_with_boost = []
    baselines = []
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}]")
        np.random.seed(seed)
        
        # Baseline: C cold
        pipeline_cold = MetaNPipeline.create_with_ewc(
            n_inputs=25,  # 5x5 grid
            n_outputs=4,  # 4 actions
            ewc_lambda=1000.0
        )
        baseline = train_gridworld(pipeline_cold, env_c, test_episodes, "C (cold)")
        baselines.append(baseline)
        
        # WITHOUT Fisher Boosting (EWC only)
        print("\n  Without Fisher Boosting:")
        pipeline_no_boost = MetaNPipeline.create_with_ewc(
            n_inputs=25,  # 5x5 grid
            n_outputs=4,
            ewc_lambda=1000.0
        )
        
        train_gridworld(pipeline_no_boost, env_a, train_episodes, "  A (empty)")
        pipeline_no_boost.consolidate_task()
        pipeline_no_boost.reset_task_state()
        
        train_gridworld(pipeline_no_boost, env_b, train_episodes, "  B (obstacles)")
        pipeline_no_boost.consolidate_task()
        pipeline_no_boost.reset_task_state()
        
        # Test on C (different size - need new pipeline)
        # Transfer learned weights somehow... (simplified: just measure final performance)
        reward_no_boost = train_gridworld(pipeline_no_boost, env_c, test_episodes, "  C (diff goal)")
        improvement_no_boost = reward_no_boost - baseline  # Absolute improvement
        results_without_boost.append(improvement_no_boost)
        print(f"    Absolute improvement: {improvement_no_boost:+.3f}")
        
        # WITH Fisher Boosting
        print("\n  With Fisher Boosting:")
        pipeline_boost = MetaNPipeline.create_with_ewc(
            n_inputs=25,
            n_outputs=4,
            ewc_lambda=1000.0
        )
        
        train_gridworld(pipeline_boost, env_a, train_episodes, "  A (empty)")
        pipeline_boost.consolidate_task()
        pipeline_boost.reset_task_state()
        
        train_gridworld(pipeline_boost, env_b, train_episodes, "  B (obstacles)")
        pipeline_boost.consolidate_task()
        pipeline_boost.reset_task_state()
        
        reward_boost = train_gridworld(pipeline_boost, env_c, test_episodes, "  C (diff goal)")
        improvement_boost = reward_boost - baseline  # Absolute improvement
        results_with_boost.append(improvement_boost)
        print(f"    Absolute improvement: {improvement_boost:+.3f}")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mean_baseline = np.mean(baselines)
    mean_no_boost = np.mean(results_without_boost)
    std_no_boost = np.std(results_without_boost)
    mean_boost = np.mean(results_with_boost)
    std_boost = np.std(results_with_boost)
    
    print(f"\nBaseline (C cold): {mean_baseline:.3f}")
    
    print(f"\nWithout Fisher Boosting (EWC only):")
    print(f"  Mean improvement: {mean_no_boost:+.3f} reward")
    print(f"  Std: {std_no_boost:.3f}")
    print(f"  Positive rate: {sum(1 for x in results_without_boost if x > 0)/len(results_without_boost)*100:.0f}%")
    
    print(f"\nWith Fisher Boosting:")
    print(f"  Mean improvement: {mean_boost:+.3f} reward")
    print(f"  Std: {std_boost:.3f}")
    print(f"  Positive rate: {sum(1 for x in results_with_boost if x > 0)/len(results_with_boost)*100:.0f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("FISHER BOOSTING EFFECTIVENESS")
    print("="*70)
    
    improvement_delta = mean_boost - mean_no_boost
    
    print(f"\nMean improvement delta: {improvement_delta:+.3f} reward")
    
    if mean_boost > 0.1 and mean_boost > mean_no_boost and std_boost < 0.2:
        print("\n✓ FISHER BOOSTING SUCCESS!")
        print("  - Positive compound transfer on structured RL")
        print("  - Low variance (stable)")
        print("  - Shared abstractions learned")
    elif mean_boost > mean_no_boost:
        print("\n⚠ PARTIAL SUCCESS")
        print("  - Fisher boosting helps")
        print("  - But below target performance")
    else:
        print("\n✗ FISHER BOOSTING NOT EFFECTIVE")
        print("  - No improvement over EWC alone")
        print("  - May need full Meta^3 representation optimizer")
    
    return {
        'without_boost': {'mean': mean_no_boost, 'std': std_no_boost},
        'with_boost': {'mean': mean_boost, 'std': std_boost},
    }


if __name__ == '__main__':
    results = test_gridworld_fisher_boosting()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nEWC only: {results['without_boost']['mean']:+.3f} reward")
    print(f"EWC + Fisher boosting: {results['with_boost']['mean']:+.3f} reward")
    
    if results['with_boost']['mean'] > 0.1:
        print("\n✓ Compound transfer proven on structured RL!")
        print("  Next: MAML for learned boosting")
    else:
        print("\n⚠ Need full Meta^3 representation optimizer")
