"""
Test: Throng4 on GridWorld

Verify that the dual-head ANN can learn GridWorld.
Compare to Throng3/3.5 baselines.
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.pipeline import SimplePipeline
from throng4.learning.dqn import DQNConfig
from throng3.environments.gym_envs import GridWorldAdapter


def run_episode(env, pipeline, max_steps=100, explore=True):
    """Run one episode."""
    state = env.reset()  # Returns normalized [x/(size-1), y/(size-1)]
    total_reward = 0.0
    steps = 0
    
    for _ in range(max_steps):
        # Select action
        action = pipeline.select_action(state, explore=explore)
        
        # Environment step
        next_state, reward, done, _ = env.step(action)
        
        # Update
        errors = pipeline.update(state, action, reward, next_state, done)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    return total_reward, steps


def test_gridworld_learning():
    """Test dual-head ANN on GridWorld."""
    print("=" * 70)
    print("THRONG4 GRIDWORLD TEST")
    print("=" * 70)
    
    # Create environment (5×5 GridWorld)
    env = GridWorldAdapter(size=5)
    
    # Create pipeline
    config = DQNConfig(
        learning_rate=0.01,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        aux_loss_weight=0.1
    )
    
    pipeline = SimplePipeline(
        n_inputs=2,   # Normalized (x, y) position
        n_outputs=4,  # up/down/left/right
        n_hidden=64,
        config=config
    )
    
    print(f"Environment: {env.size}×{env.size} GridWorld")
    print(f"State dim: {2} (normalized x,y), Actions: {4}")
    print(f"ANN parameters: {pipeline.ann.get_num_parameters()}")
    print(f"Config: lr={config.learning_rate}, ε={config.epsilon}, "
          f"aux_weight={config.aux_loss_weight}")
    print()
    
    # Training
    n_episodes = 200
    rewards = []
    
    print("Training...")
    for episode in range(n_episodes):
        reward, steps = run_episode(env, pipeline, explore=True)
        rewards.append(reward)
        
        # Print progress
        if (episode + 1) % 20 == 0:
            recent_mean = np.mean(rewards[-20:])
            stats = pipeline.get_stats()
            print(f"Episode {episode + 1:3d}: "
                  f"reward={reward:6.3f}, "
                  f"mean_20={recent_mean:6.3f}, "
                  f"ε={stats['epsilon']:.3f}, "
                  f"TD_err={stats['mean_td_error']:.3f}, "
                  f"R_err={stats['mean_reward_error']:.3f}")
    
    # Evaluation (greedy)
    print("\nEvaluation (greedy, 20 episodes)...")
    eval_rewards = []
    for _ in range(20):
        reward, steps = run_episode(env, pipeline, explore=False)
        eval_rewards.append(reward)
    
    mean_eval = np.mean(eval_rewards)
    std_eval = np.std(eval_rewards)
    
    print(f"Eval mean reward: {mean_eval:.3f} ± {std_eval:.3f}")
    print(f"Success rate: {np.mean([r > 0 for r in eval_rewards]) * 100:.1f}%")
    
    # Compare to baselines
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINES")
    print("=" * 70)
    print("Throng3 (SNN + Q-table):     ~0.25 (25% success, timing issues)")
    print("Throng3.5 (Tabular Q):       ~0.86 (86% success, standalone)")
    print(f"Throng4 (Dual-head ANN):     {mean_eval:.2f} ({mean_eval*100:.0f}% success)")
    print()
    
    # Check if we beat the baseline
    if mean_eval > 0.5:
        print("[SUCCESS] Throng4 learns GridWorld!")
    else:
        print("[WARNING] Performance below expected (>0.5)")
    
    # Analyze reward prediction
    print("\n" + "=" * 70)
    print("REWARD PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Sample some states and check reward predictions
    test_states = [env.reset() for _ in range(10)]
    for i, state in enumerate(test_states[:3]):
        q_vals = pipeline.get_q_values(state)
        r_pred = pipeline.get_reward_prediction(state)
        print(f"State {i}: Q_max={np.max(q_vals):.3f}, R_pred={r_pred:.3f}")
    
    stats = pipeline.get_stats()
    print(f"\nMean reward prediction error: {stats['mean_reward_error']:.3f}")
    print("(Lower is better — auxiliary task is learning)")
    
    return mean_eval


if __name__ == "__main__":
    try:
        mean_reward = test_gridworld_learning()
        
        if mean_reward > 0.5:
            print("\n[PASS] Throng4 dual-head architecture validated!")
            sys.exit(0)
        else:
            print("\n[WARNING] Needs tuning")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
