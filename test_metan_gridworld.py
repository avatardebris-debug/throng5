"""
Test Meta^N architecture on GridWorld at scale.
Compare to simple baseline to see if Meta^N learning works.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
import time


def test_metan_gridworld(n_neurons=4096, n_episodes=100, verbose=True):
    """
    Test Meta^N pipeline on GridWorld.
    
    Args:
        n_neurons: Number of neurons in Meta^0 layer (use power of 2 for efficiency)
        n_episodes: Number of training episodes
        verbose: Print progress
    """
    if verbose:
        print("="*60)
        print("Meta^N GridWorld Test")
        print("="*60)
        print(f"  Neurons (Meta^0): {n_neurons}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Architecture: Meta^0 through Meta^5")
        print()
    
    # Create environment
    env = GridWorldAdapter()
    
    # Create Meta^N pipeline
    # GridWorld: 2 inputs (x, y), 4 outputs (up, down, left, right)
    if verbose:
        print("Creating Meta^N pipeline...")
    
    start_time = time.time()
    pipeline = MetaNPipeline.create_default(
        n_neurons=n_neurons,
        n_inputs=2,
        n_outputs=4,
        include_llm=False
    )
    create_time = time.time() - start_time
    
    if verbose:
        print(f"  Pipeline created in {create_time:.2f}s")
        print(f"  Layers: {len(pipeline.stack.layers)}")
        print()
    
    # Training loop
    episode_returns = []
    episode_lengths = []
    losses = []
    
    if verbose:
        print("Training...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Step through pipeline
            result = pipeline.step(
                input_data=obs,
                reward=episode_reward,  # Cumulative reward so far
                episode_return=episode_reward
            )
            
            # Get action from output
            output = result['output']
            action = np.argmax(output)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Track loss
            if 'loss' in result:
                episode_loss.append(result['loss'])
        
        episode_returns.append(episode_reward)
        episode_lengths.append(steps)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        if verbose and (episode + 1) % 10 == 0:
            recent_returns = episode_returns[-10:]
            recent_lengths = episode_lengths[-10:]
            avg_return = np.mean(recent_returns)
            avg_length = np.mean(recent_lengths)
            avg_loss = np.mean(losses[-10:]) if losses else 0.0
            
            print(f"  Episode {episode+1}/{n_episodes}: "
                  f"return={episode_reward:.2f}, "
                  f"avg_return={avg_return:.2f}, "
                  f"avg_length={avg_length:.1f}, "
                  f"loss={avg_loss:.4f}")
    
    # Evaluation
    if verbose:
        print(f"\n{'='*60}")
        print("Evaluation (20 episodes, no exploration):")
        print(f"{'='*60}")
    
    successes = 0
    eval_returns = []
    
    for ep in range(20):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            result = pipeline.step(input_data=obs, reward=0.0)
            output = result['output']
            action = np.argmax(output)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        eval_returns.append(episode_reward)
        if env.pos == env.goal:
            successes += 1
    
    # Results
    early_returns = np.mean(episode_returns[:10])
    late_returns = np.mean(episode_returns[-10:])
    
    if verbose:
        print(f"\nTraining progress:")
        print(f"  Early (1-10): {early_returns:.3f}")
        print(f"  Late ({n_episodes-9}-{n_episodes}): {late_returns:.3f}")
        print(f"  Improvement: {late_returns - early_returns:+.3f}")
        
        print(f"\nEvaluation results:")
        print(f"  Success rate: {successes}/20 = {successes*5}%")
        print(f"  Avg return: {np.mean(eval_returns):.3f}")
        
        print(f"\n{'='*60}")
        if successes >= 15:  # 75% success
            print("✓ SUCCESS: Meta^N learned GridWorld!")
        elif successes >= 5:  # 25% success
            print("⚠ PARTIAL: Some learning, needs tuning")
        else:
            print("✗ FAILURE: Meta^N did not learn")
        print("="*60)
    
    return {
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'losses': losses,
        'eval_success_rate': successes / 20,
        'eval_avg_return': np.mean(eval_returns),
        'improvement': late_returns - early_returns
    }


if __name__ == "__main__":
    # Test with 4096 neurons (power of 2, close to 5000)
    print("\n" + "="*60)
    print("Testing Meta^N Architecture")
    print("="*60)
    print("\nUsing 4096 neurons (2^12, close to 5K baseline)")
    print("This tests if Meta^N's STDP/holographic learning works for RL")
    print()
    
    results = test_metan_gridworld(n_neurons=4096, n_episodes=100, verbose=True)
    
    print("\n" + "="*60)
    print("Comparison to Simple Baseline (5000 neurons):")
    print("="*60)
    print(f"  Simple baseline: 100% success, 0.930 return")
    print(f"  Meta^N (4096):   {results['eval_success_rate']*100:.0f}% success, {results['eval_avg_return']:.3f} return")
    print()
    
    if results['eval_success_rate'] >= 0.75:
        print("✓ Meta^N performs comparably to simple baseline!")
    elif results['eval_success_rate'] >= 0.25:
        print("⚠ Meta^N shows learning but underperforms baseline")
        print("  May need: more episodes, tuned STDP, or larger network")
    else:
        print("✗ Meta^N not learning effectively")
        print("  Needs: STDP tuning, reward-modulated plasticity, or different approach")
    
    print("="*60)
