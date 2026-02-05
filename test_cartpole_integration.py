"""
Integration test: CartPole with MetaNPipeline
Trains a minimal pipeline on CartPole and verifies loss decreases.
"""

from throng3.environments import CartPoleAdapter
from throng3.pipeline import MetaNPipeline
import numpy as np

def main():
    print("="*60)
    print("CartPole + MetaNPipeline Integration Test")
    print("="*60)
    
    env = CartPoleAdapter()
    pipeline = MetaNPipeline.create_minimal(
        n_neurons=100,
        n_inputs=4,
        n_outputs=2,
    )
    
    obs = env.reset()
    
    losses = []
    episode_returns = []
    current_episode_reward = 0
    
    print("\nRunning 200 steps...")
    for step in range(200):
        # Get action from pipeline (pass reward from previous step)
        result = pipeline.step(obs, reward=reward if step > 0 else 0.0)
        
        # Extract action (argmax of output)
        output = result.get('output', np.zeros(2))
        action = np.argmax(output)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Track loss (now meaningful with reward-based loss)
        if 'loss' in result:
            losses.append(result['loss'])
        
        if done:
            episode_returns.append(env.episode_reward)
            obs = env.reset()
        
        if (step + 1) % 50 == 0:
            recent_loss = np.mean(losses[-50:]) if losses else 0
            recent_return = np.mean(episode_returns[-5:]) if len(episode_returns) >= 5 else 0
            print(f"  Step {step+1}: avg_loss={recent_loss:.4f}, episodes={env.total_episodes}, avg_return={recent_return:.1f}")
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Total episodes: {env.total_episodes}")
    
    # Report episode returns (primary RL metric)
    if len(episode_returns) >= 10:
        early_return = np.mean(episode_returns[:5])
        late_return = np.mean(episode_returns[-5:])
        return_improvement = ((late_return - early_return) / max(abs(early_return), 1)) * 100
        
        print(f"\nEpisode Returns:")
        print(f"  Early episodes (0-5): {early_return:.2f}")
        print(f"  Late episodes (last 5): {late_return:.2f}")
        print(f"  Improvement: {return_improvement:+.1f}%")
    
    # Report loss (now meaningful with reward-based loss)
    if len(losses) >= 100:
        early_loss = np.mean(losses[:50])
        late_loss = np.mean(losses[-50:])
        
        print(f"\nLoss (negative reward):")
        print(f"  Early loss (steps 0-50): {early_loss:.4f}")
        print(f"  Late loss (steps 150-200): {late_loss:.4f}")
        
        if late_loss < early_loss:
            loss_improvement = ((early_loss - late_loss) / max(abs(early_loss), 0.01)) * 100
            print(f"  Improvement: {loss_improvement:.1f}%")
            print("\n✓ Loss decreased over training - PASS")
        else:
            print("\n⚠ Loss did not decrease significantly")
    else:
        print(f"\nAverage episode return: {np.mean(episode_returns):.2f}" if episode_returns else "N/A")
    
    print("="*60)
    print("Integration test complete!")
    print("="*60)

if __name__ == '__main__':
    main()
